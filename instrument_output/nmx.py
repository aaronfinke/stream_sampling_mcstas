"""
ESS NMX output: NeXus streaming layout from a json template, three 1280 x 1280 detector
panels, event times wrapped to the 14 Hz ESS period (in ns), NeXus geometry from McStas.
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import mcstas_to_nexus_geometry
from mcstas_geometry_xml import read_mcstas_geometry_xml

from .base import OutputFormat
from .julia import jl

mpl.rcParams["animation.embed_limit"] = 500

NMX_JSON_PATH = Path(__file__).resolve().parent.parent / "nmx.json"
NMX_PERIOD = 1/14       # seconds


class NMXOutput(OutputFormat):
    name = "nmx"
    default_output_file = "sampling_output.h5"

    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        super().__init__(args, logger)
        self._geometry = {}

    def sample(self, filenames: Sequence[str], n: int) -> List[np.ndarray]:
        """Samples panels 0..range-1 (datasets Detector_<i>_event_signal_dat_list_p_x_y_n_id_t)"""
        return do_sampling(self.args, list(filenames), self.logger, range=self.option("range", 3), n=n)

    def sample_datasets(self, filename: str, datasets: List[str], n: int) -> List[np.ndarray]:
        """Samples the given event datasets of one file (e.g. one configuration of several)"""
        if self.option("use_mask"):
            # XXX include mask support... somehow.
            self.logger.warning("Masks not currently supported.")
        eval_statement = f'sample_all_frames_generic("{filename}", "{datasets}", {n}, AlgWRSWRSKIP())'
        self.logger.info(f"Running {eval_statement} ...")
        t1 = time.perf_counter()
        sampled_jl = jl.sample_all_frames_generic(filename, datasets, n, jl.AlgWRSWRSKIP())
        self.logger.info(f"... done. Took {time.perf_counter() - t1:.2f} s.")
        return _log_panels(redistribute_sampling(sampled_jl.to_numpy()), self.logger)

    def geometry(self, input_file: str | Path):
        """McStas component poses and pixel layout (cached per input file)"""
        key = str(input_file)
        if key not in self._geometry:
            xml = self.option("xml")
            if xml:
                self._geometry[key] = mcstas_to_nexus_geometry.load_xml_geometry(
                    Path(xml), logger=self.logger, mcstas_file=Path(input_file))
            else:
                self._geometry[key] = read_mcstas_geometry_xml(Path(input_file))
        return self._geometry[key]

    def write(self, output_file: Path, sampled: List[np.ndarray], input_files: Sequence[str],
              simulation_detector_names: dict[int, str] | None = None) -> None:
        json_file = self.option("json_file", NMX_JSON_PATH)
        if not Path(json_file).exists():
            raise FileNotFoundError(f"File {json_file} not found.")
        geometry = self.geometry(input_files[0])
        create_nexus_file(args=self.args,
                          output_file=output_file,
                          sampled=sampled,
                          json_template=json_file,
                          instrument_xml=self.instrument_xml(input_files[0]),
                          logger=self.logger)
        kwargs = {"simulation_detector_names": simulation_detector_names} if simulation_detector_names else {}
        mcstas_to_nexus_geometry.insert_geometry_into_nexus(geometry, Path(output_file), self.logger, **kwargs)

    def plots(self, sampled: List[np.ndarray], folder: Path, prefix: str = "") -> None:
        tof_bins = get_tof_bins(self.args, sampled, logger=self.logger)
        self.logger.info("Generating histogram...")
        histo = make_histogram(sampled, tof_bins, logger=self.logger)
        self.logger.info("Saving sum plot...")
        sum_plot(histo, folder, filename=f"{prefix}allbins.png")
        self.logger.info("Generating animation...")
        make_animation(self.args, histo, tof_bins, folder, filename=f"{prefix}3panel_animation.gif")


def _log_panels(redistributed: List[np.ndarray], logger: logging.Logger) -> List[np.ndarray]:
    totsamp = 0
    for i, sample in enumerate(redistributed):
        logger.info(f"Number of samples in panel {i}: {len(sample)}")
        totsamp += len(sample)
    logger.debug(f"Total samples: {totsamp}")
    return redistributed


def write_template_to_nexus(fp: h5py.File | h5py.Dataset | h5py.Group, entry: Dict, logger:logging.Logger):
    def _set_attributes(target: h5py.Group | h5py.Dataset, attrs, logger: logging.Logger):
        match attrs:
            case None:
                return
            case str():
                # store raw string under a generic key
                target.attrs["text"] = attrs
            case list():
                for item in attrs:
                    try:
                        if isinstance(item, dict) and "name" in item and "values" in item:
                            target.attrs[item["name"]] = item["values"]
                        else:
                            logger.warning(f"Unsupported attribute list item (expect dict with 'name'/'values'): {item!r}")
                    except Exception as e:
                        logger.error(f"Failed to set attribute {item!r}: {e}")
            case dict():
                try:
                    for k, v in attrs.items():
                        target.attrs[k] = v
                except Exception as e:
                    logger.error(f"Failed to set dict attributes {attrs!r}: {e}")
            case _:
                logger.warning(f"Unsupported attributes type: {type(attrs)}")

    if "group" in entry.get("type",""):
        # create group
        new_group = fp.create_group(entry["name"])
        # create attributes
        _set_attributes(new_group, entry.get("attributes"), logger)
        for dset in entry["children"]:
            write_template_to_nexus(new_group, dset, logger)

    elif "dataset" in entry.get("module",""):
        data = entry["config"].get("values")
        dset = fp.create_dataset(entry["config"].get("name"), data=data)
        _set_attributes(dset, entry.get("attributes"), logger)

def redistribute_sampling(sampled):
    pixids = sampled["f0"]
    d0 = sampled[pixids < 1280 * 1280]
    d1 = sampled[(1280 * 1280 <= pixids) & (pixids < 2 * 1280 * 1280)]
    d2 = sampled[(pixids >= 2 * 1280 * 1280)]
    return [d0, d1, d2]

def do_sampling(args:argparse.Namespace, filenames: List[str], logger:logging.Logger, range: int = 3, n: int = 10 ** 5) -> List[np.ndarray]:
    rangeval = f"0:{range - 1}"
    multifile = len(filenames) > 1
    if multifile:
        filenames_julia = "[" + ", ".join(f'"{f}"' for f in filenames) + "]"
        if args.use_mask:
            logger.info("Using mask around beam center.")
            eval_statement = f'multifile_sample_all_frames_mask({filenames_julia}, {rangeval}, {n}, AlgWRSWRSKIP())'
        else:
            eval_statement = f'multifile_sample_all_frames({filenames_julia}, {rangeval}, {n}, AlgWRSWRSKIP())'
    else:
        filename = filenames[0]
        if args.use_mask:
            logger.info("Using mask around beam center.")
            eval_statement = f'sample_all_frames_mask("{filename}", {rangeval}, {n}, AlgWRSWRSKIP())'
        else:
            eval_statement = f'sample_all_frames("{filename}", {rangeval}, {n}, AlgWRSWRSKIP())'
    logger.info(f"Running {eval_statement} ...")
    t1 = time.perf_counter()
    sampled_jl = jl.seval(eval_statement)
    logger.info(f"... done. Took {time.perf_counter() - t1:.2f} s.")
    sampled = sampled_jl.to_numpy()
    return _log_panels(redistribute_sampling(sampled), logger)


def load_json_dict(json_path):
    if not Path(json_path).exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    return json.load(open(json_path))

def create_nexus_file(args: argparse.Namespace, output_file: str|Path, sampled: List[np.ndarray],
                      json_template: str, instrument_xml: str, logger:logging.Logger):

    metadata_dict = load_json_dict(json_template)
    subtopics = metadata_dict["children"][0]

    with h5py.File(output_file, "w") as fp:
        write_template_to_nexus(fp, subtopics, logger)
        instrument_xml_group: h5py.Group = fp.create_group('entry/instrument/instrument_xml')
        instrument_xml_group.create_dataset('data', data=instrument_xml, dtype=h5py.string_dtype(length=len(instrument_xml)))
        instrument_xml_group.create_dataset('description', data='XML contents of the instrument IDF', dtype=h5py.string_dtype(length=34))
        instrument_xml_group.create_dataset('type', data='text/xml', dtype=h5py.string_dtype(length=8))
        for index in range(3):
            datagroup: h5py.Group = fp[f"entry/instrument/detector_panel_{index}/data"]
            datagroup.create_dataset("cue_index", data=0)
            datagroup.create_dataset("cue_timestamp_0", data=0)
            sampled_index = sampled[index]
            event_ids = sampled_index["f0"].astype(int)
            toas = sampled_index["f1"].copy()
            tofs = toas if args.no_wrap_tofs else toas % NMX_PERIOD        # wrap the toas in the NMX period
            tofs *= 1e9   # set to ns
            datagroup.create_dataset("event_id", data=event_ids)
            t_offset = datagroup.create_dataset("event_time_offset", data=tofs.astype(int))
            t_offset.attrs["units"] = "ns"

            # placesetters for figuring out later- how to make this look more like "events per pulse"
            datagroup.create_dataset(
                "event_index", data=np.zeros(len(event_ids) // 30), dtype=int
            )
            evt_time_zero = datagroup.create_dataset(
                "event_time_zero", data=np.zeros(len(event_ids) // 30), dtype=int
            )
            evt_time_zero.attrs["units"] = "ns"


def get_tof_bins(args: argparse.Namespace, results: List[np.ndarray], logger:logging.Logger, n=50):
    """
    Get bins from tofs. These will probably be wrapped to the NMX pulse (14 Hz).
    """
    nmx_period = 0 if args.no_wrap_tofs else NMX_PERIOD
    tofmin = np.inf
    tofmax = -1
    for i,result in enumerate(results):
        print(type(result))
        print(result)
        toas = result["f1"]
        sample_max  = toas.max()
        if sample_max + nmx_period > 2*NMX_PERIOD + 0.001:
            logger.warning(f"Panel {i} has tof_max of {sample_max + nmx_period}")
        sample_min = toas.min()
        tofmin = sample_min if sample_min < tofmin else tofmin
        tofmax = sample_max if sample_max > tofmax else tofmax
    tofs = np.linspace(tofmin, tofmax, n)
    logger.info(f"tof_min: {tofmin}, tof_max: {tofmax}")
    return tofs

def get_tof_bins_from_hdf5(args: argparse.Namespace, results: List[np.ndarray], logger:logging.Logger, n=50):
    """
    Get bins from tofs. These will probably be wrapped to the NMX pulse (14 Hz).
    """
    nmx_period = 0 if args.no_wrap_tofs else NMX_PERIOD
    tofmin = np.inf
    tofmax = -1
    for i,result in enumerate(results):
        toas = result[:,1] / 1e9
        sample_max  = toas.max()
        if sample_max + nmx_period > 0.145:
            logger.warning(f"Panel {i} has tof_max of {sample_max + nmx_period}")
        sample_min = toas.min()
        tofmin = sample_min if sample_min < tofmin else tofmin
        tofmax = sample_max if sample_max > tofmax else tofmax
    tofs = np.linspace(tofmin, tofmax, n)
    logger.info(f"tof_min: {tofmin + nmx_period}, tof_max: {tofmax + nmx_period}")
    return tofs



def make_histogram(data_list: List, time_bin_edges: np.ndarray, logger:logging.Logger) -> List[np.ndarray]:
    # Define time binning parameters
    n_time_bins = len(time_bin_edges)

    # Now create pixel counts per time bin
    n_pixels = 1280 * 1280 * 3
    pixel_counts_per_bin = np.zeros((n_pixels, n_time_bins))

    for data in data_list:
        pixids = data["f0"].astype(int)
        times = data["f1"]

        # Assign each event to a time bin
        time_bin_indices = np.digitize(times, time_bin_edges) - 1
        # Clip to ensure all indices are within valid range
        time_bin_indices = np.clip(time_bin_indices, 0, n_time_bins - 1)

        np.add.at(pixel_counts_per_bin, (pixids, time_bin_indices), 1)
    # np.save("ouput.npy", pixel_counts_per_bin)
    # Reshape to detector coordinates if needed
    data_panel_0 = pixel_counts_per_bin[: (1280 * 1280), :]
    data_panel_1 = pixel_counts_per_bin[(1280 * 1280) : (1280 * 1280 * 2), :]
    data_panel_2 = pixel_counts_per_bin[(1280 * 1280 * 2) :, :]

    data_panel_0 = data_panel_0.reshape((1280, 1280, n_time_bins))
    data_panel_1 = data_panel_1.reshape((1280, 1280, n_time_bins))
    data_panel_2 = data_panel_2.reshape((1280, 1280, n_time_bins))

    data_panels = [data_panel_0, data_panel_1, data_panel_2]

    # detector_time_series = pixel_counts_per_bin.reshape(n_time_bins, 1280, 1280)
    # detector_time_series_corrected = detector_time_series.transpose((1, 2, 0))
    for i, data in enumerate(data_panels):
        logger.debug(f"Shape of histogram {i}: {data.shape}")
    # np.save("data.npy", data_panels)
    return data_panels

def make_histogram_from_hdf5(data_list: List, time_bin_edges: np.ndarray, logger:logging.Logger) -> List[np.ndarray]:
    # Define time binning parameters
    n_time_bins = len(time_bin_edges)

    # Now create pixel counts per time bin
    n_pixels = 1280 * 1280 * 3
    pixel_counts_per_bin = np.zeros((n_pixels, n_time_bins))

    for data in data_list:
        pixids = data[:, 0].astype(int)
        times = data[:, 1] / 1e9  # data stored in ns

        # Assign each event to a time bin
        time_bin_indices = np.digitize(times, time_bin_edges) - 1
        # Clip to ensure all indices are within valid range
        time_bin_indices = np.clip(time_bin_indices, 0, n_time_bins - 1)

        np.add.at(pixel_counts_per_bin, (pixids, time_bin_indices), 1)
    # np.save("ouput.npy", pixel_counts_per_bin)
    # Reshape to detector coordinates if needed
    data_panel_0 = pixel_counts_per_bin[: (1280 * 1280), :]
    data_panel_1 = pixel_counts_per_bin[(1280 * 1280) : (1280 * 1280 * 2), :]
    data_panel_2 = pixel_counts_per_bin[(1280 * 1280 * 2) :, :]

    data_panel_0 = data_panel_0.reshape((1280, 1280, n_time_bins))
    data_panel_1 = data_panel_1.reshape((1280, 1280, n_time_bins))
    data_panel_2 = data_panel_2.reshape((1280, 1280, n_time_bins))

    data_panels = [data_panel_0, data_panel_1, data_panel_2]

    # detector_time_series = pixel_counts_per_bin.reshape(n_time_bins, 1280, 1280)
    # detector_time_series_corrected = detector_time_series.transpose((1, 2, 0))
    for i, data in enumerate(data_panels):
        logger.debug(f"Shape of histogram {i}: {data.shape}")
    # np.save("data.npy", data_panels)
    return data_panels


def _sigma_limits(
    arr: np.ndarray, low_sigma: float = 0.0, high_sigma: float = 2.0
) -> tuple[float, float]:
    vals = arr[np.isfinite(arr) & (arr > 0)]
    if vals.size == 0:
        return 1e-8, 1.0
    mu = float(np.mean(vals))
    sigma = float(np.std(vals))
    vmin = max(vals.min(), mu - low_sigma * sigma)
    vmax = min(vals.max(), mu + high_sigma * sigma)
    if vmin >= vmax:
        vmax = vmin * 10.0
    return vmin, vmax

def sum_plot(histo,folder: Path = Path.cwd(),filename="allbins.png"):
    processed = [dset.sum(axis=2) for dset in histo]

    combined = np.concatenate([d[d > 0].ravel() for d in processed if d.size > 0])
    vmin, vmax = _sigma_limits(combined, low_sigma=0.0, high_sigma=2.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.02)

    ims = []
    for i, (ax, data) in enumerate(zip(axes, processed)):
        im = ax.imshow(data, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"Panel {i} - All Frames")
        if i == 0:
            ax.set_ylabel("Y pixel")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("X pixel")
        ims.append(im)

    # Single shared colorbar (remove per-axis plt.colorbar calls)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.set_label("Intensity")
    fig.savefig(str(folder / filename), dpi=600, bbox_inches="tight")
    plt.close()

def make_animation(
    args:argparse.Namespace,
    datasets: List[np.ndarray],
    tofs: np.ndarray,
    folder: Path = Path.cwd(),
    fps=5,
    low_sigma=0.0,
    high_sigma=2.0,
    filename: str="3panel_animation.gif"
):
    nmx_period = 0 if args.no_wrap_tofs else NMX_PERIOD
    # Compute global limits across all panels/frames
    combined = np.concatenate([d[d > 0].ravel() for d in datasets if d.size > 0])
    vmin, vmax = _sigma_limits(combined, low_sigma=low_sigma, high_sigma=high_sigma)

    # Optionally clip data to these limits (prevents outliers dominating colorbar)
    # processed_data = [np.clip(np.where(d <= 0, vmin, d), vmin, vmax) for d in datasets]
    processed_data = datasets

    # Create figure with three subplots - closer spacing
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.02)  # Reduce space between subplots

    # Initialize all three panels
    ims = []
    for i, (ax, data) in enumerate(zip(axes, processed_data)):
        im = ax.imshow(
            data[:, :, 0],
            cmap="viridis",
            origin="lower",
            vmin=vmin,
            vmax=vmax
            # norm=LogNorm(vmin=global_vmin, vmax=global_vmax),
        )
        ax.set_title(f"Panel {i} - Frame: 0 / {data.shape[2] - 1}")
        ax.set_xlabel("X pixel")

        # Only show Y label on leftmost panel
        if i == 0:
            ax.set_ylabel("Y pixel")
        else:
            ax.set_yticklabels([])  # Remove Y tick labels from other panels

        ims.append(im)

    # Add a single colorbar for all panels
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.set_label("Intensity")

    def animate(frame):
        """Animation function that updates all three panels for each frame"""
        for i, (im, data) in enumerate(zip(ims, processed_data)):
            im.set_array(data[:, :, frame])
            if i == 1:
                axes[i].set_title(
                    f"Panel {i} - Frame: {frame} / {data.shape[2] - 1}, tof {(tofs[frame]) * 1e3:.1f} ms"
                )
            else:
                axes[i].set_title(f"Panel {i} - Frame: {frame} / {data.shape[2] - 1}")
        return ims

    nframes = datasets[0].shape[2]

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=nframes,
        interval=int((1 / fps) * 1000),  # 200ms between frames
        blit=True,
        repeat=True,
    )

    # To save as GIF (uncomment if needed):
    anim.save(
        str(folder / filename),
        writer='pillow',
        dpi=300,
        fps=5,
    )

    plt.close()
