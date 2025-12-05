import h5py
import json
import numpy as np
import time
import argparse
from pathlib import Path
import sys
import logging
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List, Dict

import mcstas_to_nexus_geometry
from mcstas_geometry_xml import read_mcstas_geometry_xml
from juliacall import Main as jl

julia_path = Path(__file__).parent
jl.include(str(julia_path / "sampling_toNexus.jl"))

mpl.rcParams["animation.embed_limit"] = 500

NMX_JSON_PATH = "/project/project_465002030/streaming_mcstas_nexus.json"
NMX_PERIOD = 1/14       # seconds

def _get_logger(args: argparse.Namespace) -> logging.Logger:
    # Parse arguments for logging level
    level = logging.DEBUG if args.verbose else logging.INFO
    # Initialize logger
    cur_file_name = Path(__file__).stem
    logger = logging.getLogger(cur_file_name)
    # Initialize handler and formatter
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


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

def do_sampling(args:argparse.Namespace,filename: str, logger:logging.Logger, range: int = 3, n: int = 10 ** 5) -> List[np.ndarray]:
    rangeval = f"0:{range - 1}"
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
    redistributed = redistribute_sampling(sampled)
    totsamp = 0
    for i,sample in enumerate(redistributed):
        logger.info(f"Number of samples in panel {i}: {len(sample)}")
        totsamp += len(sample)
    logger.debug(f"Total samples: {totsamp}")
    return redistributed


def load_json_dict(json_path):
    if not Path(json_path).exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    return json.load(open(json_path))


def create_nexus_file(args: argparse.Namespace, output_file: str, sampled: List[np.ndarray], json_template: str, logger:logging.Logger):
    
    metadata_dict = load_json_dict(json_template)
    subtopics = metadata_dict["children"][0]

    with h5py.File(output_file, "w") as fp:
        write_template_to_nexus(fp, subtopics, logger)
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
        if sample_max + nmx_period > 0.145:
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

def sum_plot(histo,folder: Path = Path.cwd(),):
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
    fig.savefig(str(folder / "allbins.png"))

def make_animation(
    args:argparse.Namespace,
    datasets: List[np.ndarray],
    tofs: np.ndarray,
    folder: Path = Path.cwd(),
    fps=5,
    low_sigma=0.0,
    high_sigma=2.0,
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
                    f"Panel {i} - Frame: {frame} / {data.shape[2] - 1}, tof {(tofs[frame] + nmx_period) * 1e3:.1f} ms"
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
        folder / "3panel_animation.gif",
        writer='pillow',
        fps=5,
    )

def animation_only(args, logger: logging.Logger):
    """Generates the animations only, using sampling output."""
    datas = []
    try:
        logger.info(f"Loading file {args.input_file}...")
        with h5py.File(args.input_file) as fp:
            for i in range(3):
                data = fp[f"entry/instrument/detector_panel_{i}/data"]
                pixids = data["event_id"][:]
                tofs = data["event_time_offset"][:]
                datas.append(np.column_stack((pixids,tofs)))
    except (KeyError, FileNotFoundError) as e:
        logger.error(f"Error in reading HDF5: {e}")
        raise
    sampled = [np.array(x) for x in datas]
    tof_bins = get_tof_bins_from_hdf5(args,sampled,logger=logger)
    print(tof_bins)
    logger.info("Generating histogram...")
    histo = make_histogram_from_hdf5(sampled, tof_bins,logger=logger)
    del sampled
    sum_plot(histo, Path(args.input_file).parent)

    logger.info("Generating animation...")
    make_animation(args,histo, tof_bins, Path(args.input_file).parent)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample HDF5 data and create NeXus file"
    )
    parser.add_argument(
        "-i",
        "--input_file", 
        type=str, 
        required=True,
        help="Input HDF5 file path")
    
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Output NeXus file path",
        default="sampling_output.h5",
        required=False,
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=100_000,
        help="Number of samples (default: 100_000)",
    )
    parser.add_argument(
        "-j",
        "--json-file",
        type=str,
        help="Path to json template",
        default=NMX_JSON_PATH,
    )
    parser.add_argument(
        "-x",
        "--xml",
        type=str,
        help="Mantid xml geometry file (if the one in McStas file is not to be used)"
    )
    parser.add_argument(
        "--use-mask",
        action="store_true",
        help="Use a mask around the beam center. Useful for ensuring direct beam is not oversampled."
    )
    parser.add_argument(
        "--do-histogram",
        action="store_true",
        help="Generate histogram data and animation",
    )
    parser.add_argument(
        "--animation-only",
        action="store_true",
        help="Generate animation only (after sampling)",
    )
    parser.add_argument(
        "--range",
        type=int,
        default=3,
        help="Number of panels (default: 3)"
    )
    parser.add_argument(
        "--no-wrap-tofs",
        action="store_true",
        help="Don't wrap tof values to ESS pulse length (14 Hz)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = _get_logger(args)
    if args.animation_only:
        animation_only(args,logger)
        sys.exit(0)

    if not Path(args.json_file).exists():
        raise FileNotFoundError(f"File {args.json_file} not found.")
    sampled = do_sampling(args=args, filename=args.input_file, logger=logger, range=args.range, n=args.n_samples)

    if Path(args.output_file).exists():
        logger.warning(f"Overwriting {args.output_file}...")
    if args.no_wrap_tofs:
        logger.warning(f"Not wrapping tofs to NMX pulse length.")
    if args.xml:
        logger.info(f"Using geometry from xml file {args.xml}...")
        geometry = mcstas_to_nexus_geometry.load_xml_geometry(Path(args.xml),logger=logger)
    else:
        geometry = read_mcstas_geometry_xml(Path(args.input_file))
    create_nexus_file(args, args.output_file, sampled, args.json_file, logger=logger)
    mcstas_to_nexus_geometry.insert_geometry_into_nexus(geometry,Path(args.output_file),logger)
    if args.do_histogram:
        tof_bins = get_tof_bins(args,sampled,logger=logger)
        logger.info("Generating histogram...")
        histo = make_histogram(sampled, tof_bins,logger=logger)
        logger.info("Saving sum plot...")
        sum_plot(histo, Path(args.output_file).parent)
        logger.info("Generating animation...")
        make_animation(args,histo, tof_bins, Path(args.output_file).parent)


if __name__ == "__main__":
    main()
