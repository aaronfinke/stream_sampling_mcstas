
import scipp
import scippnexus
import h5py
import json
import numpy as np
import time
import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import List, Any

from juliacall import Main as jl
jl.include("samping_toNexus.jl")

mpl.rcParams["animation.embed_limit"] = 500

NMX_JSON_PATH = "/Users/aaronfinke/nmx_mcstas/nmx-dynamic.json"

def write_to_nexus(fp:h5py.File, entry: List[Any]):
    if entry.get("type") == "group":
        # create group
        new_group = fp.create_group(entry["name"])
        # create attributes
        if entry.get("attributes") is not None:
            for attribute in entry.get("attributes"):
                new_group.attrs[attribute["name"]] = attribute["values"]
        for dset in entry["children"]:
            write_to_nexus(new_group, dset)

    elif entry.get("module") == "dataset":
        data = entry["config"].get("values")
        dset = fp.create_dataset(entry["config"].get("name"), data=data)
        if entry.get("attributes") is not None:
            for attribute in entry.get("attributes"):
                dset.attrs[attribute["name"]] = attribute["values"]

def do_sampling(filename:str, range:int = 3, n:int = 10^5) -> List[np.ndarray]:
    rangeval = f"0:{range-1}"
    eval_statement = f'sample_frames("{filename}", {rangeval}, {n}, AlgWRSWRSKIP())'
    print(f"Running {eval_statement} ...")
    t1 = time.perf_counter()
    sampled_jl = jl.seval(eval_statement)
    print(f"... done. Took {time.perf_counter()-t1:.2f} s.")
    sampled = [x.to_numpy() for x in sampled_jl]
    # np.save("output.py", sampled)
    return sampled

def load_json_dict():
    if not Path(NMX_JSON_PATH).exists():
        raise FileNotFoundError(f"File not found: {NMX_JSON_PATH}")
    return json.load(open(NMX_JSON_PATH))

def create_nexus_file(output_file: str, sampled: List[np.ndarray]):
    metadata_dict = load_json_dict()
    subtopics = metadata_dict["children"][0]

    with h5py.File(output_file, "w") as fp:
        write_to_nexus(fp, subtopics)
        for index in range(3):
            datagroup = fp[f'entry/instrument/detector_panel_{index}/data']
            datagroup.create_dataset('cue_index',data=0)
            datagroup.create_dataset('cue_timestamp_0',data=0)
            sampled_index = sampled[index]
            event_ids = sampled_index["f0"].astype(int)
            toas = sampled_index["f1"]*1e9
            datagroup.create_dataset("event_id", data=event_ids)
            t_offset = datagroup.create_dataset("event_time_offset",data=toas)
            t_offset.attrs['units'] = 'ns'
            
            #placesetters for figuring out later- how to make this look more like "events per pulse"
            datagroup.create_dataset(
                "event_index", data=np.zeros(len(event_ids) // 30), dtype=int
            )
            evt_time_zero = datagroup.create_dataset(
                "event_time_zero", data=np.zeros(len(event_ids) // 30), dtype=int
            )
            evt_time_zero.attrs["units"] = "ns"

def get_tof_bins(results: List[np.ndarray], n=50):
    tofmax = 0
    tofmin = np.inf
    for data in results:
        alltofs = data["f1"] # time values
        tofs_max = alltofs.max()
        if tofs_max > tofmax:
            tofmax = tofs_max
        tofs_min = alltofs.min()
        if tofs_min < tofmin:
            tofmin = tofs_min
    tofs = np.linspace(tofmin, tofmax, n)
    print(f"tof_min: {tofmin}, tof_max: {tofmax}")
    return tofs


def make_histogram(data_list: List, time_bin_edges: np.ndarray) -> List[np.ndarray]:
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
        print(f"Shape of histogram {i}: {data.shape}")
    # np.save("data.npy", data_panels)
    return data_panels


def make_animation(datasets: List[np.ndarray], tofs: np.ndarray, folder: Path, fps=5):
    # Process all three datasets
    processed_data = []
    global_vmin, global_vmax = np.inf, 0

    for i, data in enumerate(datasets):
        # Calculate min/max for each panel
        data_nonzero = data[data > 0]
        if len(data_nonzero) > 0:
            vmin, vmax = data_nonzero.min(), data_nonzero.max()
            global_vmin = min(global_vmin, vmin)
            global_vmax = max(global_vmax, vmax)
        else:
            vmin = 1e-8

        # Set background to minimum value for log scale
        data[data <= 0] = vmin
        processed_data.append(data)

    # print(f"Global data range: {global_vmin:.2e} to {global_vmax:.2e}")

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
            axes[i].set_title(f"Panel {i} - Frame: {frame + 1} / {data.shape[2]}")
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
    anim.save(folder / "3panel_animation.gif", writer="pillow", fps=5)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample HDF5 data and create NeXus file")
    parser.add_argument("input_file", type=str, help="Input HDF5 file path")
    parser.add_argument("output_file", type=str, help="Output NeXus file path")
    parser.add_argument(
        "--do-histogram", action="store_true", help="Generate histogram data"
    )
    parser.add_argument("--range", type=int, default=3, help="Number of panels (default: 3)")
    parser.add_argument("--n-samples", type=int, default=10**5, help="Number of samples (default: 100000)")
    return parser.parse_args()

def main():
    args = parse_args()
    sampled = do_sampling(args.input_file, args.range, args.n_samples)
    if Path(args.output_file).exists():
        print(f"Warning: overwriting {args.output_file}...")
    create_nexus_file(args.output_file, sampled)
    if args.do_histogram:
        tof_bins = get_tof_bins(sampled)
        print("Generating histogram...")
        histo = make_histogram(sampled,tof_bins)
        print("Generating animation...")
        make_animation(histo,tof_bins, Path.cwd())

if __name__ == '__main__':
    main()