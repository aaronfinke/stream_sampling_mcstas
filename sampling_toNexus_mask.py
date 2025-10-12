import h5py
import numpy as np
import time
import shutil

import sys
from multiprocessing import Pool
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib as mpl

import argparse
from pathlib import Path
from typing import List, Tuple

current_datetime_str = time.strftime("%Y-%m-%d_%H-%M-%S")

sys.path.insert(0, "/Users/aaronfinke/StreamSampling.py")
import weighted_sampling_multi_python

mpl.rcParams["animation.embed_limit"] = 500

def get_rectangular_mask_pixids(
    x_start, x_end, y_start, y_end, detector_id=1, grid_width=1280, grid_height=1280
):
    """
    Get pixel IDs for a rectangular region to mask out.
    Only works for Detector 1 (ie. the middle detector), for e.g.
    masking beamstop.
    """
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[y_start:y_end, x_start:x_end]

    # Convert 2D coordinates to 1D pixel IDs
    # Formula: pixelID = y * grid_width + x
    pixel_ids = y_coords * grid_width + x_coords
    # multiply coordinates by detector number
    pixel_ids += detector_id * grid_width * grid_height
    # Flatten to get 1D array of pixel IDs
    return pixel_ids.flatten()


def sample_frame(
    index: int, f: Path, n: int = 10_000_000, mask_regions: List[Tuple] = None) -> List[Tuple]:

    sampler = weighted_sampling_multi_python.stream_weighted_sample_multi(
        n=n, method=weighted_sampling_multi_python.SamplingMethod.WRSWR_SKIP
    )
    print(f"Masked regions selected: {mask_regions}")
    masked_pixids = set()
    if mask_regions:
        for x_start, x_end, y_start, y_end in mask_regions:
            mask_pixids = get_rectangular_mask_pixids(x_start, x_end, y_start, y_end)
            masked_pixids.update(mask_pixids)
        print(f"Detector {index}: Masking {len(masked_pixids)} pixels")

    timelist = []
    total_events = 0
    masked_events = 0

    with h5py.File(f) as fp:
        dset0 = fp[
            f"entry1/data/Detector_{index}_event_signal_dat_list_p_x_y_n_id_t/events"
        ]
        for s in dset0.iter_chunks():
            t0 = time.perf_counter()
            chunk = dset0[s]
            weights = chunk[:, 0]

            # Extract pixel IDs and times
            pixel_ids = chunk[:, 4].astype(int)
            times = chunk[:, 5]

            total_events += len(chunk)

            # Filter out masked pixels
            if masked_pixids:
                # Create boolean mask for non-masked pixels
                valid_mask = ~np.isin(pixel_ids, list(masked_pixids))

                # Apply mask
                weights = weights[valid_mask]
                pixel_ids = pixel_ids[valid_mask]
                times = times[valid_mask]

                masked_events += (~valid_mask).sum()

            # Create elements list from filtered data
            elements = [(pid, t) for pid, t in zip(pixel_ids, times)]

            # Fit sampler with filtered data
            for weight, element in zip(weights, elements):
                sampler.fit(element, weight)

    if masked_pixids:
        print(
            f"Detector {index}: Masked {masked_events}/{total_events} events ({100 * masked_events / total_events:.1f}%)"
        )

    return sampler.value()


def get_tof_bins(results: List[np.ndarray], n=50):
    tofmax = 0
    tofmin = np.inf
    for data in results:
        alltofs = np.array([x[1] for x in data])  # time values
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
        pixids = np.array([int(x[0]) for x in data])  # pixel IDs
        times = np.array([x[1] for x in data])  # time values

        # Assign each event to a time bin
        time_bin_indices = np.digitize(times, time_bin_edges) - 1
        # Clip to ensure all indices are within valid range
        time_bin_indices = np.clip(time_bin_indices, 0, n_time_bins - 1)

        np.add.at(pixel_counts_per_bin, (pixids, time_bin_indices), 1)
    np.save("ouput.npy", pixel_counts_per_bin)
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
            axes[i].set_title(f"Panel {i} - Frame: {frame + 1} / {data.shape[0]}")
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


def change_hdf5File(histos: List[np.ndarray], tofs: np.ndarray, filename: Path):
    with h5py.File(filename, "r+") as fp:
        g0 = fp["/entry/instrument/nD_Mantid_0"]
        del g0["data"]
        del g0["time_of_flight"]
        g0.create_dataset("data", data=histos[0], dtype=int)
        g0.create_dataset("time_of_flight", data=tofs, dtype=float)

        g1 = fp["/entry/instrument/nD_Mantid_1"]
        del g1["data"]
        del g1["time_of_flight"]
        g1.create_dataset("data", data=histos[1], dtype=int)
        g1.create_dataset("time_of_flight", data=tofs, dtype=float)

        g2 = fp["/entry/instrument/nD_Mantid_2"]
        del g2["data"]
        del g2["time_of_flight"]
        g2.create_dataset("data", data=histos[2], dtype=int)
        g2.create_dataset("time_of_flight", data=tofs, dtype=float)


def main():
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Sample nexus output file, bin/histogram, and save in scipp format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "mcstas_file",
        type=Path,
        help="Path to the input McStas NeXus file (HDF5 format)",
    )

    parser.add_argument(
        "scipp_file", type=Path, help="Path to the output Scipp file (HDF5 format)"
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        default=Path.cwd(),
        help="Path to the output directory for animations and processed data",
    )

    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=10_000_000,
        help="Number of samples for weighted sampling",
    )

    parser.add_argument(
        "--n-time-bins",
        type=int,
        default=50,
        help="Number of time bins for histogramming",
    )

    # Add masking arguments
    parser.add_argument(
        "--mask-beamstop",
        action="store_true",
        help="Mask out center beamstop region (590:690, 590:690)",
    )

    parser.add_argument(
        "--mask-regions",
        nargs=4,
        metavar=("x_start", "x_end", "y_start", "y_end"),
        type=int,
        action="append",
        help="Mask rectangular region (can be used multiple times)",
    )

    parser.add_argument(
        "--no-animation", action="store_true", help="Skip creating animation GIF"
    )

    parser.add_argument(
        "--fps", type=int, default=5, help="Frames per second for animation GIF"
    )

    args = parser.parse_args()

    # Validate input files exist
    if not args.mcstas_file.exists():
        parser.error(f"McStas file does not exist: {args.mcstas_file}")

    if not args.scipp_file.exists():
        parser.error(f"Scipp file does not exist: {args.scipp_file}")

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up mask regions
    mask_regions = []

    if args.mask_beamstop:
        mask_regions.append((590, 690, 590, 690))  # Center beamstop
        print("Adding beamstop mask to panel 1: (590:690, 590:690)")

    if args.mask_regions:
        for region in args.mask_regions:
            mask_regions.append(tuple(region))
            print(
                f"Adding custom mask: ({region[0]}:{region[1]}, {region[2]}:{region[3]})"
            )

    print(f"Input McStas file: {args.mcstas_file}")
    print(f"Output Scipp file: {args.scipp_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Number of time bins: {args.n_time_bins}")

    print("Sampling datasets...")
    with Pool(processes=3) as pool:
        results = pool.starmap(
            sample_frame,
            [
                (detector_id, args.mcstas_file, args.n_samples, mask_regions)
                for detector_id in range(3)
            ],
        )

    tofs = get_tof_bins(results, args.n_time_bins)

    print("Generating histos...")
    histos = make_histogram(results, tofs)

    if not args.no_animation:
        print("making animation...")
        make_animation(histos, tofs, args.output_dir, fps=args.fps)
    copy_scipp_path = args.output_dir / "sample_output.h5"
    if copy_scipp_path.exists():
        print(f"Path {str(copy_scipp_path)} exists.")
        copy_scipp_path = args.output_dir / f"sample_output_{current_datetime_str}.h5"
        print(f"Saving output as {str(copy_scipp_path)}")

    print(f"Copying {str(copy_scipp_path)} to {str(copy_scipp_path)}...")
    shutil.copy2(args.scipp_file, copy_scipp_path)

    change_hdf5File(histos, tofs, copy_scipp_path)

    print("Processing completed successfully!")
    print(f"Time: {time.perf_counter() - t0} s")


if __name__ == "__main__":
    main()
