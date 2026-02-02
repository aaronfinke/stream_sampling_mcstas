import h5py
import numpy as np
from pathlib import Path
from time import perf_counter
import shutil
import sys
import argparse
from typing import List, Tuple

import matplotlib.pyplot as plt
import bitshuffle.h5
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.colors import LogNorm

pixid_binsize = 1280*1280

def min_max_probability_tof(filename:Path) -> tuple[float]:
    max_p, min_p = 0.0, np.inf
    max_t, min_t = 0.0, np.inf

    with h5py.File(filename,"r") as fp:
        for i in range(3):
            print(f"minmax_t frame {i}")
            dset = fp["entry1/data"][f"Detector_{i}_event_signal_dat_list_p_x_y_n_id_t"]["events"]
            slicelist = get_chunk_slices(dset, nchunk=50_000_000)            
            for s in slicelist:
                chunk = dset[s]
                chunk_t = chunk.T[5]
                chunk_p = chunk.T[0]
                tempmax_t = np.max(chunk_t)
                tempmin_t = np.min(chunk_t[chunk_t > 0])
                tempmax_p = np.max(chunk_p)
                tempmin_p = np.min(chunk_p[chunk_p > 0])
                if tempmax_t > max_t:
                    max_t = tempmax_t
                if tempmax_p > max_p:
                    max_p = tempmax_p
                if tempmin_p < min_p:
                    min_p = tempmin_p
                if tempmin_t < min_t:
                    min_t = tempmin_t
    print(f"max_t: {max_t}, min_t: {min_t}")
    print(f"max_p: {max_p}, min_t: {min_p}")
    return (max_t,min_t,max_p,min_p)

def get_chunk_slices(dset: h5py.Dataset, nchunk: int=10_000) -> List[Tuple[slice, slice]]:
    """
    Get a list of slices of a dataset of size nchunk.
    """
    nentry = dset.shape[0]
    remainder = nentry % nchunk
    ntot = nentry - remainder
    slicelist = []
    for start in range(0, ntot, nchunk):
        slicelist.append((slice(start, start + nchunk, 1), slice(0, 6, 1)))
    slicelist.append((slice(ntot, ntot + remainder, 1), slice(0, 6, 1)))
    return slicelist

def make_gif(dset_file: Path):
    mpl.rcParams["animation.embed_limit"] = 500
    gif_output = dset_file.with_suffix(".gif")

    # Load data from all three detector panels
    with h5py.File(dset_file) as fp:
        data0 = fp["entry/detector_0/data"][:]
        data1 = fp["entry/detector_1/data"][:]
        data2 = fp["entry/detector_2/data"][:]

    print(f"Panel 0 shape: {data0.shape}")
    print(f"Panel 1 shape: {data1.shape}")
    print(f"Panel 2 shape: {data2.shape}")

    # Process all three datasets
    datasets = [
        data0.reshape(1280, 1280, 200),
        data1.reshape(1280, 1280, 200),
        data2.reshape(1280, 1280, 200),
    ]
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

        # Set background to minimum value
        data[data <= 0] = vmin
        processed_data.append(data)

    print(f"Global data range: {global_vmin:.2e} to {global_vmax:.2e}")

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
            norm=LogNorm(vmin=1e-15, vmax=global_vmax),
        )
        ax.set_title(f"Panel {i} - Time bin: 0 / {data.shape[2] - 1}")
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
        """Animation function that updates all three panels for each time bin"""
        for i, (im, data) in enumerate(zip(ims, processed_data)):
            im.set_array(data[:, :, frame])
            axes[i].set_title(f"Panel {i} - Time bin: {frame + 1} / {data.shape[2]}")
        return ims


    # Create animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=200,
        interval=100,  # 200ms between frames
        blit=True,
        repeat=True,
    )
    anim.save(
        gif_output,
        writer="pillow",
        fps=10,
    )
    return

def main():
    t0 = perf_counter()
    parser = argparse.ArgumentParser(description="Process McStas data.")
    parser.add_argument("--input_file", type=str, help="Path to the input file")
    parser.add_argument(
        "--output_file",
        type=str,
        default="scipp_output.nxs",
        help="Path to the output file",
    )

    args = parser.parse_args()

    input_file = Path(args.input_file).resolve()
    output_file = Path(args.output_file).resolve()

    with h5py.File(output_file,'w') as fp:
        fp.create_group('entry')

    max_t, min_t, max_p, min_p = min_max_probability_tof(input_file)

    tof_bins = np.linspace(min_t,max_t,200)
    t1 = perf_counter()
    print(f"processing file {input_file}...")
    if not Path(input_file).exists():
        sys.exit(f"file {input_file} not found.")
    with h5py.File(input_file,"r") as fp:
        for j in range(3):
            print("here we go!")
            id_p = np.zeros((pixid_binsize,len(tof_bins)))      # instantiate the histogram for each panel
            id_error = np.zeros((pixid_binsize,len(tof_bins)))  # and for the error (which is just the square of the sum(probability) for each pixel)

            print(f"frame {j}")
            dset = fp["entry1/data"][f"Detector_{j}_event_signal_dat_list_p_x_y_n_id_t"]["events"]
            slicelist = get_chunk_slices(dset, nchunk=50_000_000)   # it is not straightforward to use h5py.iterchunks() here so I had to write my own
            for s in slicelist:                         # manually chunking data
                chunk = dset[s]                         # get the chunk data
                tof = chunk.T[5]                        # oddly, this is the fastest way in numpy to get a column from a 2D matrix...
                indexed = np.digitize(tof, tof_bins)    # which row goes into which TOF bin
                for i,_ in enumerate(tof_bins):         # now iterate over all TOF bins...
                    f = chunk[np.where(indexed == i)]   # take only the data that is in this TOF bin.
                    eventdata = np.bincount(            # for each pixelID in this TOF bin, add up all probabilities (the weights= parameter here). 
                        f[:, 4].astype(int),            # pixelIDs are listed for three panels 0-3 where each panel has 1280*1280 pixels,
                        weights=f[:, 0],                # so for e.g. panel 0, pixelIDs are 0:(1280*1280), panel 1 are (1280*1280):(1280*1280)*2, etc.
                        minlength=1280 * 1280 * (j + 1),
                    )                                   
                    eventdata = eventdata[pixid_binsize * j :]
                    errordata = np.bincount(
                        f[:, 4].astype(int),
                        weights=np.power(f[:, 0], 2),
                        minlength=1280 * 1280 * (j + 1),
                    )
                    errordata = errordata[pixid_binsize * j :]
                    id_p[:, i] += eventdata             # now add the sum of all probabilities for each pixelID in this TOF bin to the main histogram
                    id_error[:, i] += errordata         # and the error too
            for i in range(len(tof_bins)):
                print(
                    f"sum of frame{j}, tof {tof_bins[i - 1]:.4f} - {tof_bins[i]:.4f}: {id_p[:, i].sum():.2e}"
                )
            t2 = perf_counter()
            with h5py.File(output_file,'r+') as fp1:
                grp = fp1['entry'].create_group(f'detector_{j}')
                data = grp.create_dataset("data", data=id_p)
                data.attrs["max_p"] = max_p
                data.attrs["min_p"] = min_p
                grp.create_dataset("errors",data=id_error)

            print(f"time for frame {j}: {t2-t1:.2f} s")
            
    if output_file.exists():
        print("making gif...")
        make_gif(output_file)

    t3 = perf_counter()
    print(f"time for all frames: {t3-t0:.2f} s")



if __name__ == "__main__":
    main()
