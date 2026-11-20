import h5py
import numpy as np
from pathlib import Path
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List
from string import Template
from time import perf_counter
import argparse

import logging
import sys

from instrument_output import NMXOutput

NMX_JSON_PATH = Path(__file__).parent / "nmx.json"

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



def parse_args():
    parser = argparse.ArgumentParser(
        description="""Sample McStas/Nexus data and create NeXus file.
        For McStas files with multiple detector configurations.
        """
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
        help="Output NeXus file path, prepends 'config{n}_ to each output file name",
        default="sampling_output.h5",
        required=False,
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=100_000_000,
        help="Number of samples (default: 100_000_000)",
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

    detnums = [str(i) for i in range(3)]
    with h5py.File(args.input_file) as fp:
        confignums = sorted({k.split("_")[0].removeprefix("config") for k in fp["entry1/data"]
                             if k.startswith("config") and "_event_signal_dat_list" in k}, key=int)
        
    s = Template("entry1/data/config${confignum}_Detector_${detnum}_event_signal_dat_list_p_x_y_n_id_t/events")
    output = NMXOutput(args, logger)

    for confignum in confignums:
        datasets = []
        for detnum in detnums:
            dataset = s.substitute(detnum=detnum,confignum=confignum)
            datasets.append(dataset)
        # check n_samples
        total = 0
        n = args.n_samples
        with h5py.File(args.input_file) as fp:
            for dataset in datasets:
                total += len(fp[dataset])
        if total < args.n_samples:
            logger.warning(f"Total events ({total}) < n_samples ({args.n_samples}).")
            logger.warning(f"Setting n_samples to lower value.")
            while n > total:
                n //= 2
            logger.warning(f"New n_samples: {n}")

        sampled = output.sample_datasets(args.input_file, datasets, n)
        output_file_path = Path(args.output_file).resolve()
        output_file =  output_file_path.parent / f"config{confignum}_{output_file_path.name}"

        output.write(output_file, sampled, [args.input_file],
                     simulation_detector_names={
                         0: f"nD_Mantid_config{confignum}_panel0",
                         1: f"nD_Mantid_config{confignum}_panel1",
                         2: f"nD_Mantid_config{confignum}_panel2"
                     })
        if args.do_histogram:
            output.plots(sampled, output_file_path.parent, prefix=f"config{confignum}_")



if __name__ == "__main__":
    main()
