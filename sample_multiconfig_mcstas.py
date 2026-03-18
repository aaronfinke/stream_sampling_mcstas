from juliacall import Main as jl
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
import sampling_to_nexus
import mcstas_geometry_xml
import mcstas_to_nexus_geometry
import sys

julia_path = Path("/project/project_465002587/stream_sampling_mcstas")
jl.include(str(julia_path / "sampling_toNexus.jl"))
NMX_PERIOD = 1/14
mpl.rcParams["animation.embed_limit"] = 500
NMX_JSON_PATH = "/project/project_465002030/streaming_mcstas_nexus.json"

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


def do_sampling(args:argparse.Namespace,datasets: List[str], filename: str, logger:logging.Logger, n: int = 100_000_000) -> List[np.ndarray]:

    def redistribute_sampling(sampled):
        pixids = sampled["f0"]
        d0 = sampled[pixids < 1280 * 1280]
        d1 = sampled[(1280 * 1280 <= pixids) & (pixids < 2 * 1280 * 1280)]
        d2 = sampled[(pixids >= 2 * 1280 * 1280)]
        return [d0, d1, d2]

    if args.use_mask:
        # XXX include mask support... somehow.
        logger.warning("Masks not currently supported.")

    eval_statement = f'sample_all_frames_generic("{filename}", "{datasets}", {n}, AlgWRSWRSKIP())'
    logger.info(f"Running {eval_statement} ...")
    t1 = perf_counter()
    sampled_jl = jl.seval(eval_statement)
    logger.info(f"... done. Took {perf_counter() - t1:.2f} s.")
    sampled = sampled_jl.to_numpy()
    redistributed = redistribute_sampling(sampled)
    totsamp = 0
    for i,sample in enumerate(redistributed):
        logger.info(f"Number of samples in panel {i}: {len(sample)}")
        totsamp += len(sample)
    logger.debug(f"Total samples: {totsamp}")

    return redistributed

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
    confignums = [str(i) for i in range(1,12)]

    s = Template("entry1/data/Config${confignum}_Panel${detnum}_event_signal_dat_list_p_x_y_n_id_t/events")
    
    if args.xml:
        logger.info(f"Using geometry from xml file {args.xml}...")
        instrument_xml = ' '.join(open(args.xml).readlines())
        geometry = mcstas_to_nexus_geometry.load_xml_geometry(Path(args.xml),logger=logger)
    else:
        geometry = mcstas_geometry_xml.read_mcstas_geometry_xml(Path(args.input_file))
        instrument_xml = mcstas_geometry_xml.get_instrument_xml_nexus(file_path=args.input_file)

    configs = {}
    for confignum in confignums:
        
        datasets = []
        for detnum in detnums:
            dataset = s.substitute(detnum=detnum,confignum=confignum)
            datasets.append(dataset)
        sampled = do_sampling(args, datasets, args.input_file, logger, n=args.n_samples)
        output_file_path = Path(args.output_file).parent
        output_file =  Path(args.output_file).parent / f"config{confignum}_{args.output_file}"

        sampling_to_nexus.create_nexus_file(args=args, 
                                            output_file=output_file, 
                                            sampled=sampled, 
                                            json_template=args.json_file, 
                                            instrument_xml=instrument_xml,
                                            logger=logger)
        mcstas_to_nexus_geometry.insert_geometry_into_nexus(geometry,output_file,logger)
        if args.do_histogram:
            tof_bins = sampling_to_nexus.get_tof_bins(args,sampled,logger=logger)
            logger.info("Generating histogram...")
            histo = sampling_to_nexus.make_histogram(sampled, tof_bins,logger=logger)
            logger.info("Saving sum plot...")
            sampling_to_nexus.sum_plot(histo, output_file_path, filename=f"config{confignum}_allbins.png")
            logger.info("Generating animation...")
            sampling_to_nexus.make_animation(args,histo, tof_bins, output_file_path, filename=f"config{confignum}3panel_animation.gif")



if __name__ == "__main__":
    main()
