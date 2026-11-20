import h5py
import numpy as np
import argparse
from pathlib import Path
import sys
import logging
import matplotlib.pyplot as plt

from instrument_output import FORMATS, get_instrument_xml_nexus  # noqa: F401  (re-exported)
# NMX functions, kept importable from here
from instrument_output.nmx import (  # noqa: F401
    NMX_PERIOD,
    create_nexus_file,
    do_sampling,
    get_tof_bins,
    get_tof_bins_from_hdf5,
    load_json_dict,
    make_animation,
    make_histogram,
    make_histogram_from_hdf5,
    redistribute_sampling,
    sum_plot,
    write_template_to_nexus,
)

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


def animation_only(args, logger: logging.Logger):
    """Generates the animations only, using sampling output (NMX)."""
    datas = []
    try:
        logger.info(f"Loading file {args.input_file[0]}...")
        with h5py.File(args.input_file[0]) as fp:
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
    plt.close()



def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample HDF5 data and create NeXus file"
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="+",
        required=True,
        help="Input HDF5 file path(s). Multiple files are combined via multifile sampling.")

    parser.add_argument(
        "--format",
        choices=sorted(FORMATS),
        default="nmx",
        help="Output layout: nmx (ESS NMX, json template) or mandi (SNS MaNDi NXsnsevent, "
             "as MANDI_<run>.nxs.h5). Default: nmx",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Output NeXus file path (default: sampling_output.h5 for nmx, "
             "MANDI_<run-number>.nxs.h5 for mandi)",
        default=None,
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
        "-x",
        "--xml",
        type=str,
        help="Mantid xml geometry file (if the one in McStas file is not to be used)"
    )
    nmx = parser.add_argument_group("nmx options")
    nmx.add_argument(
        "-j",
        "--json-file",
        type=str,
        help="Path to json template",
        default=NMX_JSON_PATH,
    )
    nmx.add_argument(
        "--use-mask",
        action="store_true",
        help="Use a mask around the beam center. Useful for ensuring direct beam is not oversampled."
    )
    nmx.add_argument(
        "--do-histogram",
        action="store_true",
        help="Generate histogram data and animation",
    )
    nmx.add_argument(
        "--animation-only",
        action="store_true",
        help="Generate animation only (after sampling)",
    )
    nmx.add_argument(
        "--range",
        type=int,
        default=3,
        help="Number of panels (default: 3)"
    )
    nmx.add_argument(
        "--no-wrap-tofs",
        action="store_true",
        help="Don't wrap tof values to ESS pulse length (14 Hz)"
    )
    mandi = parser.add_argument_group("mandi options")
    mandi.add_argument(
        "--run-number",
        type=int,
        default=0,
        help="Run number written to the file and used in the default file name (default: 0)",
    )
    mandi.add_argument(
        "--start-time",
        type=str,
        help="Run start, ISO 8601 (default: now)",
    )
    mandi.add_argument(
        "--title",
        type=str,
        help="Run title (default: from the McStas output directory name)",
    )
    mandi.add_argument(
        "--lau",
        type=str,
        help="Reflection file (.lau) for the lattice in entry/sample "
             "(default: XtalReflections looked up next to the input file)",
    )
    mandi.add_argument(
        "--seed",
        type=int,
        help="Seed for spreading the events over pulses",
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

    output = FORMATS[args.format](args, logger)
    output_file = Path(args.output_file or output.default_output_file.format(run=args.run_number))
    if args.format != "nmx":
        ignored = [opt for opt in ("use_mask", "do_histogram", "no_wrap_tofs") if getattr(args, opt)]
        if ignored:
            logger.warning(f"Options {ignored} only apply to --format nmx, ignored.")
    elif not Path(args.json_file).exists():
        raise FileNotFoundError(f"File {args.json_file} not found.")

    sampled = output.sample(args.input_file, args.n_samples)

    if output_file.exists():
        logger.warning(f"Overwriting {output_file}...")
    if args.format == "nmx" and args.no_wrap_tofs:
        logger.warning(f"Not wrapping tofs to NMX pulse length.")
    output.write(output_file, sampled, args.input_file)
    if args.do_histogram and args.format == "nmx":
        output.plots(sampled, output_file.parent)


if __name__ == "__main__":
    main()
