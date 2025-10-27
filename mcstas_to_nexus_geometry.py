import argparse
import warnings
import logging
import scipp as sc

import scippnexus as snx
from pathlib import Path
from mcstas_geometry_xml import read_mcstas_geometry_xml, McStasInstrument, DetectorDesc


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


def load_xml_geometry(file_path: Path, logger: logging.Logger) -> McStasInstrument:
    geometry = read_mcstas_geometry_xml(file_path)
    logger.debug("Successfully loaded geometry: %s", geometry)
    return geometry


def _insert_detector_numbers(nexus_det: snx.NXdetector, det: DetectorDesc):
    detector_number = sc.arange(
        dim="detector_number",
        start=det.id_start,
        stop=det.id_start + det.num_x * det.num_y,
        dtype=sc.DType.int32,
        unit=None,
    )
    detector_number = detector_number.fold(
        dim="detector_number",
        sizes={"y_pixel_offset": det.num_y, "x_pixel_offset": det.num_x},
    )
    nexus_det.create_field(key="detector_number", value=detector_number)


def _insert_pixel_offset(nexus_det: snx.NXdetector, det: DetectorDesc):
    x_pixel_offset = sc.linspace(
        dim="x_pixel_offset", start=-255.9, stop=255.9, num=det.num_x, unit="mm"
    )
    x_offset_dset = nexus_det.create_field(key="x_pixel_offset", value=x_pixel_offset)
    x_offset_dset.attrs["axis"] = 1  # X axis

    y_pixel_offset = sc.linspace(
        dim="y_pixel_offset", start=-255.9, stop=255.9, num=det.num_y, unit="mm"
    )
    y_offset_dset = nexus_det.create_field(key="y_pixel_offset", value=y_pixel_offset)
    y_offset_dset.attrs["axis"] = 2  # Y axis


def insert_geometry_into_nexus(
    geometry: McStasInstrument, output_file: Path, logger: logging.Logger
):
    logger.info("Inserting geometry into NeXus file: %s", output_file.as_posix())
    with snx.File(output_file.as_posix(), "r+") as nexus_file:
        nexus_detectors = nexus_file["entry/instrument"][snx.NXdetector]
        # Sort them by name
        nexus_detectors = sorted(nexus_detectors.items(), key=lambda item: item[0])
        nexus_detectors = {name: det for name, det in nexus_detectors}
        for det, nexus_det_name in zip(geometry.detectors, nexus_detectors.keys()):
            nexus_det = nexus_detectors[nexus_det_name]
            logger.debug("Inserting detector: %s into %s", det.name, nexus_det_name)
            # Insert detector number
            if "detector_number" not in nexus_det:
                logger.debug("Creating detector_number dataset")
                _insert_detector_numbers(nexus_det, det)

            # Insert pixel offsets
            if "x_pixel_offset" not in nexus_det and "y_pixel_offset" not in nexus_det:
                logger.debug("Creating x_pixel_offset and y_pixel_offset datasets")
                _insert_pixel_offset(nexus_det, det)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample HDF5 data and create NeXus file"
    )
    parser.add_argument("input_file", type=str, help="Input HDF5 file path")
    parser.add_argument("output_file", type=str, help="Output NeXus file path")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = _get_logger(args)

    # Check input and output file paths
    if not (input_file_path := Path(args.input_file)).exists():
        raise FileNotFoundError(
            f"Input file {input_file_path.as_posix()} does not exist."
        )
    else:
        logger.debug(
            "Loading XML geometry from input file: %s", input_file_path.as_posix()
        )
        simulation_geometry = load_xml_geometry(input_file_path, logger)

    if (output_file_path := Path(args.output_file)).exists():
        warnings.warn(
            f"Warning: overwriting {output_file_path.as_posix()}...",
            RuntimeWarning,
            stacklevel=3,
        )

    insert_geometry_into_nexus(simulation_geometry, output_file_path, logger)


if __name__ == "__main__":
    main()
