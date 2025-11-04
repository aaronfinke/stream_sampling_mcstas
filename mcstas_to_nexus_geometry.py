import argparse
import h5py
import warnings
import logging
import scipp as sc

import scippnexus as snx
from pathlib import Path
from mcstas_geometry_xml import (
    read_mcstas_geometry_xml,
    McStasInstrument,
    DetectorDesc,
    SampleDesc,
)


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


def _overwrite_detector_numbers(
    nexus_det: h5py.Group, det: DetectorDesc, id_start: int | None = None
):
    id_start = id_start if id_start is not None else det.id_start
    detector_number = sc.arange(
        dim="detector_number",
        start=id_start,
        stop=id_start + det.num_x * det.num_y,
        dtype=sc.DType.int32,
        unit=None,
    )
    detector_number = detector_number.fold(
        dim="detector_number",
        sizes={"y_pixel_offset": det.num_y, "x_pixel_offset": det.num_x},
    )
    _overwrite_or_create_dataset(
        var=detector_number, nexus_det=nexus_det, name="detector_number"
    )


def _overwrite_or_create_dataset(
    var: sc.Variable, nexus_det: h5py.Group, name: str, *, attrs: dict | None = None
):
    if name in nexus_det:
        del nexus_det[name]  # Remove existing dataset if any

    values = var.value if var.dims == () else var.values
    dset = nexus_det.create_dataset(name=name, data=values)

    attrs = {**attrs} if attrs is not None else {}
    if var.unit is not None:
        attrs["units"] = str(var.unit)

    dset.attrs.update(attrs)


def _overwrite_pixel_offsets(nexus_det: h5py.Group, det: DetectorDesc):
    # Always assuming centered detector
    x_length = det.step_x * det.num_x
    y_length = det.step_y * det.num_y
    # X pixel offsets
    x_offset = sc.arange(
        dim="x_pixel_offset",
        start=det.start_x,
        stop=det.start_x + x_length,
        step=det.step_x,
        unit=det.step_x.unit,
    )
    # Y pixel offsets
    y_offset = sc.arange(
        dim="y_pixel_offset",
        start=det.start_y,
        stop=det.start_y + y_length,
        step=det.step_y,
        unit=det.step_y.unit,
    )

    offset_names = ["x_pixel_offset", "y_pixel_offset"]
    for name in offset_names:
        if name in nexus_det:
            del nexus_det[name]  # Remove existing dataset if any

    # Slow axis means the axis that detector numbers change more slowly
    # i.e. the outer loop in nested loops over pixels
    # For example, if slow_axis is 'y', then:
    # for y in range(num_y):
    #     for x in range(num_x):
    #         detector_number += 1
    # Therefore slow axis should be the lower number axis in NeXus convention
    # as they are ordered from slowest to fastest changing axis (outer to inner loop)
    x_axis_order = 2 if det.slow_axis_name == "y" else 1
    y_axis_order = 1 if det.slow_axis_name == "y" else 2
    _overwrite_or_create_dataset(
        var=x_offset,
        nexus_det=nexus_det,
        name="x_pixel_offset",
        attrs={"axis": x_axis_order},
    )
    _overwrite_or_create_dataset(
        var=y_offset,
        nexus_det=nexus_det,
        name="y_pixel_offset",
        attrs={"axis": y_axis_order},
    )


def _overwrite_transformations(
    nexus_det: h5py.Group,
    det_desc: DetectorDesc,
    sample_desc: SampleDesc,
    handedness: str = "right",
):
    transformations: h5py.Group = nexus_det["transformations"]
    rotation_vector = det_desc.rotation_vector
    orientations = transformations["orientation"]
    orientations.attrs["depends_on"] = f"{nexus_det.name}/transformations/stageZ"
    orientations.attrs["vector"] = list(rotation_vector.value)
    rot_angle = (
        -det_desc.rotation_angle if handedness == "right" else det_desc.rotation_angle
    )
    orientations[...] = rot_angle.to(unit=orientations.attrs["units"]).value

    stageZ = transformations["stageZ"]
    original_attrs = stageZ.attrs

    det_position = sample_desc.position_from_sample(det_desc.position)
    translation_offset = (det_position).to(unit=original_attrs["offset_units"])
    original_attrs["offset"] = list(translation_offset.value)
    _overwrite_or_create_dataset(
        var=sc.scalar(0.0, unit="mm"),
        nexus_det=transformations,
        name="stageZ",
        attrs=original_attrs,
    )
    for axis_i in range(1, 7):
        if (axis_name := f"axis{axis_i}") in transformations.keys():
            del transformations[axis_name]


def _map_mcstas_to_nexus_detector_name(
    nexus_detector_names: list[str],
    mcstas_detector_names: list[str],
    *,
    logger: logging.Logger,
) -> dict[str, str]:
    detector_names = sorted(nexus_detector_names)
    mcstas_detector_names = sorted(mcstas_detector_names)
    logger.debug("Found detectors: %s", detector_names)
    mcstas_to_nexus_detector_name_map = dict(zip(mcstas_detector_names, detector_names))
    logger.debug(
        "Mapping detectors from XML geometry to NeXus file like this: %s,",
        mcstas_to_nexus_detector_name_map,
    )
    return dict(zip(mcstas_detector_names, nexus_detector_names))


def insert_geometry_into_nexus(
    geometry: McStasInstrument, output_file: Path, logger: logging.Logger
):
    logger.info("Inserting geometry into NeXus file: %s", output_file.as_posix())
    instrument_path = Path("entry/instrument")

    # Reading existing detectors from NeXus file with scippnexus
    # It is really convenient to use scippnexus to read NeXus structure
    # but not to write...
    with snx.File(output_file.as_posix(), "r") as nexus_file:
        inst_gr: snx.Group = nexus_file[instrument_path.as_posix()]
        detectors = inst_gr[snx.NXdetector].keys()

    mcstas_det_map = {det.name: det for det in geometry.detectors}
    mcstas_to_nexus_names = _map_mcstas_to_nexus_detector_name(
        nexus_detector_names=list(detectors),
        mcstas_detector_names=list(mcstas_det_map.keys()),
        logger=logger,
    )

    geometry.simulation_settings.handedness

    with h5py.File(output_file.as_posix(), "r+") as nexus_file:
        for mcstas_det_name, nexus_det_name in mcstas_to_nexus_names.items():
            logger.debug(
                "Inserting detector: %s into %s", mcstas_det_name, nexus_det_name
            )
            det_desc = mcstas_det_map[mcstas_det_name]

            detector_gr_path = instrument_path / nexus_det_name
            nexus_det = nexus_file[detector_gr_path.as_posix()]

            logger.debug("Overwriting detector_number dataset")
            _overwrite_detector_numbers(nexus_det, det_desc)

            logger.debug("Overwriting pixel offsets")
            _overwrite_pixel_offsets(nexus_det, det_desc)

            logger.debug("Overwriting transformation")
            _overwrite_transformations(
                nexus_det=nexus_det,
                det_desc=det_desc,
                sample_desc=geometry.sample,
                handedness=geometry.simulation_settings.handedness,
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse mcstas xml geometry "
        "and insert the information as nexus standard geometry"
        "(pixel offsets, detector_numbers and NXtransformation) into a NeXus file.\n"
        "Note that this will open the NeXus file in editing mode and "
        "may overwrite existing geometry information in the file.",
    )
    parser.add_argument(
        "input_file", type=str, help="Input HDF5 or xml file path with geometry"
    )
    parser.add_argument(
        "output_file", type=str, help="Output NeXus file path to insert geometry into"
    )
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
