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
    SourceDesc,
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


def _load_crystal_rotation(
    mcstas_file: h5py.File, instrument: McStasInstrument
) -> sc.Variable:
    """Retrieve crystal rotation from the file.

    Raises
    ------
    KeyError
        If the crystal rotation is not found in the file.

    """
    param_keys = tuple(f"entry1/simulation/Param/XtalPhi{key}" for key in "XYZ")
    if not all(key in mcstas_file for key in param_keys):
        raise KeyError(
            f"Crystal rotations [{', '.join(param_keys)}] not found in file."
        )
    return sc.vector(
        value=[float(mcstas_file[param_key][()][0]) for param_key in param_keys],
        unit=instrument.simulation_settings.angle_unit,
    )


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
    slow_axis_num = det.num_y if det.slow_axis_name == "y" else det.num_x
    fast_axis_num = det.num_x if det.slow_axis_name == "y" else det.num_y
    detector_number = detector_number.fold(
        dim="detector_number",
        sizes={
            f"{det.slow_axis_name}_pixel_offset": slow_axis_num,
            f"{det.fast_axis_name}_pixel_offset": fast_axis_num,
        },
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


def _overwrite_detector_transformations(
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


def _overwrite_sample_transformations(nexus_sample: h5py.Group):
    if "depends_on" not in nexus_sample:
        warnings.warn(
            "Sample group does not have depends_on field. "
            "Overwriting depends_on to be default value /entry/sample/transformations/axis6",
            RuntimeWarning,
            stacklevel=1,
        )

    nexus_sample.create_dataset(
        "depends_on", data=b"/entry/sample/transformations/axis6"
    )

    depends_on = (
        nexus_sample["depends_on"][()]
        .decode("utf-8")
        .removeprefix("/entry/sample/transformations/")
    )
    transformations: h5py.Group = nexus_sample["transformations"]
    # All geometry is relative to sample so we just set offsets to zero
    first_depend_on = transformations[depends_on]
    original_attrs = first_depend_on.attrs
    original_attrs["offset"] = [0.0, 0.0, 0.0]
    original_attrs["depends_on"] = "."
    _overwrite_or_create_dataset(
        var=sc.scalar(0.0, unit="degree"),
        nexus_det=transformations,
        name=depends_on,
        attrs=original_attrs,
    )


def _insert_crystal_rotation(
    nexus_sample: h5py.Group, crystal_rotation: sc.Variable | None
):
    if crystal_rotation is None:
        return

    _overwrite_or_create_dataset(
        var=crystal_rotation, nexus_det=nexus_sample, name="crystal_rotation"
    )


def _overwrite_source_transformations(
    nexus_source: h5py.Group, source_desc: SourceDesc, sample_desc: SampleDesc
):
    if "depends_on" not in nexus_source:
        warnings.warn(
            "Sample group does not have depends_on field. "
            "Overwriting depends_on to be default value "
            "/entry/instrument/source/transformations/translation",
            RuntimeWarning,
            stacklevel=1,
        )

    nexus_source.create_dataset(
        "depends_on", data=b"/entry/instrument/source/transformations/translation"
    )
    transformations: h5py.Group = nexus_source["transformations"]
    source_position = sample_desc.position_from_sample(source_desc.position)
    x, y, z = source_position.value
    # Assuming only translation to x, z axis
    translation_x = transformations["translation_x"]
    x_original_attrs = translation_x.attrs
    translation_x_value = sc.scalar(value=x, unit=source_position.unit)
    _overwrite_or_create_dataset(
        var=translation_x_value,
        nexus_det=transformations,
        name="translation_x",
        attrs=x_original_attrs,
    )
    translation_z_value = sc.scalar(value=z, unit=source_position.unit)
    translation_z = transformations["translation"]
    z_original_attrs = translation_z.attrs
    _overwrite_or_create_dataset(
        var=translation_z_value,
        nexus_det=transformations,
        name="translation",
        attrs=z_original_attrs,
    )


def _map_mcstas_to_nexus_detector_name(
    nexus_detector_names: list[str],
    mcstas_detector_names: list[str],
    *,
    logger: logging.Logger,
) -> dict[str, str]:
    detector_names = sorted(nexus_detector_names)
    mcstas_detector_names = sorted(mcstas_detector_names)
    logger.debug("Found detectors: %s", detector_names)
    mcstas_to_nexus_detector_name_map = dict(
        zip(mcstas_detector_names, detector_names, strict=False)
        # We cut off extra detectors in the nexus file and only use
        # as many as in the mcstas geometry.
        # It is because nexus file may contain extra placeholder detectors.
    )
    logger.debug(
        "Mapping detectors from XML geometry to NeXus file like this: %s,",
        mcstas_to_nexus_detector_name_map,
    )
    return mcstas_to_nexus_detector_name_map


def insert_geometry_into_nexus(
    geometry: McStasInstrument,
    output_file: Path,
    logger: logging.Logger,
    crystal_rotation_file_path: str = "",
):
    # Try loading crystal rotation if file path is provided
    if crystal_rotation_file_path:
        logger.info(
            "Loading crystal rotation from file: %s", crystal_rotation_file_path
        )
        with h5py.File(crystal_rotation_file_path, "r") as mcstas_file:
            crystal_rotation = _load_crystal_rotation(
                mcstas_file=mcstas_file, instrument=geometry
            )
        logger.info("Setting crystal rotation to: %s", crystal_rotation)

    else:
        logger.info(
            "No crystal rotation file path provided, skipping crystal rotation."
        )
        crystal_rotation = None

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
        logger.debug("Overwriting source/sample transformation")
        _overwrite_sample_transformations(nexus_sample=nexus_file["entry/sample"])
        _overwrite_source_transformations(
            nexus_source=nexus_file["entry/instrument/source"],
            source_desc=geometry.source,
            sample_desc=geometry.sample,
        )
        _insert_crystal_rotation(
            nexus_sample=nexus_file["entry/sample"], crystal_rotation=crystal_rotation
        )
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

            logger.debug("Overwriting detector transformation")
            _overwrite_detector_transformations(
                nexus_det=nexus_det,
                det_desc=det_desc,
                sample_desc=geometry.sample,
                handedness=geometry.simulation_settings.handedness,
            )


def wrap_event_time_offsets(
    output_file: Path, pulse_period: sc.Variable, logger: logging.Logger
):
    logger.info("Wrapping event time offsets in NeXus file: %s", output_file.as_posix())
    instrument_path = Path("entry/instrument")

    # Reading existing detectors from NeXus file with scippnexus
    # It is really convenient to use scippnexus to read NeXus structure
    # but not to write...
    with snx.File(output_file.as_posix(), "r") as nexus_file:
        inst_gr: snx.Group = nexus_file[instrument_path.as_posix()]
        detectors = inst_gr[snx.NXdetector].keys()
        event_time_offsets: dict[str, sc.Variable] = {
            det_name: nexus_file[(instrument_path / det_name / "data").as_posix()][
                "event_time_offset"
            ][()]
            for det_name in detectors
        }
        event_time_offsets = {
            det_name: etf % pulse_period.to(unit=etf.unit)
            for det_name, etf in event_time_offsets.items()
        }

    with h5py.File(output_file.as_posix(), "r+") as nexus_file:
        for nexus_det_name in detectors:
            logger.debug("Wrapping event time offsets for detector: %s", nexus_det_name)
            data_gr_path = instrument_path / nexus_det_name / "data"
            nexus_det = nexus_file[data_gr_path.as_posix()]
            logger.debug(nexus_det.keys())

            if "event_time_offset" not in nexus_det:
                raise ValueError(
                    f"Detector {nexus_det_name} does not have event_time_offset dataset."
                )
            event_time_offset = event_time_offsets[nexus_det_name]
            wrapped_time_offset = event_time_offset
            del nexus_det["event_time_offset"]  # Remove existing dataset
            _overwrite_or_create_dataset(
                var=wrapped_time_offset,
                nexus_det=nexus_det,
                name="event_time_offset",
            )
            logger.debug(
                "Overwriting event_time_offset dataset %s", wrapped_time_offset
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
        "--crystal-rotation-file-path",
        type=str,
        default="",
        help="Path to the mcstas file to read crystal rotation from "
        "(if not provided, crystal rotation will not be inserted)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    parser.add_argument(
        "--wrap-event-time-offset",
        action="store_true",
        help="Wrap event time offsets to be within a single pulse period. "
        "**Note**: This option is a temporary workaround "
        "until the sampling implementation is updated.",
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

    insert_geometry_into_nexus(
        simulation_geometry,
        output_file_path,
        logger,
        crystal_rotation_file_path=args.crystal_rotation_file_path,
    )

    if args.wrap_event_time_offset:
        warnings.warn(
            "`--wrap-event-time-offset` is a temporary workaround "
            "until the sampling implementation is updated.",
            DeprecationWarning,
        )
        pulse_period = sc.scalar(1 / 14, unit="s")  # 14 Hz pulse frequency at ESS
        wrap_event_time_offsets(
            output_file_path,
            pulse_period=pulse_period,
            logger=logger,
        )


if __name__ == "__main__":
    main()
