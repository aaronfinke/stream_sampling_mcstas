# McStas instrument geometry xml description related functions.
# Copied from ESSNMX
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np
import scipp as sc
import h5py
import pathlib

from defusedxml.ElementTree import fromstring
from numpy.typing import NDArray


_AXISNAME_TO_UNIT_VECTOR = MappingProxyType(
    {
        "x": sc.vector([1.0, 0.0, 0.0]),
        "y": sc.vector([0.0, 1.0, 0.0]),
        "z": sc.vector([0.0, 0.0, 1.0]),
    }
)


def axis_angle_to_quaternion(
    *, x: float, y: float, z: float, theta: sc.Variable
) -> NDArray:
    """Convert axis-angle to queternions, [x, y, z, w].

    Parameters
    ----------
    x:
        X component of axis of rotation.
    y:
        Y component of axis of rotation.
    z:
        Z component of axis of rotation.
    theta:
        Angle of rotation, with unit of ``rad`` or ``deg``.

    Returns
    -------
    :
        A list of (normalized) quaternions, [x, y, z, w].

    Notes
    -----
    Axis of rotation (x, y, z) does not need to be normalized,
    but it returns a unit quaternion (x, y, z, w).

    """

    w: sc.Variable = sc.cos(theta.to(unit="rad") / 2)
    xyz: sc.Variable = -sc.sin(theta.to(unit="rad") / 2) * sc.vector([x, y, z])
    q = np.array([*xyz.values, w.value])
    return q / np.linalg.norm(q)


def quaternion_to_matrix(*, x: float, y: float, z: float, w: float) -> sc.Variable:
    """Convert quaternion to rotation matrix.

    Parameters
    ----------
    x:
        x(a) component of quaternion.
    y:
        y(b) component of quaternion.
    z:
        z(c) component of quaternion.
    w:
        w component of quaternion.

    Returns
    -------
    :
        A 3x3 rotation matrix.

    """
    from scipy.spatial.transform import Rotation

    return sc.spatial.rotations_from_rotvecs(
        rotation_vectors=sc.vector(
            Rotation.from_quat([x, y, z, w]).as_rotvec(),
            unit="rad",
        )
    )


class _XML(Protocol):
    """XML element or tree type.

    Temporarily used for type hinting.
    Builtin XML type is blocked by bandit security check."""

    tag: str
    attrib: dict[str, str]

    def find(self, name: str) -> "_XML | None": ...

    def __iter__(self) -> "_XML": ...

    def __next__(self) -> "_XML": ...


def _check_and_unpack_if_only_one(xml_items: list[_XML], name: str) -> _XML:
    """Check if there is only one element with ``name``."""
    if len(xml_items) > 1:
        raise ValueError(f"Multiple {name}s found.")
    elif len(xml_items) == 0:
        raise ValueError(f"No {name} found.")

    return xml_items.pop()


def select_by_tag(xml_items: _XML, tag: str) -> _XML:
    """Select element with ``tag`` if there is only one."""

    return _check_and_unpack_if_only_one(list(filter_by_tag(xml_items, tag)), tag)


def filter_by_tag(xml_items: Iterable[_XML], tag: str) -> Iterable[_XML]:
    """Filter xml items by tag."""
    return (item for item in xml_items if item.tag == tag)


def filter_by_type_prefix(xml_items: Iterable[_XML], prefix: str) -> Iterable[_XML]:
    """Filter xml items by type prefix."""
    return (
        item for item in xml_items if item.attrib.get("type", "").startswith(prefix)
    )


def select_by_type_prefix(xml_items: Iterable[_XML], prefix: str) -> _XML:
    """Select xml item by type prefix."""

    cands = list(filter_by_type_prefix(xml_items, prefix))
    return _check_and_unpack_if_only_one(cands, prefix)


def find_attributes(component: _XML, *args: str) -> dict[str, float]:
    """Retrieve ``args`` as float from xml."""

    return {key: float(component.attrib[key]) for key in args}


@dataclass
class SimulationSettings:
    """Simulation settings extracted from McStas instrument xml description."""

    # From <defaults>
    length_unit: str  # 'unit' of <length>
    angle_unit: str  # 'unit' of <angle>
    # From <reference-frame>
    beam_axis: str  # 'axis' of <along-beam>
    handedness: str  # 'val' of <handedness>

    @classmethod
    def from_xml(cls, tree: _XML) -> "SimulationSettings":
        """Create simulation settings from xml."""
        defaults = select_by_tag(tree, "defaults")
        length_desc = select_by_tag(defaults, "length")
        angle_desc = select_by_tag(defaults, "angle")
        reference_frame = select_by_tag(defaults, "reference-frame")
        along_beam = select_by_tag(reference_frame, "along-beam")
        handedness = select_by_tag(reference_frame, "handedness")

        return cls(
            length_unit=length_desc.attrib["unit"],
            angle_unit=angle_desc.attrib["unit"],
            beam_axis=along_beam.attrib["axis"],
            handedness=handedness.attrib["val"],
        )


def _position_from_location(location: _XML, unit: str = "m") -> sc.Variable:
    """Retrieve position from location."""
    x, y, z = find_attributes(location, "x", "y", "z").values()
    return sc.vector([x, y, z], unit=unit)


def _rotation_angle(location: _XML, angle_unit: str = "degree") -> sc.Variable:
    """Retrieve rotation angle from location."""

    attribs = find_attributes(location, "axis-x", "axis-y", "axis-z", "rot")
    return sc.scalar(-attribs["rot"], unit=angle_unit)


def _rotation_vector(location: _XML) -> sc.Variable:
    """Retrieve rotation vector from location."""

    attribs = find_attributes(location, "axis-x", "axis-y", "axis-z", "rot")
    return sc.vector(value=[attribs[f"axis-{axis}"] for axis in "xyz"])


def _rotation_matrix_from_location(
    location: _XML, angle_unit: str = "degree"
) -> sc.Variable:
    """Retrieve rotation matrix from location."""

    attribs = find_attributes(location, "axis-x", "axis-y", "axis-z", "rot")
    x, y, z, w = axis_angle_to_quaternion(
        x=attribs["axis-x"],
        y=attribs["axis-y"],
        z=attribs["axis-z"],
        theta=sc.scalar(-attribs["rot"], unit=angle_unit),
    )
    return quaternion_to_matrix(x=x, y=y, z=z, w=w)


@dataclass
class DetectorDesc:
    """Detector information extracted from McStas instrument xml description."""

    # From <component type="MonNDtype-n" ...>
    component_type: str  # 'type'
    name: str
    id_start: int  # 'idstart'
    fast_axis_name: str  # 'idfillbyfirst'
    # From <type name="MonNDtype-n" ...>
    num_x: int  # 'xpixels'
    num_y: int  # 'ypixels'
    step_x: sc.Variable  # 'xstep'
    step_y: sc.Variable  # 'ystep'
    start_x: sc.Variable  # 'xstart'
    start_y: sc.Variable  # 'ystart'
    # From <location> under <component type="MonNDtype-n" ...>
    position: sc.Variable  # <location> 'x', 'y', 'z'
    # Calculated fields
    rotation_matrix: sc.Variable
    rotation_angle: sc.Variable
    rotation_vector: sc.Variable
    slow_axis_name: str
    fast_axis: sc.Variable
    slow_axis: sc.Variable

    @classmethod
    def from_xml(
        cls,
        *,
        component: _XML,
        type_desc: _XML,
        simulation_settings: SimulationSettings,
    ) -> "DetectorDesc":
        """Create detector description from xml component and type."""

        location = select_by_tag(component, "location")
        rotation_matrix = _rotation_matrix_from_location(
            location, simulation_settings.angle_unit
        )
        fast_axis_name = component.attrib["idfillbyfirst"]
        slow_axis_name = "xy".replace(fast_axis_name, "")

        length_unit = simulation_settings.length_unit

        # Type casting from str to float and then int to allow *e* notation
        # For example, '1e4' -> 10000.0 -> 10_000

        def lengthify(value: str | float) -> sc.Variable:
            return sc.scalar(float(value), unit=length_unit)

        def integerize(value: str | float) -> int:
            # String may need to be float first
            return int(float(value))

        return cls(
            component_type=type_desc.attrib["name"],
            name=component.attrib["name"],
            id_start=integerize(component.attrib["idstart"]),
            fast_axis_name=fast_axis_name,
            slow_axis_name=slow_axis_name,
            num_x=integerize(type_desc.attrib["xpixels"]),
            num_y=integerize(type_desc.attrib["ypixels"]),
            step_x=lengthify(type_desc.attrib["xstep"]),
            step_y=lengthify(type_desc.attrib["ystep"]),
            start_x=lengthify(type_desc.attrib["xstart"]),
            start_y=lengthify(type_desc.attrib["ystart"]),
            position=_position_from_location(location, simulation_settings.length_unit),
            rotation_matrix=rotation_matrix,
            rotation_vector=_rotation_vector(location),
            rotation_angle=_rotation_angle(location, simulation_settings.angle_unit),
            fast_axis=rotation_matrix * _AXISNAME_TO_UNIT_VECTOR[fast_axis_name],
            slow_axis=rotation_matrix * _AXISNAME_TO_UNIT_VECTOR[slow_axis_name],
        )

    @property
    def total_pixels(self) -> int:
        return self.num_x * self.num_y

    @property
    def slow_step(self) -> sc.Variable:
        return self.step_y if self.fast_axis_name == "x" else self.step_x

    @property
    def fast_step(self) -> sc.Variable:
        return self.step_x if self.fast_axis_name == "x" else self.step_y

    @property
    def num_fast_pixels_per_row(self) -> int:
        """Number of pixels in each row of the detector along the fast axis."""
        return self.num_x if self.fast_axis_name == "x" else self.num_y

    @property
    def detector_shape(self) -> tuple:
        """Shape of the detector panel. (num_x, num_y)"""
        return (self.num_x, self.num_y)


def _collect_detector_descriptions(tree: _XML) -> tuple[DetectorDesc, ...]:
    """Retrieve detector geometry descriptions from mcstas file."""
    type_list = list(filter_by_tag(tree, "type"))
    simulation_settings = SimulationSettings.from_xml(tree)

    def _find_type_desc(det: _XML) -> _XML:
        for type_ in type_list:
            if type_.attrib["name"] == det.attrib["type"]:
                return type_

        raise ValueError(
            f"Cannot find type {det.attrib['type']} for {det.attrib['name']}."
        )

    detector_components = [
        DetectorDesc.from_xml(
            component=det,
            type_desc=_find_type_desc(det),
            simulation_settings=simulation_settings,
        )
        for det in filter_by_type_prefix(filter_by_tag(tree, "component"), "MonNDtype")
    ]

    return tuple(sorted(detector_components, key=lambda x: x.id_start))


@dataclass
class SampleDesc:
    """Sample description extracted from McStas instrument xml description."""

    # From <component type="sampleMantid-type" ...>
    component_type: str
    name: str
    # From <location> under <component type="sampleMantid-type" ...>
    position: sc.Variable
    rotation_matrix: sc.Variable | None

    @classmethod
    def from_xml(
        cls, *, tree: _XML, simulation_settings: SimulationSettings
    ) -> "SampleDesc":
        """Create sample description from xml component."""
        source_xml = select_by_type_prefix(tree, "sampleMantid-type")
        location = select_by_tag(source_xml, "location")
        try:
            rotation_matrix = _rotation_matrix_from_location(
                location, simulation_settings.angle_unit
            )
        except KeyError:
            rotation_matrix = None

        return cls(
            component_type=source_xml.attrib["type"],
            name=source_xml.attrib["name"],
            position=_position_from_location(location, simulation_settings.length_unit),
            rotation_matrix=rotation_matrix,
        )

    def position_from_sample(self, other: sc.Variable) -> sc.Variable:
        """Position of ``other`` relative to the sample.

        All positions and distance are stored relative to the sample position.

        Parameters
        ----------
        other:
            Position of the other object in 3D vector.

        """
        return other - self.position


@dataclass
class SourceDesc:
    """Source description extracted from McStas instrument xml description."""

    # From <component type="Source" ...>
    component_type: str
    name: str
    # From <location> under <component type="Source" ...>
    position: sc.Variable

    @classmethod
    def from_xml(
        cls, *, tree: _XML, simulation_settings: SimulationSettings
    ) -> "SourceDesc":
        """Create source description from xml component."""
        source_xml = select_by_type_prefix(tree, "sourceMantid-type")
        location = select_by_tag(source_xml, "location")

        return cls(
            component_type=source_xml.attrib["type"],
            name=source_xml.attrib["name"],
            position=_position_from_location(location, simulation_settings.length_unit),
        )


def _construct_pixel_id(detector_desc: DetectorDesc) -> sc.Variable:
    """Pixel IDs for single detector."""
    start, stop = (
        detector_desc.id_start,
        detector_desc.id_start + detector_desc.total_pixels,
    )
    return sc.arange("id", start, stop, unit=None)


def _construct_pixel_ids(detector_descs: tuple[DetectorDesc, ...]) -> sc.Variable:
    """Pixel IDs for all detectors."""
    ids = [_construct_pixel_id(det) for det in detector_descs]
    return sc.concat(ids, "id")


def _pixel_positions(
    detector: DetectorDesc, position_offset: sc.Variable
) -> sc.Variable:
    """Position of pixels of the ``detector``.

    Position of each pixel is relative to the position_offset.
    """
    pixel_idx = sc.arange("id", detector.total_pixels)
    n_col = sc.scalar(detector.num_fast_pixels_per_row)

    pixel_n_slow = pixel_idx // n_col
    pixel_n_fast = pixel_idx % n_col

    fast_axis_steps = detector.fast_axis * detector.fast_step
    slow_axis_steps = detector.slow_axis * detector.slow_step

    return (
        (pixel_n_slow * slow_axis_steps)
        + (pixel_n_fast * fast_axis_steps)
        + detector.rotation_matrix
        * sc.vector(
            [detector.start_x.value, detector.start_y.value, 0.0],
            unit=position_offset.unit,
        )  # Detector pixel offset should also be rotated first.
    ) + position_offset


@dataclass
class McStasInstrument:
    simulation_settings: SimulationSettings
    detectors: tuple[DetectorDesc, ...]
    source: SourceDesc
    sample: SampleDesc

    @classmethod
    def from_xml(cls, tree: _XML) -> "McStasInstrument":
        """Create McStas instrument from xml."""
        simulation_settings = SimulationSettings.from_xml(tree)

        return cls(
            simulation_settings=simulation_settings,
            detectors=_collect_detector_descriptions(tree),
            source=SourceDesc.from_xml(
                tree=tree, simulation_settings=simulation_settings
            ),
            sample=SampleDesc.from_xml(
                tree=tree, simulation_settings=simulation_settings
            ),
        )

    def pixel_ids(self, *det_names: str) -> sc.Variable:
        """Pixel IDs for the detectors.

        If multiple detectors are requested, all pixel IDs will be concatenated along
        the 'id' dimension.

        Parameters
        ----------
        det_names:
            Names of the detectors to extract pixel IDs for.

        """
        detectors = tuple(det for det in self.detectors if det.name in det_names)
        return _construct_pixel_ids(detectors)

    def experiment_metadata(self) -> dict[str, sc.Variable]:
        """Extract experiment metadata from the McStas instrument description."""
        return {
            "sample_position": self.sample.position_from_sample(self.sample.position),
            "source_position": self.sample.position_from_sample(self.source.position),
            "sample_name": sc.scalar(self.sample.name),
        }

    def _detector_metadata(self, det_name: str) -> dict[str, sc.Variable]:
        try:
            detector = next(det for det in self.detectors if det.name == det_name)
        except StopIteration as e:
            raise KeyError(f"Detector {det_name} not found.") from e
        return {
            "fast_axis": detector.fast_axis,
            "slow_axis": detector.slow_axis,
            "origin_position": self.sample.position_from_sample(detector.position),
            "position": _pixel_positions(
                detector, self.sample.position_from_sample(detector.position)
            ),
            "detector_shape": sc.scalar(detector.detector_shape),
            "x_pixel_size": detector.step_x,
            "y_pixel_size": detector.step_y,
            "detector_name": sc.scalar(detector.name),
        }

    def detector_metadata(self, *det_names: str) -> dict[str, sc.Variable]:
        """Extract detector metadata from the McStas instrument description.

        If multiple detector is requested, all metadata will be concatenated along the
        'panel' dimension.

        Parameters
        ----------
        det_names:
            Names of the detectors to extract metadata for.

        """
        if len(det_names) == 1:
            return self._detector_metadata(det_names[0])
        detector_metadatas: dict[str, dict[str, sc.Variable]] = {
            det_name: self._detector_metadata(det_name) for det_name in det_names
        }
        # Concat all metadata into panel dimension
        metadata_keys: set[str] = set().union(
            *(set(detector_metadatas[det_name].keys()) for det_name in det_names)
        )
        return {
            key: sc.concat(
                [metadata[key] for metadata in detector_metadatas.values()], "panel"
            )
            for key in metadata_keys
        }


def read_mcstas_geometry_xml(
    file_path: str | pathlib.Path,
    xml_path: str = "entry/instrument/instrument_xml/data",
) -> McStasInstrument:
    """Retrieve geometry parameters from mcstas file"""
    if pathlib.Path(file_path).suffix == ".xml":
        with open(file_path, "r") as file:
            tree = fromstring(file.read())
    else:
        with h5py.File(file_path) as file:
            tree = fromstring(file[xml_path][...][0])

    return McStasInstrument.from_xml(tree)
