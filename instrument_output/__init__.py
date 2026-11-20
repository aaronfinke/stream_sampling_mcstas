"""
Instrument-specific output of the sampled McStas events.

    nmx    ESS NMX (json template, 3 panels, 14 Hz wrapped times, NeXus geometry)
    mandi  SNS MaNDi (NXsnsevent like the facility MANDI_<run>.nxs.h5, for Mantid)
"""
from .base import OutputFormat, get_instrument_xml_nexus
from .mandi import MANDIOutput
from .nmx import NMXOutput

FORMATS: dict[str, type[OutputFormat]] = {cls.name: cls for cls in (NMXOutput, MANDIOutput)}
