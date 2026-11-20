"""Common interface of the instrument output formats."""
import argparse
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Sequence

import h5py


def get_instrument_xml_nexus(
    file_path: str | Path,
    xml_path: str = "entry1/instrument/instrument_xml/data",
) -> str:
    """Mantid IDF embedded by McStas (``--IDF``) in its NeXus output"""
    with h5py.File(file_path) as fp:
        instrument_xml = fp[xml_path][...][0].decode('UTF-8')
    return instrument_xml


class OutputFormat(ABC):
    """
    Turns McStas event lists into an instrument's event NeXus file.

    ``sample`` draws events from the McStas output (weighted by ``p``), ``write`` stores
    them in the layout the instrument's data reduction reads. The object ``sample``
    returns is only meant to be passed to ``write`` / ``plots`` of the same format.
    """

    name: ClassVar[str]
    default_output_file: ClassVar[str]

    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        self.args = args
        self.logger = logger

    def option(self, name: str, default: Any = None) -> Any:
        """Command line option, or default when the calling script does not define it"""
        return getattr(self.args, name, default)

    def instrument_xml(self, input_file: str | Path) -> str:
        """Mantid IDF from ``--xml`` if given, else the one McStas embedded"""
        xml = self.option("xml")
        if xml:
            self.logger.info(f"Using geometry from xml file {xml}...")
            return ' '.join(open(xml).readlines())
        return get_instrument_xml_nexus(file_path=input_file)

    @abstractmethod
    def sample(self, filenames: Sequence[str], n: int) -> Any:
        """Draw n events from the McStas event file(s)"""

    @abstractmethod
    def write(self, output_file: Path, sampled: Any, input_files: Sequence[str]) -> None:
        """Write the sampled events to output_file"""

    def plots(self, sampled: Any, folder: Path, prefix: str = "") -> None:
        """Detector images of the sampled events (optional)"""
        self.logger.warning(f"No histogram/animation output for format '{self.name}'.")
