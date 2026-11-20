"""
SNS MaNDi (BL-11B) output: an NXsnsevent file laid out like the facility's
``MANDI_<run>.nxs.h5`` (ADARA), which Mantid ``Load`` / ``LoadEventNexus`` reads.

Input: McStas NeXus output of the MaNDi model (mcstas_mandi), one Monitor_nD event list per
Anger camera, ``entry1/data/bank<N>_events_list_p_x_y_n_id_t/events`` with columns
``p x y n id t``; ``id`` is already the MaNDi pixel id (65536*bank + col*256 + row), ``t`` the
time of flight from the moderator in s, ``p`` in n/s.

Mapping to the SNS layout:

* events: ``entry/bank<N>_events`` (linked to ``entry/instrument/bank<N>``) with event_id
  (uint32), event_time_offset (float32, microsecond, not wrapped: as in the facility files the
  time of flight is counted from the pulse that emitted the neutron), event_index and
  event_time_zero per pulse.
* exposure: the n sampled events stand for ``T = n / rate`` seconds, with rate the summed
  McStas weight (n/s; averaged over input files, which are taken to be repeats of the same
  setting). T sets the number of pulses (T * chopper_frequency), the events are spread
  uniformly over them, and each pulse carries the proton charge of ``beam_power``.
* DASlogs: proton_charge and frequency per pulse; omega, chi, phi (goniometer, Mantid MaNDi
  convention), BeamPower, ProtonBeamEnergy, Wavelength, LambdaRequest, Speed1-3 as single
  values, from the McStas instrument parameters.
* instrument_xml: the IDF McStas embedded (``--IDF``, exact DetCal geometry), or ``--xml``.
* the McStas parameters and sampling bookkeeping go to ``entry/mcstas`` (NXcollection).

Not written: beam monitors (the McStas monitors are histograms without detector efficiency),
bank_error / bank_unmapped, users, chopper NXdisk_chopper groups.
"""
import argparse
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from .base import OutputFormat
from .julia import jl

PIXELS_PER_BANK = 65536
PROTON_ENERGY_MEV = 1300.0    # SNS linac, DASlogs/ProtonBeamEnergy of recent MaNDi runs
EPICS_EPOCH = datetime(1990, 1, 1, tzinfo=timezone.utc)
EVENT_LIST = re.compile(r"^bank(\d+)_events.*_list_p_x_y_n_id_t$")
TIME_TARGET = "/entry/DASlogs/frequency/time"
# McStas parameter defaults of the MaNDi model, used when a parameter is not in the file
DEFAULT_PARAMETERS = {"omega": 0.0, "chi": 0.0, "phi": 0.0, "chopper_frequency": 60.0,
                      "Lmin": 2.0, "Lmax": 4.16, "beam_power": 1.7}
LAU_LATTICE_KEYS = {"lattice_a": "lattice_a", "lattice_b": "lattice_b", "lattice_c": "lattice_c",
                    "lattice_aa": "lattice_alpha", "lattice_bb": "lattice_beta",
                    "lattice_cc": "lattice_gamma"}


@dataclass
class MANDISample:
    events: np.ndarray        # structured: f0 pixel id, f1 time of flight [s]
    total_weight: float       # summed McStas weight of all events read [n/s]
    n_files: int


def _str(group: h5py.Group, name: str, value: str) -> h5py.Dataset:
    """String dataset in the ADARA style: fixed-length bytes, shape (1,)"""
    data = str(value).encode()
    return group.create_dataset(name, data=np.array([data], dtype=f"S{max(len(data), 1)}"))


def _scalar(group: h5py.Group, name: str, value, dtype, units: str | None = None) -> h5py.Dataset:
    dset = group.create_dataset(name, data=np.array([value], dtype=dtype))
    if units:
        dset.attrs["units"] = units
    return dset


def _group(parent: h5py.Group, name: str, nx_class: str) -> h5py.Group:
    group = parent.create_group(name)
    group.attrs["NX_class"] = nx_class
    return group


def _array(group: h5py.Group, name: str, data: np.ndarray) -> h5py.Dataset:
    if data.size > 4096:
        return group.create_dataset(name, data=data, compression="gzip", compression_opts=4,
                                    chunks=(min(data.size, 1 << 20),))
    return group.create_dataset(name, data=data)


def _time_attributes(dset: h5py.Dataset, start: datetime) -> None:
    since_epoch = start - EPICS_EPOCH
    dset.attrs["units"] = "second"
    dset.attrs["offset"] = start.isoformat()
    dset.attrs["start"] = start.isoformat()
    dset.attrs["offset_seconds"] = np.uint32(int(since_epoch.total_seconds()))
    dset.attrs["offset_nanoseconds"] = np.uint32(since_epoch.microseconds * 1000)


def _nxlog(logs: h5py.Group, name: str, values, units: str | None, start: datetime,
           times: np.ndarray | h5py.Dataset | None = None) -> h5py.Group:
    """
    NXlog of values at times [s after start]: default a single value at t = 0; an existing
    time dataset is linked, not copied (the per-pulse logs share one, like ADARA files)
    """
    log = _group(logs, name, "NXlog")
    values = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if isinstance(times, h5py.Dataset):
        log["time"] = times
    else:
        times = np.zeros(1) if times is None else np.asarray(times, dtype=np.float64)
        _time_attributes(_array(log, "time", times), start)
    value = _array(log, "value", values)
    stats = {"average_value": values.mean(), "average_value_error": values.std(),
             "minimum_value": values.min(), "maximum_value": values.max()}
    for key, v in stats.items():
        dset = _scalar(log, key, v, np.float64)
        if units:
            dset.attrs["units"] = units
    if units:
        value.attrs["units"] = units
    return log


def read_mcstas_parameters(filename: str | Path) -> dict[str, str]:
    """McStas instrument parameters (entry1/simulation/Param) as strings"""
    with h5py.File(filename) as f:
        params = f.get("entry1/simulation/Param")
        if params is None:
            return {}
        out = {}
        for key, dset in params.items():
            value = dset[()]
            value = value[0] if isinstance(value, np.ndarray) else value
            out[key] = value.decode() if isinstance(value, bytes) else str(value)
        return out


def read_lau_lattice(path: Path) -> dict[str, float]:
    """Lattice parameters from the header of a McStas .lau / .laz file"""
    lattice = {}
    for line in open(path, errors="replace"):
        if not line.startswith("#"):
            break
        words = line[1:].split()
        if len(words) >= 2 and words[0] in LAU_LATTICE_KEYS:
            try:
                lattice[LAU_LATTICE_KEYS[words[0]]] = float(words[1])
            except ValueError:
                pass
    return lattice


class MANDIOutput(OutputFormat):
    name = "mandi"
    default_output_file = "MANDI_{run}.nxs.h5"

    def event_datasets(self, filename: str | Path) -> list[str]:
        """The per-bank event lists in a McStas NeXus file"""
        with h5py.File(filename) as f:
            data = f["entry1/data"]
            return sorted((f"entry1/data/{key}/events" for key in data if EVENT_LIST.match(key)),
                          key=lambda k: int(EVENT_LIST.match(k.split("/")[2]).group(1)))

    def sample(self, filenames: Sequence[str], n: int) -> MANDISample:
        filenames = [str(Path(f).resolve()) for f in filenames]
        datasets = [self.event_datasets(f) for f in filenames]
        total = 0
        for filename, file_datasets in zip(filenames, datasets):
            with h5py.File(filename) as f:
                total += sum(f[d].shape[0] for d in file_datasets)
            self.logger.info(f"{filename}: {len(file_datasets)} banks with events")
        if total == 0:
            raise ValueError(f"No bank event lists found in {filenames}")
        if n > total:
            self.logger.warning(f"Sampling {n} events from only {total} McStas events: "
                                f"repeated events, the statistics look better than they are.")
        self.logger.info(f"Sampling {n} of {total} events ...")
        t1 = time.perf_counter()
        sampled_jl, total_weight = jl.multifile_sample_with_weight(filenames, datasets, n, jl.AlgWRSWRSKIP())
        self.logger.info(f"... done. Took {time.perf_counter() - t1:.2f} s.")
        return MANDISample(events=sampled_jl.to_numpy(), total_weight=float(total_weight),
                           n_files=len(filenames))

    def _start_time(self) -> datetime:
        start = self.option("start_time")
        if start:
            start = datetime.fromisoformat(start)
            return start if start.tzinfo else start.astimezone()
        return datetime.now(timezone.utc).astimezone().replace(microsecond=0)

    def _lau_file(self, reflections: str | None, input_file: str) -> Path | None:
        candidates = [self.option("lau")] if self.option("lau") else []
        if reflections:
            candidates += [reflections, Path(input_file).parent / reflections,
                           Path(input_file).parent.parent / reflections]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return Path(candidate)
        return None

    def write(self, output_file: Path, sampled: MANDISample, input_files: Sequence[str]) -> None:
        params = read_mcstas_parameters(input_files[0])
        par = {k: float(params.get(k, v)) for k, v in DEFAULT_PARAMETERS.items()}
        missing = [k for k in DEFAULT_PARAMETERS if k not in params]
        if missing:
            self.logger.warning(f"McStas parameters {missing} not in {input_files[0]}, using defaults.")
        frequency = par["chopper_frequency"]
        run = str(self.option("run_number", 0))
        start = self._start_time()

        events = sampled.events
        n_events = len(events)
        rate = sampled.total_weight / sampled.n_files
        if rate <= 0:
            raise ValueError("Summed McStas weight is zero: no exposure time can be derived.")
        duration = n_events / rate
        n_pulses = max(1, int(round(duration * frequency)))
        charge_per_pulse = par["beam_power"] * 1E6 / (PROTON_ENERGY_MEV * 1E6) / frequency * 1E12  # pC
        self.logger.info(f"{n_events} events = {duration:.4g} s ({duration/3600:.3g} h) at "
                         f"{rate:.4g} n/s: {n_pulses} pulses of {charge_per_pulse:.4g} pC")

        rng = np.random.default_rng(self.option("seed"))
        pixel_ids = events["f0"].astype(np.uint32)
        tof_us = (events["f1"] * 1E6).astype(np.float32)
        pulses = rng.integers(0, n_pulses, size=n_events)
        banks_in_events = pixel_ids // PIXELS_PER_BANK
        instrument_xml = self.instrument_xml(input_files[0])
        banks = sorted(set(int(b) for b in re.findall(r'name="bank(\d+)"', instrument_xml))
                       | set(int(b) for b in np.unique(banks_in_events)))

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_file, "w") as f:
            f.attrs["NX_class"] = "NXroot"
            f.attrs["file_name"] = output_file.name
            f.attrs["file_time"] = datetime.now(timezone.utc).astimezone().isoformat()
            entry = _group(f, "entry", "NXentry")
            logs = _group(entry, "DASlogs", "NXcollection")

            frequency_log = _nxlog(logs, "frequency", np.full(n_pulses, frequency), "Hz", start,
                                   times=np.arange(n_pulses) / frequency)
            pulse_time = frequency_log["time"]
            pulse_time.attrs["target"] = TIME_TARGET
            _nxlog(logs, "proton_charge", np.full(n_pulses, charge_per_pulse), "picoCoulombs", start,
                   times=pulse_time)

            for name in ("omega", "chi", "phi"):
                _nxlog(logs, name, par[name], "deg", start)
            centre = 0.5 * (par["Lmin"] + par["Lmax"])
            _nxlog(logs, "Wavelength", centre, "A", start)
            _nxlog(logs, "LambdaRequest", centre, "A", start)
            _nxlog(logs, "BeamPower", par["beam_power"], "MW", start)
            _nxlog(logs, "ProtonBeamEnergy", PROTON_ENERGY_MEV, "MeV", start)
            for i in (1, 2, 3):
                _nxlog(logs, f"Speed{i}", frequency, "Hz", start)

            instrument = _group(entry, "instrument", "NXinstrument")
            _str(instrument, "beamline", "BL11B")
            _str(instrument, "name", "MANDI").attrs["short_name"] = "MANDI"
            _scalar(instrument, "target_station_number", 1, np.uint32)
            note = _group(instrument, "instrument_xml", "NXnote")
            _str(note, "data", instrument_xml)
            _str(note, "description", "XML contents of the instrument IDF")
            _str(note, "type", "text/xml")

            for bank in banks:
                select = banks_in_events == bank
                order = np.argsort(pulses[select], kind="stable")
                bank_pulses = pulses[select][order]
                detector = _group(instrument, f"bank{bank}", "NXdetector")
                fields = {
                    "event_id": _array(detector, "event_id", pixel_ids[select][order]),
                    "event_index": _array(detector, "event_index", np.searchsorted(
                        bank_pulses, np.arange(n_pulses)).astype(np.uint64)),
                    "event_time_offset": _array(detector, "event_time_offset", tof_us[select][order]),
                    "total_counts": _scalar(detector, "total_counts", select.sum(), np.uint64),
                }
                fields["event_time_offset"].attrs["units"] = "microsecond"
                detector["event_time_zero"] = pulse_time
                fields["event_time_zero"] = pulse_time
                event_group = _group(entry, f"bank{bank}_events", "NXevent_data")
                for key, dset in fields.items():
                    event_group[key] = dset
                    if key != "event_time_zero":
                        dset.attrs["target"] = f"/entry/instrument/bank{bank}/{key}"

            reflections = params.get("XtalReflections")
            sample = _group(entry, "sample", "NXsample")
            _str(sample, "name", Path(reflections).stem if reflections else "McStas sample")
            _str(sample, "nature", "Single Crystal")
            lau = self._lau_file(reflections, input_files[0])
            if lau:
                for key, value in read_lau_lattice(lau).items():
                    dset = _scalar(sample, key, value, np.float64)
                    if key.endswith(("alpha", "beta", "gamma")):
                        dset.attrs["units"] = "deg"
            elif reflections:
                self.logger.info(f"Reflection file {reflections} not found (--lau): no lattice in entry/sample")

            end = start + timedelta(seconds=duration)
            _str(entry, "definition", "NXsnsevent")
            _scalar(entry, "duration", duration, np.float32, "second")
            _str(entry, "start_time", start.isoformat())
            _str(entry, "end_time", end.isoformat())
            _str(entry, "run_number", run)
            _str(entry, "entry_identifier", run)
            _str(entry, "experiment_identifier", "McStas")
            _str(entry, "title", self.option("title") or f"McStas MaNDi {Path(input_files[0]).parent.name}")
            _str(entry, "notes", "Events sampled from a McStas simulation (stream_sampling_mcstas)")
            _scalar(entry, "proton_charge", charge_per_pulse * n_pulses, np.float64, "picoCoulombs")
            _scalar(entry, "total_counts", n_events, np.uint64)
            _scalar(entry, "total_uncounted_counts", 0, np.uint64)
            _scalar(entry, "total_other_counts", 0, np.uint64)
            total_pulses = _scalar(entry, "total_pulses", n_pulses, np.uint64)
            entry["raw_frames"] = total_pulses
            total_pulses.attrs["target"] = "/entry/total_pulses"

            mcstas = _group(entry, "mcstas", "NXcollection")
            for key, value in params.items():
                _str(mcstas, key, value)
            _str(mcstas, "input_files", " ".join(str(Path(p).resolve()) for p in input_files))
            _scalar(mcstas, "summed_weight", sampled.total_weight, np.float64, "counts/second")
            _scalar(mcstas, "event_rate", rate, np.float64, "counts/second")
            _scalar(mcstas, "sampled_events", n_events, np.uint64)

        self.logger.info(f"Wrote {output_file}: {len(banks)} banks, {n_events} events, run {run}")
