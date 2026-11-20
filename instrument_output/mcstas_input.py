"""
McStas NeXus event lists as (file, dataset) pieces for the sampler.

An MPI run of McStas 3.99 writes the event lists of each MPI process to its own
``mccode_node<rank>.h5``; ``mccode.h5`` then holds each event list as an HDF5 virtual
dataset stitched from those node files. HDF5 reads parts whose source file it cannot
find as the fill value (zero weight), so a missing node file would silently drop
events. The virtual datasets are therefore resolved here and the node files read
directly; a missing or incomplete node file is an error.

The weights of every node file are normalised to the ray count of the whole run (the
sources normalise in INITIALIZE, before McStas splits the rays over the processes),
so the node files of one run add up: they are slices of one run, not repeats.
"""
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np


@dataclass(frozen=True)
class EventList:
    """One stored event list (rows p x y n id t) that the sampler reads in full"""
    file: str
    dataset: str
    rows: int


def _vds_prefixes() -> list[Path]:
    """Extra source directories HDF5 searches (HDF5_VDS_PREFIX, ':'-separated)"""
    prefix = os.environ.get("HDF5_VDS_PREFIX", "")
    return [Path(p) for p in prefix.split(os.pathsep) if p]


def _source_file(name: str, vds_file: Path) -> Path | None:
    """The source file of a virtual dataset, found as HDF5 would, or None"""
    if name == ".":
        return vds_file
    candidates = [Path(name)] if Path(name).is_absolute() else \
        [p / name for p in _vds_prefixes()] + [vds_file.parent / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def event_list_sources(filename: str | Path, dataset: str) -> list[EventList]:
    """
    The stored pieces of an event list: itself, or for a virtual dataset its sources.
    Raises FileNotFoundError when a source file cannot be found.
    """
    path = Path(filename).resolve()
    with h5py.File(path) as f:
        dset = f[dataset]
        if not dset.is_virtual:
            return [EventList(str(path), dataset, dset.shape[0])]
        if dset.ndim != 2:
            raise ValueError(f"{path}:{dataset}: expected a 2D event list, shape {dset.shape}")
        pieces, missing = [], []
        for source in dset.virtual_sources():
            if source.src_space.get_select_type() != h5py.h5s.SEL_ALL:
                raise ValueError(f"{path}:{dataset}: maps only part of {source.file_name}:"
                                 f"{source.dset_name}, not supported")
            try:
                (row0, _), (row1, _) = source.vspace.get_select_bounds()
                rows = row1 - row0 + 1
            except ValueError:  # empty selection: a process without events
                rows = 0
            source_file = _source_file(source.file_name, path)
            if source_file is None:
                missing.append(source.file_name)
                continue
            pieces.append(EventList(str(source_file), source.dset_name, rows))
    if missing:
        shown = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        raise FileNotFoundError(
            f"{path}:{dataset} is a virtual dataset, {len(missing)} of its {len(missing) + len(pieces)} "
            f"source files are missing (looked in {path.parent}"
            f"{' and HDF5_VDS_PREFIX' if _vds_prefixes() else ''}): {shown}")
    return pieces


def check_event_lists(pieces: Sequence[EventList]) -> None:
    """Every piece exists in its file with the rows the virtual dataset expects"""
    by_file = group_by_file(pieces)
    problems = []
    for filename, file_pieces in by_file.items():
        try:
            with h5py.File(filename) as f:
                for piece in file_pieces:
                    if piece.dataset not in f:
                        problems.append(f"{filename}: no dataset {piece.dataset}")
                    elif f[piece.dataset].shape[0] != piece.rows:
                        problems.append(f"{filename}:{piece.dataset} has {f[piece.dataset].shape[0]} rows, "
                                        f"the virtual dataset maps {piece.rows}")
        except OSError as err:
            problems.append(f"{filename}: cannot open ({err})")
    if problems:
        shown = "\n  ".join(problems[:10]) + ("\n  ..." if len(problems) > 10 else "")
        raise OSError(f"{len(problems)} event list source(s) unreadable or incomplete:\n  {shown}")


def group_by_file(pieces: Iterable[EventList]) -> dict[str, list[EventList]]:
    """Pieces grouped by file (first-seen order), so each file is opened once"""
    groups: dict[str, list[EventList]] = defaultdict(list)
    for piece in pieces:
        groups[piece.file].append(piece)
    return dict(groups)


def resolve_event_lists(inputs: Sequence[tuple[str | Path, Sequence[str]]],
                        logger: logging.Logger) -> list[EventList]:
    """
    (input file, event list datasets) pairs -> the stored pieces to read, checked.
    Virtual datasets (McStas MPI node files) are expanded to their sources.
    """
    pieces = []
    for filename, datasets in inputs:
        file_pieces = [p for d in datasets for p in event_list_sources(filename, d)]
        n_files = len({p.file for p in file_pieces})
        if n_files > 1 or (file_pieces and file_pieces[0].file != str(Path(filename).resolve())):
            logger.info(f"{filename}: {len(datasets)} event lists stored in {n_files} node files")
        pieces += file_pieces
    check_event_lists(pieces)
    return pieces


def sample_event_lists(pieces: Sequence[EventList], n: int, logger: logging.Logger,
                       exclude_ids=None) -> tuple[np.ndarray, float]:
    """
    Weighted sampling (by p, with replacement) of n events over all pieces.
    Returns the (id, t) records and the summed weight of all events read.
    """
    from .julia import jl  # Julia starts only when sampling

    total = sum(p.rows for p in pieces)
    if total == 0:
        raise ValueError("The event lists hold no events.")
    if n > total:
        logger.warning(f"Sampling {n} events from only {total} McStas events: "
                       f"repeated events, the statistics look better than they are.")
    groups = group_by_file(pieces)
    files = list(groups)
    datasets = [[p.dataset for p in groups[f]] for f in files]
    logger.info(f"Sampling {n} of {total} events from {len(pieces)} event lists in {len(files)} files "
                f"(Julia threads: {jl.Threads.nthreads()}) ...")
    t1 = time.perf_counter()
    kwargs = {} if exclude_ids is None else {"exclude": exclude_ids}
    sampled_jl, total_weight = jl.multifile_sample_with_weight(files, datasets, n, jl.AlgWRSWRSKIP(), **kwargs)
    logger.info(f"... done. Took {time.perf_counter() - t1:.2f} s.")
    return sampled_jl.to_numpy(), float(total_weight)
