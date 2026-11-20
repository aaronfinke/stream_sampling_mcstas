# McStas Stream Sampling to Nexus

Uses the [StreamSampling.jl](https://juliadynamics.github.io/StreamSampling.jl/stable/) package to do weighted event sampling from [McStas](https://www.mcstas.org) event data in NeXus format, and generate a NeXus TOFRaw event file. 

Calling the StreamSampling Julia algorithm directly from Python (using `juliacall`) is about 30x faster than implementing a [similar algorithm in Python](https://github.com/aaronfinke/StreamSampling.py).

Also included: a Dockerfile and an apptainer .def file to generate containers that have all the needed packages.

## Requirements
- Julia installation and JuliaCall
    - Required Julia packages: StreamSampling, ChunkSplitters, Random, HDF5
    - After installing Julia, you can install them on the command line like so: 
    `julia -e 'using Pkg; Pkg.add(["StreamSampling", "ChunkSplitters", "Random", "HDF5"])'`
- h5py, numpy, matplotlib
- [scipp](https://scipp.github.io)
- a json file describing metadata to save into the NeXus file
- McStas event file in NeXus format

## Usage
`python /path/to/sampling_to_nexus.py -h`

for McStas files with multiple configurations: 
`python /path/to/sample_multiconfig_mcstas.py -h`

Note that for the multiconfiguration the configurations should be in the form `/entry1/data/Config${confignum}_Panel${detnum}_event_signal_dat_list_p_x_y_n_id_t/events` 

## McStas MPI runs (node files)
An MPI run of McStas 3.99 writes each process's event lists to `mccode_node<rank>.h5`; `mccode.h5`
holds the event lists as HDF5 virtual datasets over those files. Pass `mccode.h5` as usual (`-i`):
the virtual datasets are resolved and the node files read directly (`instrument_output/mcstas_input.py`).

- Node files are looked up next to `mccode.h5` (and in `HDF5_VDS_PREFIX`, as HDF5 does). A missing
  node file, a missing dataset in one, or a row count that differs from the virtual dataset is an
  error. (HDF5 itself would read missing parts as zeros, silently dropping events.)
- The node files of one run are slices of that run, not repeats: McStas sources normalise the
  weights to the whole run's ray count, so the weights of all node files add up. One `mccode.h5`
  counts as one run; several `-i` files are repeats of the same setting (MaNDi: rate = summed
  weight / number of `-i` files).
- Sampling is shared over the Julia threads (`PYTHON_JULIACALL_THREADS`), one reservoir per thread
  over its share of the files, merged at the end.

## Output formats
`sampling_to_nexus.py --format {nmx,mandi}` (default `nmx`). Each format is a class in
`instrument_output/` (`OutputFormat`: `sample()` draws the events, `write()` stores them):

- `nmx` (`NMXOutput`): ESS NMX, NeXus streaming layout from the json template (`-j`), three
  1280 x 1280 panels, event times wrapped to 14 Hz in ns, NeXus geometry from McStas.
  `sample_multiconfig_mcstas.py` uses the same class.
- `mandi` (`MANDIOutput`): SNS MaNDi (BL-11B), an `NXsnsevent` file laid out like the facility's
  `MANDI_<run>.nxs.h5`, for Mantid `Load` / `LoadEventNexus`. Input: McStas NeXus output of the
  MaNDi model (`bank<N>_events` Monitor_nD event lists, pixel ids `65536*bank + col*256 + row`).
  ```
  python sampling_to_nexus.py --format mandi -i run/mccode.h5 -n 50_000_000 --run-number 900001
  # -> MANDI_900001.nxs.h5
  ```
  - Events: `entry/bank<N>_events`, time of flight in µs counted from the pulse that emitted the
    neutron (not wrapped, as in the facility files).
  - Exposure: with McStas weights in n/s, the n sampled events correspond to `n / rate` seconds.
    That sets the number of 60 Hz pulses the events are spread over, and the proton charge
    (`beam_power` per pulse), so `NormaliseByCurrent` works. Several input files are treated
    as repeats of the same setting (an MPI run's node files are not, see above).
  - DASlogs: `proton_charge`, `frequency` (per pulse), `omega`, `chi`, `phi` (for
    `SetGoniometer(Axis0="omega,0,1,0,1", Axis1="chi,0,0,1,1", Axis2="phi,0,1,0,1")`),
    `Wavelength`, `BeamPower`, `Speed1-3`, all taken from the McStas parameters. The McStas
    parameters are also copied to `entry/mcstas`.
  - Geometry: the IDF McStas embedded (`--IDF`), or `-x`.
  - Not written: beam monitors (the McStas monitors are histograms without detector efficiency).
  - Mantid 6.14 with the `MANDI_Parameters.xml` that `DownloadInstrument` puts in `~/.mantid`
    fails on embedded IDFs ("Wrong type of parameter"). Real facility files fail the same way.
    `LoadEventNexus(..., LoadNexusInstrumentXML=False)` uses Mantid's own MaNDi IDF instead;
    follow it with `LoadIsawDetCal`.
## Geometry
Detector, source and sample poses are taken from the McStas component table
(`entry1/instrument/components/*/Position`, `Rotation` in the McStas NeXus output) and written
in the **sample frame**: origin at `sampleMantid`, axes of `sampleMantid`, so the beam at the
sample runs along +z. The Mantid IDF McStas writes keeps the global orientation (after NMX's
curved guide the beam is 0.3 deg off z) and rounds positions to 1 mm, so it is only used for the
pixel layout. With `--xml`, the poses are still read from the McStas input file; a bare IDF
without a McStas file falls back to the IDF poses and logs a warning.
