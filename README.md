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