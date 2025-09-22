# McStas Stream Sampling to Nexus

Uses the [StreamSampling.jl](https://juliadynamics.github.io/StreamSampling.jl/stable/) package to do weighted event sampling from McStas event data in NeXus format, and generate a NeXus TOFRaw event file. 

Calling the StreamSampling Julia algorithm directly from Python (using `juliacall`) is about 30x faster than implementing a [similar algorithm in Python](https://github.com/aaronfinke/StreamSampling.py).

Also included: a dockerfile to generate a container that has all the needed packages.

## Requirements
- Julia installation and JuliaCall
    - Required Julia packages: StreamSampling, ChunkSplitters, Random, HDF5
    - After installing Julia, you can install them on the command line like so: 
    `julia -e 'using Pkg; Pkg.add(["StreamSampling", "ChunkSplitters", "Random", "HDF5"])'`
- h5py, numpy, matplotlib
- a json file describing metadata to save into the NeXus file
- McStas event file in NeXus format