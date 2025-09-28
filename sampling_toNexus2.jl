"""
Helper functions for loading Nexus/HDF5 files into the StreamSampling.jl
interface.
"""

using StreamSampling, ChunkSplitters, Random, HDF5

"""
If the dataset is chunked, get the chunks.
Due to the limitations of HDF5.jl, this requires going into the 
H5D plist and extracting it directly.
"""
function get_chunk_dims(dset::HDF5.Dataset)
    dcpl = HDF5.API.h5d_get_create_plist(dset)   # takes the Dataset (converted to hid_t)
    try
        # number of dataset dimensions
        nd = length(size(dset))
        # allocate buffer of the HDF5 hsize_t type
        buf = Vector{HDF5.API.hsize_t}(undef, nd)
        # call the underlying HDF5 H5Pget_chunk via HDF5.jl wrapper
        ret = HDF5.API.h5p_get_chunk(dcpl, Cint(nd), pointer(buf))
        if ret < 0
            # not chunked or error
            return nothing
        end
        return Tuple(Int.(buf))   # convert to normal Ints
    finally
        # close the property list (H5Pclose)
        HDF5.API.h5p_close(dcpl)
    end
end

"""
The actual sampling for each frame.
"""
function sample_frame(filename, index, rng, n, alg; 
    chunk_size=(10^5,6), 
    dataname = "entry1/data/Detector_$(index)_event_signal_dat_list_p_x_y_n_id_t/events")

    rs = ReservoirSampler{NTuple{2, Float64}}(rng, n, alg)
    
    h5open(filename, "r") do file
        dset = file[dataname]
        
        # get chunk size of dataset
        dsize = size(dset)
        csize = get_chunk_dims(dset)
        println("Chunk dims for index $index: ",csize)
        if csize === nothing
            csize = chunk_size
        end
        infos = HDF5.get_chunk_info_all(dset)
        for ci in infos
            start = Tuple(ci.offset .+ 1)
            ranges = ntuple(i -> start[i]:min(start[i] + csize[i] - 1, dsize[i]), length(dsize))
            chunk = dset[ranges...]
            for c in eachcol(chunk)
                fit!(rs, (c[5], c[6]), c[1])
            end
        end
    end
    return value(rs)
end

"""
Call this in Python to do the sampling.
"""
function sample_frames(filename, indices, n, alg)
    results = Vector{Vector{NTuple{2, Float64}}}(undef, length(indices))
    rngs = [Xoshiro(rand(UInt)) for _ in 1:Threads.nthreads()]
    Threads.@threads for (i, c) in enumerate(chunks(1:length(indices), n=Threads.nthreads()))
        for j in c
            results[j] = sample_frame(filename, indices[j], rngs[i], n, alg)
        end
    end
    return results
end

# using BenchmarkTools
# @btime sample_frames(0:2, 10^5, AlgWRSWRSKIP());

