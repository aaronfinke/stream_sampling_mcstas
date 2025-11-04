using StreamSampling, ChunkSplitters, Random, HDF5, BenchmarkTools

function createMask(xrange::AbstractRange,yrange::AbstractRange,
                    detector_id::Int=1,grid_width::Int=1280,grid_height::Int=1280)::Set{Int}
    x = collect(xrange)
    y = collect(yrange)
    offset = detector_id * grid_width * grid_height
    ids2d = y .* grid_width .+ x' .+ offset
    return BitSet(ids2d)
end

function sample_frame_mask(filename, index, rng, n, alg)
    rs = ReservoirSampler{NTuple{2, Float64}}(rng, n, alg)
    chunksize = 5*10^5
    mask_set = createMask(590:690, 590:690)
    dataname = "entry1/data/Detector_1_event_signal_dat_list_p_x_y_n_id_t/events"
    h5open(filename, "r") do file
        dset = file[dataname]
        totalsize = size(dset)[2]
        println(ceil(Int, totalsize/chunksize))
        for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
            chunk_data = dset[:, ch]
            @inbounds for c in eachcol(chunk_data)
                !(c[5] in mask_set) && fit!(rs, (c[5], c[6]), c[1])
            end
        end
    end
    return value(rs)
end

function sample_frame(filename, index, rng, n, alg)
    rs = ReservoirSampler{NTuple{2, Float64}}(rng, n, alg)
    chunksize = 5*10^5
    dataname = "entry1/data/Detector_$(index)_event_signal_dat_list_p_x_y_n_id_t/events"
    h5open(filename, "r") do file
        dset = file[dataname]
        totalsize = size(dset)[2]
        for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
            for c in eachcol(dset[:, ch])
                fit!(rs, (c[5], c[6]), c[1])
            end
        end
    end
    return value(rs)
end


function sample_frames(filename, indices, n, alg)
    results = Vector{Vector{NTuple{2, Float64}}}(undef, length(indices))
    rngs = [Xoshiro(rand(UInt)) for _ in 1:Threads.nthreads()]
    Threads.@threads for (i, c) in enumerate(chunks(1:length(indices), n=Threads.nthreads()))
        for j in c
            if j == 1
                results[j] = sample_frame_mask(filename, indices[j], rngs[i], n, alg)
            else
                results[j] = sample_frame(filename, indices[j], rngs[i], n, alg)
            end
        end
    end
    return results
end

# using BenchmarkTools
# @btime sample_frames("mccode.h5", 0:2, 10^5, AlgWRSWRSKIP());

