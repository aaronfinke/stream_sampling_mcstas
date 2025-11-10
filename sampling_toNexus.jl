using StreamSampling, ChunkSplitters, Random, HDF5, BenchmarkTools

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
            results[j] = sample_frame(filename, indices[j], rngs[i], n, alg)
        end
    end
    return results
end

function sample_all_frames(filename, indices, n, alg)
    rng = Xoshiro(rand(UInt))
    rs = ReservoirSampler{NTuple{2, Float64}}(rng, n, alg)
    chunksize = 5*10^5
    h5open(filename, "r") do file
        for index in indices
            print("Sampling index $(index)...")
            dataname = "entry1/data/Detector_$(index)_event_signal_dat_list_p_x_y_n_id_t/events"
            dset = file[dataname]
            totalsize = size(dset)[2]
            for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
                for c in eachcol(dset[:, ch])
                    fit!(rs, (c[5], c[6]), c[1])
                end
            end
        end
    end
    return value(rs)
end

# using BenchmarkTools
# @btime sample_frames("mccode.h5", 0:2, 10^5, AlgWRSWRSKIP());

