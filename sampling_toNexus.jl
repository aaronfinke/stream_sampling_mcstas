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

function sample_frame_generic(filename, dataname, n, alg)
    rng = Xoshiro(rand(UInt))
    rs = ReservoirSampler{NTuple{2, Float64}}(rng, n, alg)
    chunksize = 5*10^5
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


function createMask(xrange::AbstractRange=590:690,yrange::AbstractRange=590:690,
                    detector_id::Int=1,grid_width::Int=1280,grid_height::Int=1280)
    x = collect(xrange)
    y = collect(yrange)
    offset = detector_id * grid_width * grid_height
    ids2d = y .* grid_width .+ x' .+ offset
    return Set(ids2d)
end

function sample_frame_mask(filename, index, rng, n, alg)
    rs = ReservoirSampler{NTuple{2, Float64}}(rng, n, alg)
    chunksize = 5*10^5
    mask_set = createMask()
    dataname = "entry1/data/Detector_$(index)_event_signal_dat_list_p_x_y_n_id_t/events"
    h5open(filename, "r") do file
        dset = file[dataname]
        totalsize = size(dset)[2]
        for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
            chunk_data = dset[:, ch]
            filtered_data = chunk_data[:,.!(in.(chunk_data[5,:], Ref(mask_set)))]
            for c in eachcol(filtered_data)
                fit!(rs, (c[5], c[6]), c[1])
            end
        end
    end
    return value(rs)
end

function sample_frame_mask_generic(filename, dataname, rng, n, alg)
    rs = ReservoirSampler{NTuple{2, Float64}}(rng, n, alg)
    chunksize = 5*10^5
    mask_set = createMask()
    h5open(filename, "r") do file
        dset = file[dataname]
        totalsize = size(dset)[2]
        for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
            chunk_data = dset[:, ch]
            filtered_data = chunk_data[:,.!(in.(chunk_data[5,:], Ref(mask_set)))]
            for c in eachcol(filtered_data)
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

function sample_frames_generic(filename, datasets, n, alg)
    results = Vector{Vector{NTuple{2, Float64}}}(undef, length(datasets))
    rngs = [Xoshiro(rand(UInt)) for _ in 1:Threads.nthreads()]
    Threads.@threads for (i, c) in enumerate(chunks(1:length(datasets), n=Threads.nthreads()))
        for j in c
            results[j] = sample_frame_generic(filename, datasets[j], rngs[i], n, alg)
        end
    end
    return results
end


function sample_all_frames(filename, indices, n, alg)
    rng = Xoshiro(rand(UInt))
    rs = [ReservoirSampler{NTuple{2, Float64}}(rng, n, alg) for _ in indices]
    chunksize = 5*10^5
    h5open(filename, "r") do file
        Threads.@threads for (i, index) in collect(enumerate(indices))
            @info "Sampling index $(index)..."
            dataname = "entry1/data/Detector_$(index)_event_signal_dat_list_p_x_y_n_id_t/events"
            dset = file[dataname]
            totalsize = size(dset)[2]
            for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
                for c in eachcol(dset[:, ch])
                    fit!(rs[i], (c[5], c[6]), c[1])
                end
            end       
        end
    end
    return value(merge(rs...))
end

function multifile_sample_all_frames(filenames, indices, n, alg)
    rng = Xoshiro(rand(UInt))
    rs = [ReservoirSampler{NTuple{2, Float64}}(rng, n, alg) for _ in indices]
    chunksize = 5*10^5
    for filename in filenames
        @info "Sampling filename $(filename)..."
        h5open(filename, "r") do file
            Threads.@threads for (i, index) in collect(enumerate(indices))
                @info "Sampling index $(index)..."
                dataname = "entry1/data/Detector_$(index)_event_signal_dat_list_p_x_y_n_id_t/events"
                dset = file[dataname]
                totalsize = size(dset)[2]
                for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
                    for c in eachcol(dset[:, ch])
                        fit!(rs[i], (c[5], c[6]), c[1])
                    end
                end       
            end
        end
    end
    return value(merge(rs...))
end


function sample_all_frames_generic(filename, datasets, n, alg)
    rng = Xoshiro(rand(UInt))
    rs = [ReservoirSampler{NTuple{2, Float64}}(rng, n, alg) for _ in datasets]
    chunksize = 5*10^5
    h5open(filename, "r") do file
        for (i, dataset) in collect(enumerate(datasets))
            @info "Sampling dataset $(dataset)..."
            dset = file[dataset]
            totalsize = size(dset)[2]
            for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
                for c in eachcol(dset[:, ch])
                    fit!(rs[i], (c[5], c[6]), c[1])
                end
            end       
        end
    end
    return value(merge(rs...))
end

function multifile_sample_all_frames_generic(filenames, datasets, n, alg)
    rng = Xoshiro(rand(UInt))
    rs = [ReservoirSampler{NTuple{2, Float64}}(rng, n, alg) for _ in datasets]
    chunksize = 5*10^5
    for filename in filenames
        h5open(filename, "r") do file
            for (i, dataset) in collect(enumerate(datasets))
                @info "Sampling dataset $(dataset)..."
                dset = file[dataset]
                totalsize = size(dset)[2]
                for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
                    for c in eachcol(dset[:, ch])
                        fit!(rs[i], (c[5], c[6]), c[1])
                    end
                end       
            end
        end
    end
    return value(merge(rs...))
end



function sample_all_frames_mask(filename, indices, n, alg)
    rng = Xoshiro(rand(UInt))
    rs = [ReservoirSampler{NTuple{2, Float64}}(rng, n, alg) for _ in indices]
    mask_set = createMask()
    chunksize = 5*10^5
    h5open(filename, "r") do file
        Threads.@threads for (i, index) in collect(enumerate(indices))
            @info "Sampling index $(index)..."
            dataname = "entry1/data/Detector_$(index)_event_signal_dat_list_p_x_y_n_id_t/events"
            dset = file[dataname]
            totalsize = size(dset)[2]
            if index == 1
                for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
                    chunk_data = dset[:, ch]
                    for c in eachcol(chunk_data)
                        !(c[5] in mask_set) && fit!(rs[i], (c[5], c[6]), c[1])
                    end
                end
            else
                for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
                    for c in eachcol(dset[:, ch])
                        fit!(rs[i], (c[5], c[6]), c[1])
                    end
                end  
            end              
        end
    end
    return value(merge(rs...))
end

function multifile_sample_all_frames_mask(filenames, indices, n, alg)
    rng = Xoshiro(rand(UInt))
    rs = [ReservoirSampler{NTuple{2, Float64}}(rng, n, alg) for _ in indices]
    mask_set = createMask()
    chunksize = 5*10^5
    for filename in filenames
        h5open(filename, "r") do file
            Threads.@threads for (i, index) in collect(enumerate(indices))
                @info "Sampling index $(index)..."
                dataname = "entry1/data/Detector_$(index)_event_signal_dat_list_p_x_y_n_id_t/events"
                dset = file[dataname]
                totalsize = size(dset)[2]
                if index == 1
                    for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
                        chunk_data = dset[:, ch]
                        for c in eachcol(chunk_data)
                            !(c[5] in mask_set) && fit!(rs[i], (c[5], c[6]), c[1])
                        end
                    end
                else
                    for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
                        for c in eachcol(dset[:, ch])
                            fit!(rs[i], (c[5], c[6]), c[1])
                        end
                    end  
                end              
            end
        end
    end
    return value(merge(rs...))
end

"""
    multifile_sample_with_weight(filenames, datasets, n, alg)

Weighted sampling of `(id, t)` from McStas event lists into one reservoir. `datasets[i]` lists
the event datasets of `filenames[i]` (banks without events are absent from a McStas file).
Also returns the summed weight `p` of all events read: with weights in n/s this is the event
rate, so `n / total_weight` is the measuring time the sample stands for.
"""
function multifile_sample_with_weight(filenames, datasets, n, alg)
    rng = Xoshiro(rand(UInt))
    rs = ReservoirSampler{NTuple{2, Float64}}(rng, n, alg)
    total_weight = 0.0
    chunksize = 5*10^5
    for (filename, file_datasets) in zip(filenames, datasets)
        h5open(String(filename), "r") do file
            for dataset in file_datasets
                @info "Sampling $(filename): $(dataset)..."
                dset = file[String(dataset)]
                totalsize = size(dset)[2]
                totalsize == 0 && continue
                for ch in chunks(1:totalsize, n=ceil(Int, totalsize/chunksize))
                    for c in eachcol(dset[:, ch])
                        fit!(rs, (c[5], c[6]), c[1])
                        total_weight += c[1]
                    end
                end
            end
        end
    end
    return value(rs), total_weight
end

# using BenchmarkTools
# @btime sample_frames("mccode.h5", 0:2, 10^5, AlgWRSWRSKIP());

