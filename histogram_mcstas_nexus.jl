using Distributed, OnlineStats, HDF5, Plots

addprocs(3)
@everywhere begin
    using OnlineStats,HDF5
    pixid_binsize = 1280*1280
    function min_max(filename,index)
        minmax_p=Extrema()
        minmax_t=Extrema()
        h5open(filename,"r") do file
            dataname = "entry1/data/Detector_$(index)_event_signal_dat_list_p_x_y_n_id_t/events"
            dset = file[dataname]
            for col in eachcol(dset[:,:])
                fit!(minmax_p,col[1])
                fit!(minmax_t,col[6])
            end
        end
        return (minmax_p,minmax_t)
    end
    filen = "/Users/aaronfinke/nmx_mcstas/lumi_output/rubredoxin_simple/rubr_5e10/mccode.h5"

    function histogram_tof_pixelid(filename, detector_index, tof_bins, pixid_binsize)
        # Create 2D histogram: pixel ID × TOF
        hist_data = Hist((1:pixid_binsize, tof_bins))
        hist_error = Hist((1:pixid_binsize, tof_bins))
        
        h5open(filename, "r") do file
            dataname = "entry1/data/Detector_$(detector_index)_event_signal_dat_list_p_x_y_n_id_t/events"
            dset = file[dataname]
            
            for row in eachrow(dset)
                prob = row[1]
                pixid = Int(row[5])
                tof = row[6]
                
                if 1 <= pixid <= pixid_binsize
                    fit!(hist_data, (pixid, tof), prob)
                    fit!(hist_error, (pixid, tof), prob^2)
                end
            end
        end
        
        return hist_data.counts, hist_error.counts
    end

    function histogram_tof_pixelid_v2(filename, detector_index, tof_bins, pixid_binsize)
        # Create 2D histogram: pixel ID × TOF
        nbins = length(tof_bins) - 1
        countmaps = [CountMap(Tuple{Int,Int}) for _ in 1:nbins]

        h5open(filename, "r") do file
            dataname = "entry1/data/Detector_$(detector_index)_event_signal_dat_list_p_x_y_n_id_t/events"
            dset = file[dataname]
            
            for col in eachcol(dset[:,:])
                prob = col[1]
                pixid = Int(col[5])
                tof = col[6]
                x = (pixid % 1280)
                y = div(pixid,1280)
                # left-closed, right-open binning: [e[i], e[i+1])
                b = searchsortedlast(tof_bins, tof)
                if 1 <= b <= nbins
                    fit!(countmaps[b], (x, y), prob)
                end
            end
        end
        
        return heatmaps
    end

    function countmap_to_matrix(cm::CountMap, xsize=1280, ysize=1280)
        mat = zeros(Float64, xsize, ysize)
        for ((x, y), weight) in value(cm)
            if 1 <= x <= xsize && 1 <= y <= ysize
                mat[x, y] = weight
            end
        end
        return mat
    end
        
end

a = pmap(x -> min_max(filen,x),0:2)
minmax_p = map(x -> x[1], a)
minmax_t = map(x -> x[2], a)

p = reduce(merge!,minmax_p)
t = reduce(merge!,minmax_t)

println("p: $(p)")
println("t: $(t)")
tof_bins=range(minimum(t),stop=maximum(t),length=50)
a = histogram_tof_pixelid_v2(filen,0,tof_bins,pixid_binsize)

# Plot a single TOF bin
heatmap(a[10].counts, xlabel="x", ylabel="y", title="TOF Bin 10")
