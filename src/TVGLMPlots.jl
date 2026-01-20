function PlotPostParamEvolution!(plts, postquantiles, label; 
        dateVec = nothing, interval_style = :solid, sample_t0 = true, 
        kwargs...)

    # interval_style can be ":solid", ":dash" or ":shaded"
    alpha = 0.2 # transparency for shaded intervals

    T, p, nQuantiles = size(postquantiles)

    if nQuantiles != 3
        error("postquantiles must have 3 quantiles: lower, median, upper.")
    end

    # Set up time and date vectors
    if sample_t0
        T = T - 1
        timeVec = 0:T
    else 
        timeVec = 1:T
    end
    if isnothing(dateVec)
        dateVec = timeVec
    end
    if length(dateVec) != length(timeVec)
        error("Length of dateVec must match the time dimension of θpost.")
    end

    if interval_style == :shaded
        for j = 1:p
            plot!(plts[j], 0:T, postquantiles[:,j,2], fillrange = postquantiles[:,j,1], 
                label = nothing, alpha = alpha; kwargs...)
            plot!(plts[j], 0:T, postquantiles[:,j,2], fillrange = postquantiles[:,j,3],     
                label = nothing, alpha = alpha; kwargs...)
            plot!(plts[j], 0:T, postquantiles[:,j,2], label = label; kwargs...)
        end
    else
        for j = 1:p
            plot!(plts[j], 0:T, postquantiles[:,j,1], label = nothing, 
                linestyle = interval_style; kwargs...)
            plot!(plts[j], 0:T, postquantiles[:,j,3], label = nothing, 
                linestyle = interval_style; kwargs...)
            plot!(plts[j], 0:T, postquantiles[:,j,2], label = label; kwargs...)
        end
    end
    return plts
end