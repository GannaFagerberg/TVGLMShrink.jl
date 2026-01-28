function PlotPostParamEvolution!(plts, postquantiles, label, groupSizes = nothing; 
        dateVec = nothing, interval_style = :solid, sample_t0 = true, 
        kwargs...)

    # interval_style can be ":solid", ":dash" or ":shaded"
    alpha = 0.2 # transparency for shaded intervals

    Tgroup, p, nQuantiles = size(postquantiles) # Tgroup can include t=0
    if isnothing(groupSizes)
        groupSizes = ones(Int, Tgroup - sample_t0)
    end
    if nQuantiles != 3
        error("postquantiles must have 3 quantiles: lower, median, upper.")
    end
    
    if sample_t0
        # Initial obs is also duplicated by average (rounded) group size
        T0Group = ceil(Int, mean(groupSizes))
        groupSizes = [T0Group; groupSizes] 
        timeVec = -(T0Group-1):(sum(groupSizes) - T0Group)
    else
        timeVec = 1:sum(groupSizes)
    end
    

    # Set up date vectors
    if isnothing(dateVec)
        dateVec = timeVec
    end
    if length(dateVec) != length(timeVec)
        error("Length of dateVec must match the time dimension of θpost.")
    end

    for j = 1:p

        # Duplicate the group-level quantiles to observation-level
        # initialize empty matrix postquantiles_obs
        postquantiles_obs = Array{eltype(postquantiles)}(undef, 0, 3)
        for t_group in 1:length(groupSizes)
            postquantiles_obs = [
                postquantiles_obs;
                repeat(postquantiles[t_group, j, :]', groupSizes[t_group])
            ]
        end

        if interval_style == :shaded
            plot!(plts[j], timeVec, postquantiles_obs[:,2], 
                fillrange = postquantiles_obs[:,1], 
                label = nothing, alpha = alpha; kwargs...)
            plot!(plts[j], timeVec, postquantiles_obs[:,2], 
                fillrange = postquantiles_obs[:,3],     
                label = nothing, alpha = alpha; kwargs...)
            plot!(plts[j], timeVec, postquantiles_obs[:,2], label = label; kwargs...)
        else
            plot!(plts[j], timeVec, postquantiles_obs[:,1], label = nothing, 
                linestyle = interval_style; kwargs...)
            plot!(plts[j], timeVec, postquantiles_obs[:,3], label = nothing, 
                linestyle = interval_style; kwargs...)
            plot!(plts[j], timeVec, postquantiles_obs[:,2], label = label; 
                kwargs...)
        end
    end
    return plts
end