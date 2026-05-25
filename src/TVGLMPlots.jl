function PlotPostParamEvolution!(plts, postquantiles, label, groupSizes=nothing;
    dateVec=nothing, interval_style=:solid, sample_t0=true, plot_t0=false,
    interpMethod=:linear, kwargs...)

    # interval_style can be ":solid", ":dash" or ":shaded"
    alpha = 0.2 # transparency for shaded intervals

    Tgroup, p, nQuantiles = size(postquantiles) # Tgroup can include t=0
    if isnothing(groupSizes)
        groupSizes = ones(Int, Tgroup - sample_t0)
    end
    if nQuantiles != 3
        error("postquantiles must have 3 quantiles: lower, median, upper.")
    end

    postquantiles_obs, dateVec = interpParam2Obs(postquantiles, groupSizes;
        sample_t0=sample_t0, output_t0=plot_t0, interpMethod=interpMethod)

    for j = 1:p

        if j == 1
            legendPos = :topright
        else
            legendPos = :none
        end
        if interval_style == :shaded
            plot!(plts[j], dateVec, postquantiles_obs[:, j, 2],
                fillrange=postquantiles_obs[:, 1],
                label=nothing, alpha=alpha; kwargs...)
            plot!(plts[j], dateVec, postquantiles_obs[:, j, 2],
                fillrange=postquantiles_obs[:, 3],
                label=nothing, alpha=alpha; kwargs...)
            plot!(plts[j], dateVec, postquantiles_obs[:, j, 2], label=label,
                legend=legendPos; kwargs...)
        else
            plot!(plts[j], dateVec, postquantiles_obs[:, j, 1],
                label=nothing, linestyle=interval_style; kwargs...)
            plot!(plts[j], dateVec, postquantiles_obs[:, j, 3],
                label=nothing, linestyle=interval_style; kwargs...)
            plot!(plts[j], dateVec, postquantiles_obs[:, j, 2],
                label=label, legend=legendPos; kwargs...)
        end
    end
    return plts
end