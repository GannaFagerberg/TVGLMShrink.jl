
function PlotPostParamEvolution(postquantiles, label, groupSizes=nothing;
    titles=nothing, dateVec=nothing, interval_style=:solid, sample_t0=true, plot_t0=false,
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

    if isnothing(titles)
        titles = ["Parameter $j" for j in 1:p]
    end
    plts = [plot(xlabel="time, " * L"t", ylabel="", title=titles[j]) for j in 1:p]

    postquantiles_obs, dateVec = interpParam2Obs(postquantiles, groupSizes, dateVec;
        sample_t0=sample_t0, output_t0=plot_t0, interpMethod=interpMethod)

    for j = 1:p

        if j == 1
            legendPos = :topleft
        else
            legendPos = :none
        end
        if interval_style == :shaded
            plot!(plts[j], dateVec, postquantiles_obs[:, j, 2],
                fillrange=postquantiles_obs[:, j, 1],
                label=nothing, alpha=alpha; kwargs...)
            plot!(plts[j], dateVec, postquantiles_obs[:, j, 2],
                fillrange=postquantiles_obs[:, j, 3],
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

    if isempty(plts)
        plts = [plot(xlabel="time, " * L"t", ylabel="", title="Parameter $j",
            xguidefontsize=12, yguidefontsize=12, titlefontsize=18,
            legend=:topright, margin=5mm) for j in 1:p]
    end

    postquantiles_obs, dateVec = interpParam2Obs(postquantiles, groupSizes, dateVec;
        sample_t0=sample_t0, output_t0=plot_t0, interpMethod=interpMethod)

    for j = 1:p

        if j == 1
            legendPos = :topleft
        else
            legendPos = :none
        end
        if interval_style == :shaded
            plot!(plts[j], dateVec, postquantiles_obs[:, j, 2],
                fillrange=postquantiles_obs[:, j, 1],
                label=nothing, alpha=alpha; kwargs...)
            plot!(plts[j], dateVec, postquantiles_obs[:, j, 2],
                fillrange=postquantiles_obs[:, j, 3],
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


"""
    postPredCheckDistr(y, θ_obstime, distr, ygrid, X, covSel, link;
        dateVec=1:length(y), timePoints=1:length(y), thinFactor=10)

Posterior predictive check of the fitted distribution of the response over time.
    - `y`: the time series data
    - `θ_obstime`: T x p x nIter matrix of post samples of the parameters on the observation time scale.
    - `distr`: a function that takes the parameters as input and returns a distribution object (e.g. for Beta regression, distr(μ, ψ) = BetaMean(μ, ψ))
    - `ygrid`: a grid of y values for plotting the density
    - `X`: the covariate matrix used in the model, with dimensions T x nCov
    - `covSel`: a vector of vectors. covSel[j] are the column indices of X that are used for the j-th parameter of the distribution.
    - `link`: a tuple of link functions for each parameter
    - `dateVec`: a vector of dates corresponding to the time points, used for plotting
    - `timePoints`: a vector of time points to plot (e.g. 1:T or a subset of time points)
    - `thinFactor`: thin factor for the Gibbs samples.
"""

function postPredCheckDistr(y, θ_obstime, distr, ygrid, X, covSel, link;
    dateVec=1:length(y), timePoints=1:length(y), thinFactor=10, kwargs...)
    T, _, nIter = size(θ_obstime)
    nDistrParams = length(covSel)
    param_time = zeros(length(timePoints), nDistrParams, ceil(Int, nIter / thinFactor)) # Store μ and ψ for all t, in Betareg example
    ps = []
    start = 1
    for j in 1:nDistrParams
        push!(ps, length(covSel[j]))
        linPred = zeros(length(timePoints), nIter)
        for (t, time) in enumerate(timePoints)
            for (i, iter) in enumerate(1:thinFactor:nIter)
                param_time[t, j, i] = linkinv.(link[j], X[time, covSel[j]] ⋅ θ_obstime[time, start:(start+ps[j]-1), iter])
            end
        end
        start = start + ps[j]
    end
    pdfvals = zeros(length(timePoints), length(ygrid))
    for (t, time) in enumerate(timePoints)
        for i in 1:size(param_time, 3)
            pdfvals[t, :] += pdf(distr(param_time[t, :, i]...), ygrid)
        end
        pdfvals[t, :] = pdfvals[t, :] ./ size(param_time, 3) # Average over the thinned samples
    end
    pdfvals = pdfvals ./ maximum(pdfvals, dims=2) # Normalize for better color scale

    p1 = heatmap(dateVec[timePoints], ygrid, pdfvals', clims=(0, 1), color=:Blues,
        label="density", xlabel="time, " * L"t", ylabel="", colorbar=false,
        size=(800, 600); kwargs...)

    plot!(p1, dateVec[timePoints], y[timePoints], label="data", lw=2,
        legend=nothing; kwargs...)

    return p1

end