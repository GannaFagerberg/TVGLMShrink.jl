#### simulators and plotting for the Beta Regression model

using LinearAlgebra

## Simulate and plotting functions for the Beta regression model
function simulate_beta_reg_data(T, nCov, covSel, invlinkmean, invlinkprec,
    ρ, σₑ, mₑ, β₀, γ₀)

    X = ones(T + 1)
    for i = 1:nCov
        X = hcat(X, simulateAR(T + 1, ρ[i], σₑ[i], mₑ[i]))
    end
    Xmean = X[:, covSel[1]]
    Xprec = X[:, covSel[2]]

    p = length(covSel[1])
    q = length(covSel[2])
    y = zeros(T + 1)
    β = zeros(T + 1, p) # Store the regression parameters
    γ = zeros(T + 1, q) # Store the precision parameters
    β[1, :] = β₀
    γ[1, :] = γ₀
    μtime = zeros(T + 1)
    ψtime = zeros(T + 1)

    for t in 2:(T+1)
        β[t, 1] = sin(2π * t / T)
        if t < T / 3
            β[t, 2] = 0
        else
            if t < ((2 / 3) * T)
                β[t, 2] = -0.5
            else
                β[t, 2] = 1.0
            end
        end
        β[t, 3] = -0.05
        if t < T / 2
            γ[t, 1] = 1
        else
            γ[t, 1] = 3.0
        end
        μtime[t] = invlinkmean(Xmean[t, :] ⋅ β[t, :])
        ψtime[t] = invlinkprec(Xprec[t, :] ⋅ γ[t, :])
        y[t] = rand(BetaMean(μtime[t], ψtime[t]))
    end
    αtime = μtime .* ψtime
    βtime = (1 .- μtime) .* ψtime

    return y[2:end], X[2:end, :], β, γ, μtime, ψtime, αtime, βtime
end

# Plot the parameter evolution path of βₜ
function plot_param_path_betareg(β, γ)
    p = size(β, 2)
    q = size(γ, 2)
    plt = []
    for j = 1:p
        push!(plt, plot(β[:, j], label="true", xlabel="time, " * L"t",
            ylabel="", title=L"\beta_{%$(j-1)}", color=:black, lw=2))
    end
    for j = 1:q
        push!(plt, plot(γ[:, j], label="true", xlabel="time, " * L"t",
            ylabel="", title=L"\gamma_{%$(j-1)}", color=:black, lw=2))
    end
    plt = plot(plt..., layout=(3, 2), size=(1200, 1000), xguidefontsize=12,
        yguidefontsize=14, titlefontsize=20,
        legend=:bottomleft, margin=5mm)
    return plt
end

# plot \mu and \psi time series
function plot_betaparam_evolution(μtime, ψtime, αtime, βtime)

    p1 = plot(μtime, xlabel="time, " * L"t", title=L"\mu_t", lw=2,
        color=colors[1], legend=nothing)
    p2 = plot(ψtime, xlabel="time, " * L"t", title=L"\psi_t", lw=2,
        color=colors[3], legend=nothing)
    p3 = plot(αtime, xlabel="time, " * L"t", title=L"\alpha_t", lw=2,
        color=colors[2], legend=nothing)
    p4 = plot(βtime, xlabel="time, " * L"t", title=L"\beta_t", lw=2,
        color=colors[4], legend=nothing)
    plt = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800),
        xguidefontsize=12, yguidefontsize=14, titlefontsize=18, margin=5mm)
    return plt
end

# Plot the evolution of the Beta density over time and the time series
function plot_betadensity_evolution(μtime, ψtime, y)

    T = length(y)
    xgrid = 0.001:0.001:0.999
    pdfvals = zeros(T, length(xgrid))
    for t in 1:T
        pdfvals[t, :] = pdf.(BetaMean(μtime[t], ψtime[t]), xgrid)
    end
    pdfvals = pdfvals ./ maximum(pdfvals, dims=2) # Normalize for better color scale
    # plot a heatmap of the pdf evolution with logpdf scale for the colors
    p1 = heatmap(1:T, xgrid, pdfvals', clims=(0, 1), color=:Blues,
        ylabel="density", xlabel="time, " * L"t", colorbar=false,
        title="evolution of Beta density over time", colorbar_title="PDF (normalized)",
        size=(800, 600))

    p2 = plot(y, xlabel="time, " * L"t", ylabel=L"y_t", lw=1,
        color=colors[3], title="time series", legend=nothing)

    plt = plot(p1, p2, layout=(2, 1), size=(1200, 800),
        xguidefontsize=12, yguidefontsize=14, titlefontsize=18, margin=5mm)
    return plt

end