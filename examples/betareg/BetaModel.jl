#### Beta Regression model

using LinearAlgebra
using SpecialFunctions   # for trigamma in Fisher info

BetaMean(μ, ψ) = Beta(1.0e-15 + μ * ψ, 1.0e-15 + (1 - μ) * ψ)

## Set up the Beta regression model

# Static parameter - first field must always be Σᵥ::Vector{PDMat{T,S}}
mutable struct ParamBetaReg{T,S<:AbstractMatrix{T}}
    Σᵥ::Vector{PDMat{T,S}}
    Zmean::Vector{Matrix{T}}
    Zprec::Vector{Matrix{T}}
end

invlinkmean(x) = logistic(x) # inverse link function for μ in Beta regression
invlinkprecision(x) = exp(x) # inverse link function for ψ in Beta regression
observation(param, state, t) =
    product_distribution(
        BetaMean.(
            invlinkmean.(param.Zmean[t] * state[1:p]),
            invlinkprecision.(param.Zprec[t] * state[(p+1):(p+q)])
        )
    )
condMean(param, state, t) = invlinkmean.(param.Zmean[t] * state[1:p])
function condCov(param, state, t)
    μ = invlinkmean.(param.Zmean[t] * state[1:p])
    ψ = invlinkprecision.(param.Zprec[t] * state[(p+1):(p+q)])
    return diagm(μ .* (1 .- μ) ./ (1 .+ ψ))
end

## Fisher info and scaling
# NOTE: this now returns the Fisher info, not the sqrt(inv(FisherInfo)) and it is 
# Fisher for the whole same of T observations, not per observation.
function fisher_beta_blocks(Xm, Xp, βm, βp)

    T = size(Xm, 1)
    pm = size(Xm, 2)
    pp = size(Xp, 2)

    ηm = Xm * βm
    ηp = Xp * βp

    expηm = exp.(ηm)
    expηp = exp.(ηp)

    μ = 1.0 ./ (1.0 .+ exp.(-ηm))
    ϕ = expηp

    a = μ .* ϕ
    b = (1 .- μ) .* ϕ

    ψa = digamma.(a)
    ψb = digamma.(b)
    ψϕ = digamma.(ϕ)

    ψ1a = trigamma.(a)
    ψ1b = trigamma.(b)
    ψ1ϕ = trigamma.(ϕ)

    E1 = ψa .- ψϕ
    E2 = ψb .- ψϕ

    ∇Bₘ = (ψa .- ψb) .* ϕ
    ∇Bᵩ = μ .* ψa .+ (1 .- μ) .* ψb .- ψϕ

    dμ = μ .* (1 .- μ)
    dϕ = ϕ

    ∇lₘ = ϕ .* (E1 - E2) .- ∇Bₘ
    ∇lᵩ = μ .* E1 .+ (1 .- μ) .* E2 .- ∇Bᵩ

    ∇²lₘ = -(ϕ .^ 2 .* (ψ1a .+ ψ1b))

    ∇²lᵩ = -(μ .^ 2 .* ψ1a .+
             (1 .- μ) .^ 2 .* ψ1b .-
             ψ1ϕ)

    ∇²Bₘᵩ =
        ϕ .* μ .* ψ1a .+
        ψa .-
        ϕ .* (1 .- μ) .* ψ1b .-
        ψb

    ∇²lₘᵩ = E1 .- E2 .- ∇²Bₘᵩ

    Hmm = zeros(pm, pm)
    Hpp = zeros(pp, pp)
    Hmp = zeros(pm, pp)

    @inbounds for i in 1:T

        xm = @view Xm[i, :]
        xp = @view Xp[i, :]

        dμi = dμ[i]
        dϕi = dϕ[i]

        d2μi = dμi * (1 - 2μ[i])

        # βmβm block
        Hmm .+= (
            ∇²lₘ[i] * dμi^2 +
            ∇lₘ[i] * d2μi
        ) .* (xm * xm')

        # βpβp block
        Hpp .+= (
            ∇²lᵩ[i] * dϕi^2 +
            ∇lᵩ[i] * dϕi
        ) .* (xp * xp')

        # cross block
        Hmp .+= (
            ∇²lₘᵩ[i] * dμi * dϕi
        ) .* (xm * xp')
    end

    return -[Hmm Hmp; Hmp' Hpp]

end

# Function that computes the Fisher info (not Scaling matrix) for all obs 
function FisherInfo(θ, μ, t, Xmean, Xprec, p, q)
    return fisher_beta_blocks(Xmean, Xprec, μ[1:p], μ[(p+1):end])
end
FisherInfo(θ, μ, t) = FisherInfo(θ, μ, t,
    X[:, covSel[1]], X[:, covSel[2]], length(covSel[1]), length(covSel[2])
)

## Prior for initial value of the state
priorμ = [0.5, 0.5] # mean and precision in Beta dist for μ at t=0
# This is Z actually
function prior_t0(priorparam_μ, inflateFactor_ϕ, FisherInfo, κ₀, p, q)

    # μ
    f_μ(x) = priorparam_μ[1] - invlinkmean(x)
    β_m0 = [find_zero(f_μ, 0.0); zeros(p - 1)]

    f_ϕ(x) = priorparam_μ[1] - invlinkprecision(x)
    β_ϕ0 = [find_zero(f_ϕ, 0.0); zeros(q - 1)]

    μ₀ = [β_m0; β_ϕ0]

    Finfo = FisherInfo([], μ₀, 0)
    Σ₀ = (1 / κ₀) * inv((1 / T) * Finfo)

    return μ₀, Σ₀

end

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
