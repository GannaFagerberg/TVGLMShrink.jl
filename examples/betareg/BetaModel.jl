#### Beta Regression model

using LinearAlgebra
using SpecialFunctions: digamma, trigamma

BetaMean(μ, ψ) = Beta(1.0e-15 + μ * ψ, 1.0e-15 + (1 - μ) * ψ)

## Set up the Beta regression model

observation(param, state, t) =
    @views product_distribution(
        BetaMean.(
            GLM.linkinv.(param.link[1], param.Z[1][t] * state[param.Zidx[1]]),
            linkinv.(param.link[2], param.Z[2][t] * state[param.Zidx[2]])
        )
    )
    
@views condMean(param, state, t) = linkinv.(param.link[1],
    param.Z[1][t] * state[param.Zidx[1]])

function condCov(param, state, t)
    @views begin
        μ = linkinv.(param.link[1], param.Z[1][t] * state[param.Zidx[1]])
        κ = linkinv.(param.link[2], param.Z[2][t] * state[param.Zidx[2]])

        μ = clamp.(μ, 1e-12, 1.0 - 1e-12)
        κ = max.(κ, 1e-10)

        variance_y = μ .* (1.0 .- μ) ./ (κ .+ 1.0)
    end

    return diagm(variance_y)
end

## Fisher info and scaling
# NOTE: this now returns the Fisher info, not the sqrt(inv(FisherInfo)) and it is 
# Fisher for the whole same of T observations, not per observation.

function fisher_beta_blocks(Xm, Xp, βm, βp, linkm::Link, linkp::Link)

    μ = GLM.linkinv.(linkm, Xm * βm)
    ϕ = linkinv.(linkp, Xp * βp)
    dμ = GLM.mueta.(linkm, Xm * βm)   # dμ/dηm
    dϕ = mueta.(linkp, Xp * βp)   # dϕ/dηp

    a = μ .* ϕ
    b = (1 .- μ) .* ϕ

    ψa = digamma.(a)
    ψb = digamma.(b)
    ψ1a = trigamma.(a)
    ψ1b = trigamma.(b)
    ψ1ϕ = trigamma.(ϕ)

    ∇²lₘ = -(ϕ .^ 2 .* (ψ1a .+ ψ1b))
    ∇²lᵩ = -(μ .^ 2 .* ψ1a .+ (1 .- μ) .^ 2 .* ψ1b .- ψ1ϕ)
    ∇²Bₘᵩ = ϕ .* μ .* ψ1a .+ ψa .- ϕ .* (1 .- μ) .* ψ1b .- ψb
    ∇²lₘᵩ = (ψa .- ψb) .- ∇²Bₘᵩ

    wm = -∇²lₘ .* dμ .^ 2
    wp = -∇²lᵩ .* dϕ .^ 2
    wmp = -∇²lₘᵩ .* dμ .* dϕ

    Hmm = XDiagX(Xm, wm)
    Hpp = XDiagX(Xp, wp)
    Hmp = XDiagZ(Xm, wmp, Xp)

    return [Hmm Hmp; Hmp' Hpp]
end


# Function that computes the Fisher info (not Scaling matrix) for all obs 
function FisherInfoBeta(param, μ, t)
    return fisher_beta_blocks(param.X[1], param.X[2], μ[1:size(param.X[1], 2)],
        μ[(size(param.X[1], 2)+1):end], param.link[1], param.link[2])
end


# Function that computes the Fisher info (not Scaling matrix) for all obs 
function FisherInfoBeta_local(param, μ, t)
    return fisher_beta_blocks(param.Z[t][1], param.Z[t][2], μ[1:size(param.Z[t][1], 2)],
        μ[(size(param.Z[t][1], 2)+1):end], param.link[1], param.link[2])
end
