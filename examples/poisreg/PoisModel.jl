# Poisson regression model

# ### Set up the Poisson regression model
mutable struct ParamTvReg{T,S<:AbstractMatrix{T}}
    Σᵥ::Vector{PDMat{T,S}}
    Z::Vector{Matrix{T}}
end

observation(param, state, t) = product_distribution(Poisson.(invlink.(param.Z[t] * state)))
condMean(param, state, t) = invlink.(param.Z[t] * state)
condCov(param, state, t) = diagm(invlink.(param.Z[t] * state))


## Prior for initial value of the state
priorμ = [0.5, 0.5] # mean and precision in Poisson dist for μ at t=0
# This is Z actually
function prior_t0(priorparam_μ, inflateFactor_ϕ, FisherInfo, κ₀, p)

    ### Prior for state (exp)
    priorparam_μ =
        μₘ = priorparam_μ[1] # mean of the Poisson distribution for μ at t=0
    σ²ₘ = priorparam_μ[2]^2

    s² = log(σ²ₘ / (μₘ^2) + 1)
    m = log(μₘ) - s² / 2

    μ₀ = [m; zeros(p - 1)]
    Σₘ = inv(1 / T * X[:, 2:end]' * (exp.(X * βₘ) .* X[:, 2:end]))

    Σ₀ = [s² zeros(1, size(Σₘ, 2));
        zeros(size(Σₘ, 1), 1) Σₘ]
    Σ₀ = 0.5 * (Σ₀ + Σ₀')

    Finfo = FisherInfo([], μ₀, 0)
    nugget = 1e-8 * max(1.0, tr(Finfo) / size(Finfo, 1))
    Finfo = FisherInfo([], μ₀, 0) + nugget * I(length(μ₀)) # Fisher information for all data
    Σ₀ = (1 / κ₀) * inv((1 / T) * Finfo)

    return μ₀, Σ₀

end

## Scaling 
scaling = sqrt(Σ₀)
scaling = I(p)
function FisherInfo(θ, μ, t, Xm)
    if scalingType == :none
        return I(length(μ))
    end
    T = size(Xm, 1)
    if t > 1
        S = sqrt(inv((Xm' * Diagonal(exp.(Xm * μ)) * Xm) / T))
    else
        S = scaling
    end
    if scalingType == :diagonal
        return Diagonal(diag(S))
    end
    return S
end

# Function that computes the Fisher info (not Scaling matrix) for all obs 
function FisherInfo(θ, μ, t, X, p)
    return fisher_beta_blocks(Xmean, Xprec, μ[1:p], μ[(p+1):end])
end
FisherInfo(θ, μ, t) = FisherInfo(θ, μ, t,
    X[:, covSel[1]], X[:, covSel[2]], length(covSel[1]), length(covSel[2])
)

# Define function to simulate Poisson regression data with time-varying parameters
function simulate_poisson_reg_data(T, p, invlink, ρ, σₑ, mₑ, β₀)
    X = ones(T + 1)
    for i = 1:(p-1)
        X = hcat(X, simulateAR(T + 1, ρ[i], σₑ[i], mₑ[i]))
    end
    y = zeros(T + 1)
    β = zeros(T + 1, p)
    β[1, :] = β₀

    for t in 2:(T+1)
        β[t, 1] = sin(2π * t / T)
        if t < T / 3
            β[t, 2] = 0
        elseif t < ((2 / 3) * T)
            β[t, 2] = -1
        else
            β[t, 2] = 1
        end
        β[t, 3] = 0.05
        y[t] = rand(Poisson(invlink_dgp((X[t, :] ⋅ β[t, :]))))
    end
    return y[2:end], X[2:end, :], β
end