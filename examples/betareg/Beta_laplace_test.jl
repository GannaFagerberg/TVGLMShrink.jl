# Beta regression with fixed path parameter evolution

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
cd(joinpath(@__DIR__, "../.."))
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using PDMats, LogExpFunctions
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim, get_slurm_id
using Utils: mvcolors as colors
using GLM
using Roots

slurm_id = get_slurm_id() # get slurm ID, if on cluster

include(joinpath(@__DIR__, "../..") * "/examples/betareg/BetaModel.jl") # BetaReg stuff
include(joinpath(@__DIR__, "../..") * "/examples/betareg/BetaModelUtils.jl") # BetaReg stuff

gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=18, markerstrokecolor=:auto)

Random.seed!(slurm_id); # set seed for reproducibility, different seed for each slurm_id

# Simulate data from the Beta regression model with fixed parameter paths
using Random
using Distributions

# ==========================================================
# Simulate Beta data with time-varying mean and precision
# No regression covariates
# ==========================================================

T = 500
nCov = 0

# Intercept/state only for mean and precision
covSel = [[1], [1]]

p = length(covSel[1])     # = 1, mean state
q = length(covSel[2])     # = 1, precision state

nState = p + q            # = 2

# ----------------------------------------------------------
# Link functions
# ----------------------------------------------------------

logistic(x) = 1 / (1 + exp(-x))

invlink = (
    x -> logistic(x),     # mean
    x -> exp(x)           # precision
)

# ==========================================================
# TRUE UNRESTRICTED STATE PATHS
# ==========================================================

# β[:,1] = unrestricted state controlling the Beta mean
# γ[:,1] = unrestricted state controlling the precision

β = zeros(T, p)
γ = zeros(T, q)

# Example time-varying paths
for t in 1:T

    # unrestricted mean state
    β[t, 1] =
        0.8 * sin(2π * t / 150)

    # unrestricted log-precision state
    #γ[t, 1] =1.5 + 0.5 * cos(2π * t / 200)
    # Precision state: step change
end

    #γ[1:250, 1]   .= log(10.0)
    #γ[251:end, 1] .= log(5.0)

    γ[1:250, 1]   .= log(1.5)
    γ[251:end, 1] .= log(3.0)   

# ==========================================================
# TRANSFORM TO BETA MEAN AND PRECISION
# ==========================================================

μtime = similar(β[:, 1])
ψtime = similar(γ[:, 1])

for t in 1:T
    μtime[t] = invlink[1](β[t, 1])
    ψtime[t] = invlink[2](γ[t, 1])
end

# Beta shape parameters
αtime = μtime .* ψtime
βtime = (1 .- μtime) .* ψtime

# ==========================================================
# SIMULATE OBSERVATIONS
# ==========================================================

y = Vector{Float64}(undef, T)

for t in 1:T
    y[t] = rand(Beta(αtime[t], βtime[t]))
end

# Numerical protection for log(y), log(1-y), etc.
y = clamp.(y, 1e-16, 1 - 1e-16)
plot(y)

# ==========================================================
# VARIABLES KEPT FOR COMPATIBILITY WITH LATER CODE
# ==========================================================
# There are no actual regressors.
# X contains only the intercept column.
X = ones(T, 1)

# Initial true unrestricted states, if needed later
β₀ = [β[1, 1]]
γ₀ = [γ[1, 1]]

# No covariate-generating AR processes
ρ  = Float64[]
σₑ = Float64[]
mₑ = Float64[]

y = clamp.(y, 1e-16, 1 - 1e-16) # Ensure y is in (0, 1) for Beta regression
## Plot the true parameter paths and the time series

# plot the parameter evolution path of the regression coefficients
plt = plot_param_path_betareg(β, γ)

# plot the evolution of the Beta distribution parameters over time
plot_betaparam_evolution(μtime, ψtime, αtime, βtime)

# plot the evolution of the Beta density over time and the time series
plot_betadensity_evolution(μtime, ψtime, y)

## The prior for the state at time t=0 using priors on intercepts and Fisher info
m = mean(y[1:20])
v = var(y[1:20])

priorparam = [m, m * (1 - m) / v - 1] # Prior for y₀ ∼ BetaMean(priorparam[1], priorparam[2])
#f_μ(x) = priorparam[1] - linkinv(link[1], x)
f_μ(x) = priorparam[1] - invlink[1](x)
β_m0 = [find_zero(f_μ, 0.0); zeros(p - 1)]

#f_ϕ(x) = priorparam[2] - linkinv(link[2], x)
f_ϕ(x) = priorparam[2] - invlink[2](x)
β_ϕ0 = [find_zero(f_ϕ, 0.0); zeros(q - 1)]

μ₀ = [β_m0; β_ϕ0]
κ₀ = 1.0 # Prior sample size for the state at time t=0, used to scale InvFisher
Σ₀ = :fisherinfo # Σ₀ = (1 / κ₀) * inv((1 / T) * Finfo) computed inside TVGLM_Gibbs()
link = (LogitLink(), LogLinLink())

μ₀_fixedμ = [
    β[1, 1],      # unused mean state
    β_ϕ0[1]       # initial log-precision
]

Σ₀_fixedμ = Diagonal([
    1e-8,         # nearly fixed unused mean state at t=0
    1.0           # uncertainty about log-precision
])


using SpecialFunctions: digamma, trigamma
using LinearAlgebra: Diagonal

nPerGroup = 5

# One state block: log-precision only
covSel_fixedμ = [[1]]

μ_fixed_groups = [
    μtime[i:min(i + nPerGroup - 1, T)]
    for i in 1:nPerGroup:T
]


@views function condMean_fixedμ(param, state, t)

    μ = μ_fixed_groups[t]

    # Precision is now the first and only state
    ηκ = param.Z[1][t] * state[param.Zidx[1]]
    ηκ = clamp.(ηκ, -20.0, 20.0)

    κ = clamp.(exp.(ηκ), 1e-8, 1e8)
    α = clamp.(μ .* κ, 1e-8, 1e8)

    return digamma.(α) .- digamma.(κ)
end


@views function condCov_fixedμ(param, state, t)

    μ = μ_fixed_groups[t]

    ηκ = param.Z[1][t] * state[param.Zidx[1]]
    ηκ = clamp.(ηκ, -20.0, 20.0)

    κ = clamp.(exp.(ηκ), 1e-8, 1e8)
    α = clamp.(μ .* κ, 1e-8, 1e8)

    v = trigamma.(α) .- trigamma.(κ)
    v = max.(v, 1e-10)

    return Matrix(Diagonal(vec(v)))
end

@views function condMean(param, state, t)

    ημ = param.Z[1][t] * state[param.Zidx[1]]
    ηκ = param.Z[2][t] * state[param.Zidx[2]]

    # Numerical bounds used only when evaluating sigma points
    ημ = clamp.(ημ, -30.0, 30.0)
    ηκ = clamp.(ηκ, -20.0, 20.0)

    μ = 1.0 ./ (1.0 .+ exp.(-ημ))
    κ = exp.(ηκ)

    α = clamp.(μ .* κ, 1e-8, 1e8)
    κ = clamp.(κ, 1e-8, 1e8)

    return digamma.(α) .- digamma.(κ)
end


@views function condCov(param, state, t)

    ημ = param.Z[1][t] * state[param.Zidx[1]]
    ηκ = param.Z[2][t] * state[param.Zidx[2]]

    ημ = clamp.(ημ, -30.0, 30.0)
    ηκ = clamp.(ηκ, -20.0, 20.0)

    μ = 1.0 ./ (1.0 .+ exp.(-ημ))
    κ = exp.(ηκ)

    α = clamp.(μ .* κ, 1e-8, 1e8)
    κ = clamp.(κ, 1e-8, 1e8)

    v = trigamma.(α) .- trigamma.(κ)
    v = max.(v, 1e-10)

    return Matrix(Diagonal(vec(v)))
end

## Set up the prior, model and algorithm settings
#μ₀_precision = [γ[1, 1]]
μ₀_precision = [γ[1, 1]]
Σ₀_precision = Diagonal([1.0])

dataSettings = (y=y, X=X, covSel=covSel_fixedμ, nPerGroup=1)

priorSettings = (
    ϕ₀=0.5, κ₀=0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀=3.0, ψ₀=1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=μ₀_precision, Σ₀=Σ₀_precision,n₀ = 1, # Prior for βₜ at time t=0
);

modelSettings = (
    observation=observation,
    #link=link,
    link=(LogLinLink(),),
    condMean = condMean_fixedμ,
    condCov  = condCov_fixedμ,
    innovModel=:dsp,   # choices: :dsp, :homogaussuniv
    α=1/2,
    β=1/2,
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);

algoSettings = (
    stateSamplingMethod=:ffbs_laplace, # Algorithm to sample the state
    nParticles=100,           # Number of particles if using PGAS
    nIter=3000,              # Number of iterations in the Gibbs sampler
    nBurn=2000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.0,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:none,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfoBeta,# Fisher info
    nCalibScale=1000,         # No. iter to calibrate the scaling matrix :fullfixed case
    fixed_scaling = false,     # Should the scaling matrix be fixed across Gibbs iter?
    verbose=true,             # Whether to print verbose output during sampling.
);


dateVec = 1:T
keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
results = []
interpMethod = :linear
scaling = :none
#nPerGroup =5

## IPLF 
methodlabel = "IPLF"

y_orig = copy(y)
y_log  = log.(y_orig)

algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_slr);
dataSettings = (y=y_log , X=X, covSel=covSel_fixedμ, nPerGroup=nPerGroup);

θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings, priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime_iplf = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

# Recreate the plot containing only the true paths
#plt_iplf = plot_param_path_betareg(β, γ)
plt_iplf = plot_param_path_betareg(β, γ)

# Add only the new IPLF posterior
PlotPostParamEvolution!(
    plt_iplf,
    quant_paramtime_iplf,
    "IPLF",
    groupSizes;
    dateVec=dateVec,
    interpMethod=interpMethod,
    plot_t0=keep_t0,
    interval_style=:solid,
    lw=2,
    c=colors[4]
)

display(plt_iplf)


## Laplace approximation 
methodlabel = "Laplace"
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_laplace);
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);

θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime_la = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

# Recreate the plot containing only the true paths
plt_la = plot_param_path_betareg(β, γ)

# Add only the new IPLF posterior
PlotPostParamEvolution!(
    plt_la,
    quant_paramtime_la,
    "IPLF",
    groupSizes;
    dateVec=dateVec,
    interpMethod=interpMethod,
    plot_t0=keep_t0,
    interval_style=:solid,
    lw=2,
    c=colors[4]
)

display(plt_la)
#savefig(figFolder * "$(applName)_param_$(methodlabel)_$(algoSettings.scaling)_$(dataSettings.nPerGroup).svg")

#=
ylims!(plt[1], (1.75, 2.75))
plot!(plt[1], legend=:bottomleft)
ylims!(plt[2], (-0.4, 0.4))
plot!(plt[2], legend=false)
ylims!(plt[3], (0.01, 0.06))
plot!(plt[3], legend=false)
=#

#savefig(figFolder * "$(applName)_param_$(methodlabel)_$(algoSettings.scaling)_$(dataSettings.nPerGroup)_withIPLF.svg")