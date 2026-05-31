# Poisson Regression with Parameters Sampled from a DSP Prior

# In this example we explore the joint posterior in the Poisson regression model with parameters following independent dynamic shrinkage process priors.
#
# ```math
# \begin{align*}
#   y_t \vert \boldsymbol{x}_t &\sim \mathrm{Poisson}\big( \exp(\boldsymbol{x}_t^\top \boldsymbol{\beta}_t) \big) \\
# \boldsymbol{\beta}_t &= \boldsymbol{\beta}_{t-1} + \boldsymbol{\nu}_t, \quad \boldsymbol{\nu}_t \sim N\Big(\boldsymbol{0},\mathrm{Diag}(\exp(\boldsymbol{h}_t/2))\Big) \\
#   \boldsymbol{h}_t &= \boldsymbol{\mu} + \phi(\boldsymbol{h}_{t-1} -\boldsymbol{\mu}) + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim Z(\alpha,\alpha, 0, \sigma_\eta) \\
# \end{align*}
# ```
#

# If on SLURM cluster, get SLURM_ARRAY_TASK_ID, otherwise use ARGS for local testing
if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    slurm_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
else
    slurm_id = isempty(ARGS) ? 0 : parse(Int, ARGS[1])
end
println("slurm_id = $slurm_id")

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
cd(joinpath(@__DIR__, "../.."))
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using PDMats, LogExpFunctions
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim
using Utils: mvcolors as colors
using Roots
includet("PoisModel.jl")  # Load simulator, Fisher info and plotting for PoisReg

gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=18, markerstrokecolor=:auto)

Random.seed!(slurm_id); # set seed for reproducibility, different seed for each slurm_id


# Simulate data from the Poisson regression model with fixed parameter paths
T = 500;
nCov = 1;       # Total number of covariates, excluding the intercept
covSel = [[1, 2, 3]] # covariates for mean and precision, first covariate is intercept
p = length(covSel[1])
β₀ = [0.0, 0, -0.05]
invlink_dgp(x) = exp_lin(x) # Inverse link function for Poisson regression
σₑ = [1 5; 5 100];        # Noise std for the AR(1) processes that generate the covariates
mₑ = [0.0, 0.0];     # Mean for the AR(1) processes that generate the covariates
φ = 0.7
y, X, β, λtime = simulate_poisson_reg_data(T, p, covSel[1], invlink_dgp, φ, σₑ, mₑ);

## Plot the true parameter paths and the time series

# plot the parameter evolution path of the regression coefficients
plt = plot_param_path_poisreg(β)

# plot the evolution of the Poisson distribution parameters over time
plot_poisparam_evolution(λtime)

# plot the evolution of the Poisson density over time and the time series
plot_poisdensity_evolution(λtime, y)

## The prior for the state at time t=0 using priors on intercepts and Fisher info
priorparam_λ = mean(y[1:20]) # Prior for λ ∼ lognormal(m, σ²)
κ₀ = 1.0 # Prior sample size for the state at time t=0, used to scale InvFisher
μ₀, Σ₀ = prior_t0(priorparam_λ, FisherInfo, κ₀, p)


# Check that the prior 95% interval to see that they make sense
priorStd = sqrt.(diag(Σ₀))
println("Prior interval for the state at time t=0:")
[μ₀ .- 1.96 * priorStd μ₀ .+ 1.96 * priorStd]


## Set up the prior, model and algorithm settings

dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=1)
priorSettings = (
    ϕ₀=0.5, κ₀=0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀=3.0, ψ₀=1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=μ₀, Σ₀=Σ₀, # Prior for βₜ at time t=0
);

modelSettings = (
    observation=observation,
    staticParam=ParamPoisReg,
    condMean=condMean,
    condCov=condCov,
    innovModel=:dsp,   # choices: :dsp, :homogaussuniv
    α=1 / 2,
    β=1 / 2,
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);

algoSettings = (
    stateSamplingMethod=:ffbs_laplace, # Algorithm to sample the state
    nParticles=100,           # Number of particles if using PGAS
    nIter=2000,               # Number of iterations in the Gibbs sampler
    nBurn=2000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.0,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:full,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfo,    # Scaling for the state
);

keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
results = []
## Laplace approximation - full Fisher scaling
algoSettings = (; algoSettings..., stateSamplingMethod=:ffbs_laplace);

scaling = :diagonal
nPerGroup = 1
algoSettings = (; algoSettings..., scaling=scaling);
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);

θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

println("Laplace failed at $(100*nFailure[]/(algoSettings.nBurn+algoSettings.nIter))% of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

PlotPostParamEvolution!(plt, quant_paramtime,
    "Laplace($(nPerGroup))$(scalingLabel(scaling))",
    groupSizes; dateVec=nothing, interpMethod=:constant, plot_t0=keep_t0, interval_style=:dash, lw=3, c=colors[3])
