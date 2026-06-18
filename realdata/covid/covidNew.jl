# Poisson Regression with for Covid data

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
using PDMats, LogExpFunctions, CSV, DataFrames, Dates
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim
using Utils: mvcolors as colors
using Roots
includet("../../examples/poisreg/PoisModel.jl")

gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=18, markerstrokecolor=:auto)

Random.seed!(slurm_id); # set seed for reproducibility, different seed for each slurm_id
applName = "Pois_Covid" # name for saving results and figures

gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=18, markerstrokecolor=:auto)
mainFolder = @__DIR__
dataFolder = joinpath(@__DIR__, "data")
figFolder = joinpath(@__DIR__, "figs/")
resFolder = joinpath(@__DIR__, "results/")

## Load the covid data 
applName = "covid_water" # all save files with this prefix
df = CSV.read(dataFolder * "/CovidDataU.csv", DataFrame)
y = df.case
X = [ones(length(y)) df.wastewater df.temp]
lag = 7
y = y[lag+1:end]
X = X[1:end-lag, :]
covSel = [[1, 2, 3]]
T = length(y)
p = length(covSel[1])
invlink_dgp(x) = exp_lin(x) # Inverse link function for Poisson regression

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
    #staticParam=ParamPoisReg,
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
    nIter=10000,               # Number of iterations in the Gibbs sampler
    nBurn=3000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.0,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:full,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfo,    # Scaling for the state
);


gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=10,
    xtickfontsize=10, ytickfontsize=10, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=14, markerstrokecolor=:auto)

dateVec = 1:T
keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
results = []
interpMethod = :linear
scaling = :none
nPerGroup = 1

## PGAS
methodlabel = "PGAS"
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:pgas);
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);

θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime_pgas = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

titles = [L"\beta_{%$(j-1)}" for j in 1:p]
plt = PlotPostParamEvolution(quant_paramtime_pgas,
    "PGAS",
    groupSizes; dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:shaded, lw=2, c=colors[1], legend=:bottomleft)
plot(plt..., layout=(3, 1), size=(1200, 1000))


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

PlotPostParamEvolution!(plt, quant_paramtime_la,
    "Laplace",
    groupSizes; dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=2, c=colors[3])

plot(plt..., layout=(3, 1), size=(1200, 1000))


savefig(figFolder * "$(applName)_param_$(methodlabel)_$(algoSettings.scaling)_$(dataSettings.nPerGroup).svg")



## IPLF 
methodlabel = "IPLF"
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_slr);
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);

θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime_iplf = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

PlotPostParamEvolution!(plt, quant_paramtime_iplf,
    "IPLF",
    groupSizes; dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=2, c=colors[2])

savefig(figFolder * "$(applName)_param_$(methodlabel)_$(algoSettings.scaling)_$(dataSettings.nPerGroup)_withIPLF.svg")