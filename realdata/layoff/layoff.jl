# US layoff proportions

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
cd(joinpath(@__DIR__, "../.."))
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using PDMats, LogExpFunctions
using Roots: find_zero
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim, get_slurm_id
using Utils: mvcolors as colors
using CSV, DataFrames, Dates, JLD2
using GLM

slurm_id = get_slurm_id() # get slurm ID, if on cluster

Random.seed!(slurm_id);

include(joinpath(@__DIR__, "../..") * "/examples/betareg/BetaModel.jl") # BetaReg stuff
include(joinpath(@__DIR__, "../..") * "/examples/betareg/BetaModelUtils.jl") # BetaReg stuff


gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=18, markerstrokecolor=:auto)
mainFolder = @__DIR__
dataFolder = joinpath(@__DIR__, "data")
figFolder = joinpath(@__DIR__, "figs/")
resFolder = joinpath(@__DIR__, "results/")



## observation data
applName = "layoff_vix" # all save files with this prefix
df = CSV.read(dataFolder * "/layoff_data.csv", DataFrame)
maxlag = 1 # maximum lag for the covariates, so we remove nLags first rows
df = df[(maxlag+1):end, :]
println("$(sum(ismissing.(eachrow(df)))) observations with missing values")
y = df.layoff_share # df.layoff_share
T = size(y, 1)
dates = df.date
X = [ones(T) df.credit_spr_lag1 df.vix_lag1 df.cpi_infl df.sentiment_lag1 df.log_oil_lag1]
# standardize covariates
#X[:, 2:end] = (X[:, 2:end] .- mean(X[:, 2:end], dims=1)) ./ std(X[:, 2:end], dims=1)
X = Matrix{Float64}(X)
dateVec = year.(df.date) .+ (month.(df.date) .- 1) ./ 12
covSel = [[1, 3], [1]]
p = length(covSel[1])
q = length(covSel[2])

# Link function for the mean and precision
link = (LogitLink(), LogLinLink())

## The prior for the state at time t=0 using priors on intercepts and Fisher info
## The prior for the state at time t=0 using priors on intercepts and Fisher info
priorparam = [0.15, exp(6)] # Best guess for y₀ ∼ BetaMean(priorparam_μ[1], priorparam_μ[2])
f(x) = priorparam[1] - linkinv(link[1], x)
g(x) = priorparam[2] - linkinv(link[2], x)
m = find_zero(f, 0.0)
s = find_zero(g, 0.0)
μ₀ = [m; zeros(p - 1); s; zeros(q - 1)]
κ₀ = 1.0 # Prior sample size for the state at time t=0, used to scale InvFisher
Σ₀ = :fisherinfo # Σ₀ = (1 / κ₀) * inv((1 / T) * Finfo) computed inside TVGLM_Gibbs()

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
    link=link, # link functions for mean and precision
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
    nIter=10000,              # Number of iterations in the Gibbs sampler
    nBurn=3000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.0,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:full,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfoBeta,# Fisher info
    nCalibScale=1000,         # No. iter to calibrate the scaling matrix :fullfixed case
    verbose=true,             # Whether to print verbose output during sampling.
);
gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14,
    titlefontsize=16, markerstrokecolor=:auto)

keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
results = []
interpMethod = :linear

# No scaling and nPerGroup = 3
scaling = :none
nPerGroup = 5

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
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

titles = vcat([L"\beta_{%$(j-1)}" for j in 1:p], [L"\gamma_{%$(j-1)}" for j in 1:q])
plt = PlotPostParamEvolution(quant_paramtime, methodlabel, groupSizes;
    titles=titles, dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:shaded, lw=1, c=colors[1])
plot(plt..., layout=(3, 1), size=(1000, 1200), margin=3mm)


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
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

PlotPostParamEvolution!(plt, quant_paramtime, methodlabel, groupSizes;
    dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=1, c=colors[3])
plot(plt..., layout=(3, 1), size=(1000, 1200), margin=3mm)

savefig(figFolder * "$(applName)_$(algoSettings.scaling)_$(dataSettings.nPerGroup)_dsp.svg")
savefig(figFolder * "$(applName)_$(algoSettings.scaling)_$(dataSettings.nPerGroup)_dsp.pdf")



## Homoscedastic Gaussian innovations

modelSettings = (
    observation=observation,
    link=link, # link functions for mean and precision
    condMean=condMean,
    condCov=condCov,
    innovModel=:homogaussuniv,   # choices: :dsp, :homogaussuniv
    α=1 / 2,
    β=1 / 2,
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);


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
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

titles = vcat([L"\beta_{%$(j-1)}" for j in 1:p], [L"\gamma_{%$(j-1)}" for j in 1:q])
plt = PlotPostParamEvolution(quant_paramtime, methodlabel, groupSizes;
    titles=titles, dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:shaded, lw=1, c=colors[1])
plot(plt..., layout=(3, 1), size=(1000, 1200), margin=3mm)


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
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

PlotPostParamEvolution!(plt, quant_paramtime, methodlabel, groupSizes;
    dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=1, c=colors[3])
plot(plt..., layout=(3, 1), size=(1000, 1200), margin=3mm)

savefig(figFolder * "$(applName)_$(algoSettings.scaling)_$(dataSettings.nPerGroup)_homo.svg")
savefig(figFolder * "$(applName)_$(algoSettings.scaling)_$(dataSettings.nPerGroup)_homo.pdf")
