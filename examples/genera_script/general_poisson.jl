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
using SpecialFunctions: digamma, trigamma

slurm_id = get_slurm_id() # get slurm ID, if on cluster

include(joinpath(
    @__DIR__,
    "..",
    "..",
    "examples",
    "poisreg",
    "PoisModel.jl"
))

include(joinpath(
    @__DIR__,
    "..",
    "..",
    "examples",
    "poisreg",
    "PoisModelUtils.jl"
))



gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=18, markerstrokecolor=:auto)

Random.seed!(slurm_id); # set seed for reproducibility, different seed for each slurm_id

# Simulate data from the Beta regression model with fixed parameter paths
using Random
using Distributions

# ==========================================================
# Simulate Poisson data with time-varying mean
# With regression covariates
# ==========================================================
Random.seed!(123)

## Simulate data from the different regression model with fixed parameter paths
T = 500;
β₀ = [2, 0]
p = size(β₀, 1)[1]

## Generate covariate
X = ones(T + 1)
X = hcat(X, simulateAR(T + 1, [0.7], 1.0))

## fixed parameter paths
function simulate_μ(X, T, p, link, β₀)
    β = zeros(T + 1, p)
    β[1, :] = β₀
    λtime = zeros(T + 1)

    for t in 2:(T+1)
        β[t, 1] = sin(2π * t / T)
        if t < T / 3
            β[t, 2] = 0
        else
            if t < ((2 / 3) * T)
                β[t, 2] = -1
            else
                β[t, 2] = 1
            end
        end

        λtime[t] = linkinv.(link[1], dot(β[t, :], X[t, :]))
        end

    return β[2:end, :], λtime[2:end]
end
 
## Poisson and Exponential regression model
link = (LogLinLink(),)
β, λtime = simulate_μ(X, T, p, link, β₀);
y = zeros(T)
for i in 1:T
    y[i] = rand.(Poisson.(λtime[i]))
end

PoisData = [y X[2:end,:] β λtime]


# Plot covariate paths
pltx = []
for j in 2:p
    push!(pltx, plot(X[:, j], title=L"X_%$(j-1)" * " path", lw=2, color=colors[3]))
end
plot(pltx..., layout=(p - 1, 1), size=(1200, 800), xguidefontsize=12, yguidefontsize=14,
    titlefontsize=18, margin=5mm)

# plot the parameter evolution path of the regression coefficients
plt = plot_param_path_poisreg(β)

# plot the evolution of the Poisson distribution parameters over time
plot_poisparam_evolution(λtime)

# plot the evolution of the Poisson density over time and the time series
plot_poisdensity_evolution(λtime, y)



# ==========================================================
# VARIABLES KEPT FOR COMPATIBILITY WITH LATER CODE
# ==========================================================
# There are no actual regressors.
# X contains only the intercept column.
Xcopy= copy(X)
X = Xcopy[2:end,:]
covSel = [[1,2]]
p = length(covSel[1])
q = 0
link = (LogLinLink(),)



## The prior for the state at time t=0 using priors on intercepts and Fisher info
priorparam = mean(y[1:20]) # prior guess for mean near t = 0
f(x) = priorparam - linkinv(link[1], x)
m = find_zero(f, 0.0)
μ₀ = [m; zeros(p - 1)]
κ₀ = 1.0 # Prior sample size for the state at time t=0, used to scale InvFisher
Σ₀ = :fisherinfo # Σ₀ = (1 / κ₀) * inv((1 / T) * Finfo) computed inside TVGLM_Gibbs()

dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=1)
priorSettings = (
    ϕ₀=0.5, κ₀=0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀= 3.0, ψ₀= 1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=μ₀, Σ₀=Σ₀,n₀ = 1, # Prior for βₜ at time t=0
);

modelSettings = (
    observation=observation,
    link=link,
    #link=(LogLinLink(),),

    # New for ILF
    
    condMean = condMean, 
    condCov=condCov,

    slrObs = nothing,

    innovModel=:dsp,   # choices: :dsp, :homogaussuniv
    α=1/2,
    β=1/2,
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);

algoSettings = (
    stateSamplingMethod=:ffbs_laplace, # Algorithm to sample the state
    nParticles=100,           # Number of particles if using PGAS
    nIter=2000,              # Number of iterations in the Gibbs sampler
    nBurn=2000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.01,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:none,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfoPois,# Fisher info
    nCalibScale=1000,         # No. iter to calibrate the scaling matrix :fullfixed case
    fixed_scaling = false,     # Should the scaling matrix be fixed across Gibbs iter?
    verbose=true,             # Whether to print verbose output during sampling.
);

dateVec = 1:T
keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
results = []
interpMethod = :linear
scaling = :none
nPerGroup = 5

## IPLF 
methodlabel = "IPLF"

# Poisson
#obsTransform = BetaSuffStatsAveraged()
obsTransform = IdentityTransform()
Y, _, _, groupSizes = splitEqualGroups(y, X, covSel, nPerGroup)

slrObs = prepare_observation_transform(
    obsTransform,
    Y,
    condMean,
    condCov,
    nPerGroup
)

algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_slr);
dataSettings = (y=y , X=X, covSel=covSel, nPerGroup=nPerGroup);
modelSettings = (;modelSettings...,slrObs = slrObs)

Random.seed!(678)
θpost, groupSizes, nFailure = GibbsTVGLM(dataSettings, priorSettings, modelSettings, algoSettings);
prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% of the simulated trajectories")

#size(θpost)
quant_paramtime_iplf = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);
plt_iplf = plot_param_path_poisreg(β)

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


#### Run from 5 seeds and plot 

## Laplace approximation - none
methodlabel = "Laplace-None"
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_laplace);
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);

θpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime_lanone = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);
PlotPostParamEvolution!(plt, quant_paramtime_lanone, methodlabel,groupSizes; dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=2, c=colors[3])
