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
using TVGLMShrink: PositiveHardLink
#pathof(TVGLMShrink)
#pathof(DynamicGlobalLocalShrinkage)

slurm_id = get_slurm_id() # get slurm ID, if on cluster

include(joinpath(@__DIR__, "../..") * "/examples/betareg/BetaModel.jl") # BetaReg stuff
include(joinpath(@__DIR__, "../..") * "/examples/betareg/BetaModelUtils.jl") # BetaReg stuff

gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=18, markerstrokecolor=:auto)

Random.seed!(slurm_id); # set seed for reproducibility, different seed for each slurm_id

# Simulate data from the Beta regression model with fixed parameter paths
T = 500;
nCov = 2;       # Total number of covariates, excluding the intercept
covSel = [[1, 2, 3], [1]] # covariates for mean and precision, first covariate is intercept
p = length(covSel[1])
q = length(covSel[2])
β₀ = [0.0, 0, -0.05]
γ₀ = [1]        # log precision intercept
#link = (LogitLink(), LogLink())
logistic(x) = 1 / (1 + exp(-x))
invlink = (x -> logistic(x), x -> exp(x))
ρ = [0.7, 0.7];      # AR(1) coefficients for the covariate processes
σₑ = [1, 10];        # Noise std for the AR(1) processes that generate the covariates
mₑ = [0.0, 0.0];     # Mean for the AR(1) processes that generate the covariates

#y, X, β, γ, μtime, ψtime, αtime, βtime = simulate_beta_reg_data(T, nCov, covSel,
 #   x -> linkinv(link[1], x), x -> linkinv(link[2], x), ρ, σₑ, mₑ, β₀, γ₀);

y, Xmx, β, γ, μtime, ψtime, αtime, βtime = simulate_beta_reg_data(T, nCov, covSel,
    invlink[1], invlink[2], ρ, σₑ, mₑ, β₀, γ₀);
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

link = (LogitLink(), LogLinLink()) # best so far
#link = (CauchitLink(),LogLinLink())
#link = (LogitLink(),WoodardLink(2.0))
#link = (UnitHardLink(),LogLinLink()) ### distrotrs inference
#link = (ProbitLink(),LogLinLink()) ### bad, underestimayes
#link = (CloglogLink(),LogLinLink()) ### bad, underestimayes
#CloglogLink

## Set up the prior, model and algorithm settings

demean = false
if demean
    for j in 2:size(X, 2)
    X[:, j] .= (Xmx[:, j] .- mean(Xmx[:, j]))
    end
else
    X = Xmx
end

#plot(X[:,3])
#plot!(Xmx[:,3],color ="red")

 #./ std(X[:, j])

dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=1)
priorSettings = (
    ϕ₀=0.5, κ₀=0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀=3.0, ψ₀=1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=μ₀, Σ₀=Σ₀,n₀ = 1, # Prior for βₜ at time t=0
);

modelSettings = (
    observation=observation,
    link=link,
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
    nIter=5000,              # Number of iterations in the Gibbs sampler
    nBurn=3000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.01,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:none,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfoBeta,# Fisher info
    nCalibScale=1000,         # No. iter to calibrate the scaling matrix :fullfixed case
    fixed_scaling = true,     # Should the scaling matrix be fixed across Gibbs iter?
    verbose=true,             # Whether to print verbose output during sampling.
);

dateVec = 1:T
keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
results = []
interpMethod = :linear
scaling = :none
nPerGroup = 5

## PGAS
methodlabel = "PGAS"
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:pgas);
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);

#Random.seed!(2)
θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime_pgas = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);
plt_reg = plot_param_path_betareg(β, γ)
titles = [L"\beta_{%$(j-1)}" for j in 1:p]
PlotPostParamEvolution!(plt_reg, quant_paramtime_pgas, "PGAS",
    groupSizes; dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:shaded, lw=2, c=colors[1], legend=:bottomleft)

## Laplace approximation 
#Random.seed!(2)
methodlabel = "Laplace"
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_laplace);
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);

θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
#plt = plot_param_path_betareg(β, γ)
quant_paramtime_la = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);
PlotPostParamEvolution!(plt_reg, quant_paramtime_la, "Laplace", groupSizes; dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=2, c=colors[4])
#savefig(figFolder * "$(applName)_param_$(methodlabel)_$(algoSettings.scaling)_$(dataSettings.nPerGroup).svg")

## IPLF 
methodlabel = "IPLF"
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_slr);
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);
@show modelSettings.link

#Random.seed!(2)
#θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings, priorSettings, modelSettings, algoSettings);
θpost, groupSizes,nFailure = GibbsTVGLM(dataSettings, priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
plt_reg = plot_param_path_betareg(β, γ)
quant_paramtime_iplf = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);
PlotPostParamEvolution!(plt_reg, quant_paramtime_iplf, "Laplace", groupSizes; dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=2, c=colors[4])
#savefig(plt_reg, joinpath(@__DIR__, "beta_reg_sim.pdf"))
#display(plt_reg)

# diag - good
# woodart in precision and others - not good, use loglin or log
# cauchy not good with IPLF - perhps because it need large thaeat values if mu is near boundraies

#savefig(figFolder * "$(applName)_param_$(methodlabel)_$(algoSettings.scaling)_$(dataSettings.nPerGroup)_withIPLF.svg")

