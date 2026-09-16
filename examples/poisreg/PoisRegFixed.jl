# Poisson Regression with Fixed parameter paths

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
using GLM
using Roots: find_zero

include("PoisModel.jl")  # PoisReg model and Fisher info
include("PoisModelUtils.jl")  # Load simulator, plotting for PoisReg


figFolder = joinpath(@__DIR__, "figures/")
resFolder = joinpath(@__DIR__, "results/")

gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=18, markerstrokecolor=:auto)

Random.seed!(slurm_id); # set seed for reproducibility, different seed for each slurm_id
applName = "PoisRegPathSync" # name for saving results and figures


# Simulate data from the Poisson regression model with fixed parameter paths
T = 500;
nCov = 1;       # Total number of covariates, excluding the intercept
covSel = [[1, 2, 3]] # covariates for mean and precision, first covariate is intercept
p = length(covSel[1])
link = (LogLinLink(),)
σₑ = [1 5; 5 100];        # Noise std for the AR(1) processes that generate the covariates
mₑ = [0.0, 0.0];     # Mean for the AR(1) processes that generate the covariates
φ = 0.5
β₀ = [0.0,0.0,0.05]
y, X, β, λtime = simulate_poisson_reg_data_fixed(T, p, covSel, link, φ, σₑ, mₑ, β₀);

## Plot the true parameter paths and the time series

# plot the parameter evolution path of the regression coefficients
plt = plot_param_path_poisreg(β)

# plot the evolution of the Poisson distribution parameters over time
plot_poisparam_evolution(λtime)

# plot the evolution of the Poisson density over time and the time series
plot_poisdensity_evolution(λtime, y)

## The prior for the state at time t=0 using priors on intercepts and Fisher info
priorparam = mean(y[1:20]) # Prior for λ ∼ lognormal(m, σ²)
κ₀ = 1.0 # Prior sample size for the state at time t=0, used to scale InvFisher
f(x) = priorparam - linkinv(link[1], x)
m = find_zero(f, 0.0)
μ₀ = [m; zeros(p - 1)]
n₀ = 1.0 # Prior sample size for the state at time t=0, used to scale InvFisher
Σ₀ = :fisherinfo # Σ₀ = (1 / n₀) * inv((1 / T) * Finfo) computed inside TVGLM_Gibbs()



## Set up the prior, model and algorithm settings
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=1)
priorSettings = (
    ϕ₀=0.5, κ₀=0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀=3.0, ψ₀=1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=μ₀, Σ₀=Σ₀,n₀
     # Prior for βₜ at time t=0
);

modelSettings = (
    observation=observation,
    link=link,
    condMean=condMean,
    condCov=condCov,

    slrObs = nothing,

    innovModel=:dsp,   # choices: :dsp, :homogaussuniv
    α=1 / 2,
    β=1 / 2,
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);

algoSettings = (
    stateSamplingMethod=:ffbs_laplace, # Algorithm to sample the state
    nParticles=100,           # Number of particles if using PGAS
    nIter=3000,              # Number of iterations in the Gibbs sampler
    nBurn=3000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.0,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:full,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfoPois,# Fisher info
    FisherInfoPrior =FisherInfoPois, 
    nCalibScale=1000,         # No. iter to calibrate the scaling matrix :fullfixed case
    fixed_scaling = true,
    verbose=true,             # Whether to print verbose output during sampling.
);


gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=10,
    xtickfontsize=10, ytickfontsize=10, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=14, markerstrokecolor=:auto)

thinFactor = 10 # thinning out draws before computing density scores
dateVec = 1:T
keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
interpMethod = :linear
scaling = :none
nPerGroup = 5
CRPSAll = []
energyScoreAll = []
variogramScoreAll = []
MethodLabels = []

algoSettings = (
    stateSamplingMethod=:ffbs_laplace, # Algorithm to sample the state
    nParticles=100,           # Number of particles if using PGAS
    nIter=1000,               # Number of iterations in the Gibbs sampler
    nBurn=1000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=100,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.00,         # Offset for Polya-Gamma variables in the update of h_t
    scaling=:none,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfoBeta,# Fisher info
    FisherInfoPrior = FisherInfoPois,    # prior
    nCalibScale=1000,         # No. iter to calibrate the scaling matrix :fullfixed case
    fixed_scaling = true,    # Should the scaling matrix be fixed across Gibbs iter?
    verbose=true,             # Whether to print verbose output during sampling.
);

methodlabel = "IPLF"
obsChoice   = :y
scaling     = :fulllocal
FisherInfo  = FisherInfoPois
nPerGroup   = 5

obsTransform =
    if obsChoice === :y
        IdentityTransform()
    elseif obsChoice === :logy
        BetaSingleSuffStatGrouped(:logy)
    elseif obsChoice === :log1my
        BetaSingleSuffStatGrouped(:log1my)
    elseif obsChoice === :both
        BetaSuffStatsGrouped()
    else
        error("Unknown obsChoice = $obsChoice")
end

Y, _, _, groupSizes =splitEqualGroups(y,X,covSel,nPerGroup)
slrObs =prepare_observation_transform(obsTransform,Y,condMean,condCov,nPerGroup)

algoSettings_iplf = (;algoSettings...,scaling = scaling, nMaxIter=10, stateSamplingMethod = :ffbs_slr, FisherInfo=FisherInfo)
dataSettings_iplf = (y = y,X = X,covSel = covSel,nPerGroup = nPerGroup)
modelSettings_iplf = (;modelSettings...,slrObs = slrObs)

Random.seed!(20)
θpost_iplf, groupSizes_iplf, nFailure_iplf = GibbsTVGLM(dataSettings_iplf,priorSettings,modelSettings_iplf,algoSettings_iplf)
#θpost_iplf = copy(θpost_iplf_delta05_loglink_gr10)

# For me: without contrained version: covariances are singular, fails
# For me: constrained version, delta=0.5: better
# For me: constrained version, delta=0.5, diffrenet seed, smaller offsets: better
# Check why innovation gain is not PD at some iterations still
# Now running centered and delta 0.5 - bad
# weighted better

prcFailure_iplf = 100 * nFailure_iplf[] /(algoSettings_iplf.nBurn + algoSettings_iplf.nIter)
println("$(algoSettings_iplf.stateSamplingMethod) failed at ","$(prcFailure_iplf)% of the simulated trajectories")

param_quantiles = quantile_multidim(θpost_iplf,[0.025, 0.5, 0.975],dims = 3)
plt = plot_param_path_poisreg(β)
PlotPostParamEvolution!(plt,param_quantiles,"IPLF",groupSizes_iplf;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[4])
display(plt)


# ============================================================
# 2. LAPLACE
# ============================================================
methodlabel = "Laplace-None"
algoSettings_laplace = (;algoSettings...,scaling = scaling,stateSamplingMethod = :ffbs_laplace, FisherInfo=FisherInfo)
dataSettings_laplace = (y = y,X = X,covSel = covSel,nPerGroup = nPerGroup)
priorSettings_laplace = (;priorSettings..., n₀=1)

#Random.seed!(2)
θpost_laplace, groupSizes_laplace,  nFailure, nLaplaceFailure =GibbsTVGLM(dataSettings_laplace,priorSettings_laplace,modelSettings,algoSettings_laplace)
prcFailure_laplace =100 * nFailure[] /(algoSettings_laplace.nBurn + algoSettings_laplace.nIter)
println("$(algoSettings_laplace.stateSamplingMethod) failed at ","$(prcFailure_laplace)% of the simulated trajectories")
Y, Z, _, _ =splitEqualGroups(y,X,covSel,nPerGroup)
nLaplaceTotal =(algoSettings.nBurn + algoSettings.nIter) * length(Y)
prcLaplaceFailure =100 * nLaplaceFailure[] / nLaplaceTotal
println("Laplace mode optimization failed at ", round(prcLaplaceFailure, digits = 2),"% of the filtering updates.")

quant_paramtime_laplace =quantile_multidim(θpost_laplace,[0.025, 0.5, 0.975],dims = 3)
#plt_overlay = plot(layout = (3, 1),size = (900, 650),legend = :topright)
PlotPostParamEvolution!(plt,quant_paramtime_laplace,"Laplace",groupSizes_laplace;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[3])
display(plt)
#savefig(plt_overlay,joinpath(save_dir, "laplace_homo_gr1_layof_4000iters.pdf"))





#################
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

PlotPostParamEvolution!(plt, quant_paramtime_pgas, methodlabel,
    groupSizes; dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:shaded, lw=2, c=colors[1], legend=:bottomleft)

θ_obstime, _ = interpParam2Obs(θpost, groupSizes; sample_t0=true,
    interpMethod=interpMethod);
θ_obstime = θ_obstime[:, :, 1:thinFactor:end];

CRPS, energyScore, variogramScore = densityScores(θ_obstime, β)
push!(CRPSAll, CRPS)
push!(energyScoreAll, energyScore)
push!(variogramScoreAll, variogramScore)
push!(MethodLabels, methodlabel)

####################################
## Laplace approximation - none
####################################

methodlabel = "Laplace-None"
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_laplace);
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);

#θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure 
θpost, nFailure = GibbsTVGLM(dataSettings,priorSettings, modelSettings, algoSettings);

_, _, _, groupSizes = splitEqualGroups(y, X, covSel, nPerGroup)
prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime_lanone = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

PlotPostParamEvolution!(plt, quant_paramtime_lanone, methodlabel,
    groupSizes; dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=2, c=colors[3])

θ_obstime, _ = interpParam2Obs(θpost, groupSizes; sample_t0=true,
    interpMethod=interpMethod);
θ_obstime = θ_obstime[:, :, 1:thinFactor:end];

CRPS, energyScore, variogramScore = densityScores(θ_obstime, β)
push!(CRPSAll, CRPS)
push!(energyScoreAll, energyScore)
push!(variogramScoreAll, variogramScore)
push!(MethodLabels, methodlabel)


## Laplace approximation - diagonal
scaling = :diagonal
methodlabel = "Laplace-Diag"
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_laplace);
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);

θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings, priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime_ladiag = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

PlotPostParamEvolution!(plt, quant_paramtime_ladiag, methodlabel,
    groupSizes; dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=2, c=colors[2])

θ_obstime, _ = interpParam2Obs(θpost, groupSizes; sample_t0=true,
    interpMethod=interpMethod);
θ_obstime = θ_obstime[:, :, 1:thinFactor:end];

CRPS, energyScore, variogramScore = densityScores(θ_obstime, β)
push!(CRPSAll, CRPS)
push!(energyScoreAll, energyScore)
push!(variogramScoreAll, variogramScore)
push!(MethodLabels, methodlabel)


## Laplace approximation - full
scaling = :full
methodlabel = "Laplace-Full"
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_laplace);
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);

θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime_lafull = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

PlotPostParamEvolution!(plt, quant_paramtime_lafull, methodlabel,
    groupSizes; dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=2, c=colors[4])

θ_obstime, _ = interpParam2Obs(θpost, groupSizes; sample_t0=true,
    interpMethod=interpMethod);
θ_obstime = θ_obstime[:, :, 1:thinFactor:end];

CRPS, energyScore, variogramScore = densityScores(θ_obstime, β)
push!(CRPSAll, CRPS)
push!(energyScoreAll, energyScore)
push!(variogramScoreAll, variogramScore)
push!(MethodLabels, methodlabel)


## IPLF - none
scaling = :none
methodlabel = "IPLF"
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_slr);
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);

#θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure
θpost, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime_iplf = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

PlotPostParamEvolution!(plt, quant_paramtime_iplf, methodlabel,
    groupSizes; dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=2, c=colors[2])

θ_obstime, _ = interpParam2Obs(θpost, groupSizes; sample_t0=true,
    interpMethod=interpMethod);
θ_obstime = θ_obstime[:, :, 1:thinFactor:end];

CRPS, energyScore, variogramScore = densityScores(θ_obstime, β)
push!(CRPSAll, CRPS)
push!(energyScoreAll, energyScore)
push!(variogramScoreAll, variogramScore)
push!(MethodLabels, methodlabel)

savefig(figFolder * "$(applName)_param_$(dataSettings.nPerGroup).svg")


## Plot density scores
nMethods = length(MethodLabels)

# plot the CRPS over time for each parameter
plt_crps = []
for j in 1:p
    (j == 1) ? legendPos = :topleft : legendPos = false
    push!(plt_crps, plot(1:T, CRPSAll[1][:, j], label=MethodLabels[1], lw=2,
        color=colors[1], title="CRPS for " * L"\beta_{%$(j-1)}", legend=legendPos))
    for i in 2:nMethods
        plot!(1:T, CRPSAll[i][:, j], label=MethodLabels[i], lw=2, color=colors[1+i])
    end
end
plot(plt_crps..., layout=(1, p), size=(1200, 400), xguidefontsize=12, yguidefontsize=14,
    titlefontsize=18, margin=5mm)
savefig(figFolder * "$(applName)_crps_$(dataSettings.nPerGroup).svg")

# plot the energy score over time
plot(1:T, energyScoreAll[1], label=MethodLabels[1], lw=2, color=colors[1],
    title="Energy Score")
for i in 2:nMethods
    plot!(1:T, energyScoreAll[i], label=MethodLabels[i], lw=2, color=colors[1+i])
end
plot!(size=(1200, 400), xguidefontsize=12, yguidefontsize=14, titlefontsize=18, margin=5mm)
savefig(figFolder * "$(applName)_energy_$(dataSettings.nPerGroup).svg")


# plot the variogram score over time
plot(1:T, variogramScoreAll[1], label=MethodLabels[1], lw=2, color=colors[1],
    title="Variogram Score")
for i in 2:nMethods
    plot!(1:T, variogramScoreAll[i], label=MethodLabels[i], lw=2, color=colors[1+i])
end
plot!(size=(1200, 400), xguidefontsize=12, yguidefontsize=14, titlefontsize=18, margin=5mm)
savefig(figFolder * "$(applName)_variogram_$(dataSettings.nPerGroup).svg")

