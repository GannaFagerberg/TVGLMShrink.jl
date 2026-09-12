# Beta regression with fixed path parameter evolution
using Revise
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

Random.seed!(12345)

## Simulate data from the different regression model with fixed parameter paths
T = 500;
β₀ = [2, 0]
p = size(β₀, 1)[1]

## Generate covariate
X_regressors = ones(T + 1)
X_regressors = hcat(X_regressors, simulateAR(T + 1, [0.7], 1.0))

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

#γ = [t < T / 2 ? 1 : 3 for t in 1:T]
γ = [t < T / 2 ? 1 : 9 for t in 1:T]

ψtime = exp.(γ)
BetaMean(μ, ψ) = Beta(1.0e-15 + μ * ψ, 1.0e-15 + (1 - μ) * ψ)
invlink_logit = (LogitLink(),)

β,μtime = simulate_μ(X_regressors, T, p, invlink_logit, β₀)

y = [rand(BetaMean(μtime[t],ψtime[t])) for t in 1:T]
y = clamp.(y, 1e-16, 1-1e-16)
#BetaData = [y X[2:end,:] β γ μtime ψtime ]

# Beta shape parameters
αtime = μtime .* ψtime
βtime = (1 .- μtime) .* ψtime

# plot the parameter evolution path of the regression coefficients
plt = plot_param_path_betareg(β, γ)

# plot the evolution of the Beta distribution parameters over time
plot_betaparam_evolution(μtime, ψtime, αtime, βtime)

# plot the evolution of the Beta density over time and the time series
plot_betadensity_evolution(μtime, ψtime, y)


# ----------------------------------------------------------
# Link functions
# ----------------------------------------------------------

link = (LogitLink(), LogLinLink()) # best so far
#invlink = (x -> logistic(x), x -> exp(x))


# ==========================================================
# SIMULATE OBSERVATIONS
# ==========================================================
sum(y==1)
sum(y==0)

y = clamp.(y, 1e-12, 1 - 1e-12)
plot(y)


# ==========================================================
# VARIABLES KEPT FOR COMPATIBILITY WITH LATER CODE
# ==========================================================
# There are no actual regressors.
# X contains only the intercept column.

X        = X_regressors[2:end,:]
covSel   = [[1,2], [1]]
p        = length(covSel[1])
q        = length(covSel[2])

# No covariate-generating AR processes
#ρ  = Float64[]
#σₑ = Float64[]
#mₑ = Float64[]

## The prior for the state at time t=0 using priors on intercepts and Fisher info

## The prior for the state at time t=0 using priors on intercepts and Fisher info
m = mean(y[1:20])
v = var(y[1:20])
priorparam = [m, m * (1 - m) / v - 1] # Prior for y₀ ∼ BetaMean(priorparam[1], priorparam[2])
#f_μ(x) = priorparam[1] - linkinv(link[1], x)
f_μ(x) = priorparam[1] - linkinv(link[1], x)
β_m0 = [find_zero(f_μ, 0.0); zeros(p - 1)]

f_ϕ(x) = priorparam[2] - linkinv(link[2], x)
#f_ϕ(x) = priorparam[2] - invlink[2](x)
β_ϕ0 = [find_zero(f_ϕ, 0.0); zeros(q - 1)]

μ₀ = [β_m0; β_ϕ0]
#n₀ = 1.0 # Prior sample size for the state at time t=0, used to scale InvFisher
Σ₀ = :fisherinfo # Σ₀ = (1 / n₀) * inv((1 / T) * Finfo) computed inside TVGLM_Gibbs()
## Set up the prior, model and algorithm settings


@views function condMean(param, state, t)
    ημ = param.Z[1][t] * state[param.Zidx[1]]
    μ = linkinv.(param.link[1], ημ)
    μ = clamp.(μ, 1e-12, 1.0 - 1e-12)
    return μ
end

@views function condCov(param, state, t)
    ημ = param.Z[1][t] * state[param.Zidx[1]]
    ηκ = param.Z[2][t] * state[param.Zidx[2]]
    μ = linkinv.(param.link[1], ημ)
    κ = linkinv.(param.link[2], ηκ)
    μ = clamp.(μ, 1e-12, 1.0 - 1e-12)
    κ = max.(κ, 1e-10)
    variance_y = μ .* (1.0 .- μ) ./ (κ .+ 1.0)
    return Matrix(Diagonal(vec(variance_y)))
end


observation(param, state, t) =
    @views product_distribution(
        BetaMean.(
            GLM.linkinv.(param.link[1], param.Z[1][t] * state[param.Zidx[1]]),
            linkinv.(param.link[2], param.Z[2][t] * state[param.Zidx[2]])
        )
    )


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
    nIter=3000,              # Number of iterations in the Gibbs sampler
    nBurn=3000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.01,          # Offset for Polya-Gamma variables in the update of h_t
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
nPerGroup = 5

# Where to save everything
save_dir = joinpath(
    pkgdir(TVGLMShrink),
    "plot_res"
)

mkpath(save_dir)


# ============================================================
# 1. LAPLACE
# ============================================================

methodlabel = "Laplace-None"

algoSettings_laplace = (;algoSettings...,scaling = scaling,stateSamplingMethod = :ffbs_laplace)
dataSettings_laplace = (y = y,X = X,covSel = covSel,nPerGroup = nPerGroup)
θpost_laplace, groupSizes_laplace, nFailure_laplace =GibbsTVGLM(dataSettings_laplace,priorSettings,modelSettings,algoSettings_laplace)

prcFailure_laplace =100 * nFailure_laplace[] /(algoSettings_laplace.nBurn + algoSettings_laplace.nIter)
println("$(algoSettings_laplace.stateSamplingMethod) failed at ","$(prcFailure_laplace)% of the simulated trajectories")
quant_paramtime_laplace =quantile_multidim(θpost_laplace,[0.025, 0.5, 0.975],dims = 3)

# ============================================================
# 2. IPLF
# ============================================================

methodlabel = "IPLF"

obsChoice = :both

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

Y, _, _, groupSizes_iplf =splitEqualGroups(y,X,covSel,nPerGroup)
slrObs =prepare_observation_transform(obsTransform,Y,condMean,condCov,nPerGroup)
algoSettings_iplf = (;algoSettings..., nMaxIter=10, scaling = scaling,stateSamplingMethod = :ffbs_slr)
dataSettings_iplf = (y = y,X = X,covSel = covSel,nPerGroup = nPerGroup)
modelSettings_iplf = (;modelSettings...,slrObs = slrObs)
priorSettings_iplf = (;priorSettings..., n₀ = 1)

θpost_iplf, groupSizes_iplf, nFailure_iplf =GibbsTVGLM(dataSettings_iplf, priorSettings_iplf,modelSettings_iplf,algoSettings_iplf)
prcFailure_iplf =100 * nFailure_iplf[] /(algoSettings_iplf.nBurn + algoSettings_iplf.nIter)
println("$(algoSettings_iplf.stateSamplingMethod) failed at ","$(prcFailure_iplf)% of the simulated trajectories")

quant_paramtime_iplf =quantile_multidim(θpost_iplf,[0.025, 0.5, 0.975],dims = 3)
plt_overlay = plot_param_path_betareg(β,γ)
PlotPostParamEvolution!(plt_overlay,quant_paramtime_iplf,"IPLF",groupSizes_iplf;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[4])
display(plt_overlay)

## The initial values 
#plot(y)
# ============================================================
# 3. IEKF
# ============================================================

methodlabel = "IEKF"
obsChoice = :both

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

# Same transformed/grouped observation setup as IPLF
Y, _, _, groupSizes_iekf =splitEqualGroups(y, X, covSel, nPerGroup)
slrObs = prepare_observation_transform(obsTransform,Y,condMean,condCov,nPerGroup)

# IEKF algorithm
algoSettings_iekf = (;algoSettings...,scaling = scaling, nMaxIter=10, stateSamplingMethod = :ffbs_iekf)
dataSettings_iekf = (y = y,X = X,covSel = covSel,nPerGroup = nPerGroup)

# Add IEKF-specific functions ONLY to this modelSettings object
modelSettings_iekf = (;modelSettings...,slrObs = slrObs,sufficient_condMoments_IEKF = BetaSuffStatsCondMoments,sufficient_condJacobian = BetaSuffStatsJacobian)

#Random.seed!(1)
θpost_iekf, groupSizes_iekf, nFailure_iekf =GibbsTVGLM(dataSettings_iekf,priorSettings,modelSettings_iekf,algoSettings_iekf)
prcFailure_iekf =100 * nFailure_iekf[] /(algoSettings_iekf.nBurn + algoSettings_iekf.nIter)
println("$(algoSettings_iekf.stateSamplingMethod) failed at ","$(prcFailure_iekf)% of the simulated trajectories")

quant_paramtime_iekf = quantile_multidim( θpost_iekf,[0.025, 0.5, 0.975],dims=3)
plt_overlay = plot_param_path_betareg(β,γ)
PlotPostParamEvolution!(plt_overlay,quant_paramtime_iekf,"IEKF",groupSizes_iekf;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[2])
#PlotPostParamEvolution!(plt_overlay,quant_paramtime_iplf,"IPLF",groupSizes_iplf;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[4])
#PlotPostParamEvolution!(plt_overlay,quant_paramtime_laplace,"Laplace",groupSizes_laplace;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[1])
display(plt_overlay)

# ============================================================
# 3. OVERLAY PLOT
#
# Start from truth, then add Laplace and IPLF to same plot
# ============================================================

plt_overlay = plot_param_path_betareg(
    β,
    γ
)

# ------------------------------------------------------------
# Laplace
# ------------------------------------------------------------

PlotPostParamEvolution!(
    plt_overlay,
    quant_paramtime_laplace,
    "Laplace",
    groupSizes_laplace;
    dateVec = dateVec,
    interpMethod = interpMethod,
    plot_t0 = keep_t0,
    interval_style = :solid,
    lw = 2,
    c = colors[3]
)


# ------------------------------------------------------------
# IPLF
# ------------------------------------------------------------

PlotPostParamEvolution!(
    plt_overlay,
    quant_paramtime_iplf,
    "IPLF",
    groupSizes_iplf;
    dateVec = dateVec,
    interpMethod = interpMethod,
    plot_t0 = keep_t0,
    interval_style = :solid,
    lw = 2,
    c = colors[4]
)


display(plt_overlay)


# ============================================================
# 4. Save overlay
# ============================================================

savefig(
    plt_overlay,
    joinpath(save_dir, "laplace_iplf_overlay_beta.pdf")
)












################################################################

## IPLF 
methodlabel = "IPLF"

# Beta
#obsTransform = BetaSuffStatsAveraged()
obsTransform = BetaSuffStatsGrouped()
#obsTransform = IdentityTransform()

Y, _, _, groupSizes = splitEqualGroups(y, X, covSel, nPerGroup)

slrObs = prepare_observation_transform(obsTransform,Y,condMean,condCov,nPerGroup)
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_slr);
dataSettings = (y=y , X=X, covSel=covSel, nPerGroup=nPerGroup);
modelSettings = (;modelSettings...,slrObs = slrObs)

Random.seed!(678)
θpost, groupSizes, nFailure = GibbsTVGLM(dataSettings, priorSettings, modelSettings, algoSettings);
prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% of the simulated trajectories")

#size(θpost)
quant_paramtime_iplf = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);
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
