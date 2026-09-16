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


using Random
using Distributions

# ==========================================================
# Simulate Inverse Gaussian data with
# time-varying mean and shape
# No changes to the basic regression dynamics
# ==========================================================

Random.seed!(12345)

T = 500

β₀ = [2.0, 0.0]
p = size(β₀, 1)

# ----------------------------------------------------------
# Generate covariates
# ----------------------------------------------------------

X_regressors = ones(T + 1)
X_regressors = hcat(
    X_regressors,
    simulateAR(T + 1, [0.7], 1.0)
)

# ----------------------------------------------------------
# Fixed parameter paths for the mean
# μ_t = exp(X_t' β_t)
# ----------------------------------------------------------

function simulate_μ(X, T, p, link, β₀)

    β = zeros(T + 1, p)
    β[1, :] = β₀

    μtime = zeros(T + 1)

    for t in 2:(T + 1)

        β[t, 1] = sin(2π * t / T)

        if t < T / 3

            β[t, 2] = 0.0

        elseif t < (2 / 3) * T

            β[t, 2] = -1.0

        else

            β[t, 2] = 1.0

        end

        ημ = dot(β[t, :], X[t, :])

        μtime[t] = linkinv(link[1], ημ)
    end

    return β[2:end, :], μtime[2:end]
end


# ----------------------------------------------------------
# Time-varying shape / precision parameter
#
# λ_t = exp(γ_t)
# ----------------------------------------------------------

γ = [t < T / 2 ? 1.0 : 3.0 for t in 1:T]

ψtime = exp.(γ)


# ----------------------------------------------------------
# Mean link
#
# Inverse Gaussian mean must be positive,
# so use the log link rather than the Beta logit link
# ----------------------------------------------------------

invlink_log = (LogLinLink(),)

β, μtime = simulate_μ(
    X_regressors,
    T,
    p,
    invlink_log,
    β₀
)


# ----------------------------------------------------------
# Inverse Gaussian distribution
#
# Distributions.jl:
# InverseGaussian(mean, shape)
# ----------------------------------------------------------

InverseGaussianMean(μ, ψ) = InverseGaussian(μ, ψ)


# ----------------------------------------------------------
# Simulate observations
# ----------------------------------------------------------

y = [
    rand(InverseGaussianMean(μtime[t], ψtime[t]))
    for t in 1:T
]


# ----------------------------------------------------------
# Useful conditional variance path
#
# Var(Y_t | state_t) = μ_t^3 / ψ_t
# ----------------------------------------------------------

vartime = μtime.^3 ./ ψtime


# ----------------------------------------------------------
# Parameter path
# ----------------------------------------------------------

plt = plot_param_path_betareg(β, γ)


# ----------------------------------------------------------
# Link functions
# ----------------------------------------------------------

link = (LogLinLink(), LogLinLink())
#invlink = (x -> logistic(x), x -> exp(x))


plot(y)


# ==========================================================
# VARIABLES KEPT FOR COMPATIBILITY WITH LATER CODE
# ==========================================================

X      = X_regressors[2:end, :]
covSel = [[1, 2], [1]]

p = length(covSel[1])   # mean states
q = length(covSel[2])   # shape/precision states


# ==========================================================
# PRIOR MEAN FOR THE STATE AT t = 0
# ==========================================================

# Moment estimates from the first observations
#
# Inverse Gaussian:
#
#   E[Y]   = μ
#   Var[Y] = μ^3 / λ
#
# Hence:
#
#   μ = E[Y]
#   λ = μ^3 / Var[Y]

m = mean(y[1:20])
v = var(y[1:20])

priorparam = [
    m,
    m^3 / v
]


# ----------------------------------------------------------
# Mean state
# ----------------------------------------------------------

f_μ(x) = priorparam[1] - linkinv(link[1], x)

β_m0 = [
    find_zero(f_μ, 0.0);
    zeros(p - 1)
]


# ----------------------------------------------------------
# Shape / precision state
# ----------------------------------------------------------

f_ϕ(x) = priorparam[2] - linkinv(link[2], x)

β_ϕ0 = [
    find_zero(f_ϕ, 0.0);
    zeros(q - 1)
]


# Full initial state mean
μ₀ = [β_m0; β_ϕ0]


# ==========================================================
# PRIOR COVARIANCE
# ==========================================================

Σ₀ = :fisherinfo

# Σ₀ is computed inside TVGLM_Gibbs()
# using the Inverse Gaussian Fisher information.
@views function condMean(param, state, t)

    ημ = param.Z[1][t] * state[param.Zidx[1]]

    μ = linkinv.(param.link[1], ημ)

    # Inverse Gaussian requires μ > 0
    μ = max.(μ, 1e-12)

    return μ
end


@views function condCov(param, state, t)

    ημ = param.Z[1][t] * state[param.Zidx[1]]
    ηκ = param.Z[2][t] * state[param.Zidx[2]]

    μ = linkinv.(param.link[1], ημ)
    κ = linkinv.(param.link[2], ηκ)

    # Both parameters must be positive
    μ = max.(μ, 1e-12)
    κ = max.(κ, 1e-10)

    # Inverse Gaussian:
    # Var(Y | μ, κ) = μ^3 / κ
    variance_y = μ.^3 ./ κ

    return Matrix(Diagonal(vec(variance_y)))
end


@views function observation(param, state, t)

    ημ = param.Z[1][t] * state[param.Zidx[1]]
    ηκ = param.Z[2][t] * state[param.Zidx[2]]

    μ = linkinv.(Ref(param.link[1]), ημ)
    κ = linkinv.(Ref(param.link[2]), ηκ)

    # Numerical safeguards
    μ = max.(μ, 1e-12)
    κ = max.(κ, 1e-10)

    return product_distribution(
        InverseGaussian.(μ, κ)
    )
end

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
    nIter=5000,              # Number of iterations in the Gibbs sampler
    nBurn=2000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.00,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:none,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfoInverseGaussian,# Fisher info
    FisherInfoPrior=FisherInfoInverseGaussian,
    nCalibScale=1000,         # No. iter to calibrate the scaling matrix :fullfixed case
    fixed_scaling = false,     # Should the scaling matrix be fixed across Gibbs iter?
    verbose=true,             # Whether to print verbose output during sampling.
);


dateVec = 1:T
keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
results = []
interpMethod = :linear
scaling      = :none
FisherInfo   = FisherInfoInverseGaussian
nPerGroup    = 5

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
algoSettings_laplace = (;algoSettings...,scaling = scaling,nMaxIter=50, stateSamplingMethod = :ffbs_laplace, FisherInfo = FisherInfo)
dataSettings_laplace = (y = y,X = X,covSel = covSel,nPerGroup = nPerGroup)

θpost_laplace, groupSizes_laplace, nFailure_laplace, nLaplaceFailure=GibbsTVGLM(dataSettings_laplace,priorSettings,modelSettings,algoSettings_laplace)

prcFailure_laplace  = 100 * nFailure_laplace[] /(algoSettings_laplace.nBurn + algoSettings_laplace.nIter)
prcFailure_laplace =100 * nFailure[] /(algoSettings_laplace.nBurn + algoSettings_laplace.nIter)
println("$(algoSettings_laplace.stateSamplingMethod) failed at ","$(prcFailure_laplace)% of the simulated trajectories")
Y, Z, _, _ =splitEqualGroups(y,X,covSel,nPerGroup)
nLaplaceTotal =(algoSettings.nBurn + algoSettings.nIter) * length(Y)
prcLaplaceFailure =100 * nLaplaceFailure[] / nLaplaceTotal
println("Laplace mode optimization failed at ", round(prcLaplaceFailure, digits = 2),"% of the filtering updates.")

quant_paramtime_laplace = quantile_multidim(θpost_laplace,[0.025, 0.5, 0.975],dims = 3)
plt_overlay             = plot_param_path_betareg(β,γ)
PlotPostParamEvolution!(plt_overlay,quant_paramtime_laplace,"Laplace",groupSizes_laplace;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[4])
display(plt_overlay)

# ============================================================
# 2. IPLF
# ============================================================

methodlabel = "IPLF"
obsChoice = :both

obsTransform =
    if obsChoice === :y
        IdentityTransform()
    elseif obsChoice === :both
        InverseGaussianSuffStatsGrouped()
    else
        error("Unknown obsChoice = $obsChoice")
    end


slrObs =prepare_observation_transform(obsTransform,Y,condMean,condCov,nPerGroup)

algoSettings_iplf = (;algoSettings..., nMaxIter=10, scaling = scaling,stateSamplingMethod = :ffbs_slr, FisherInfo = FisherInfo)
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
modelSettings_iekf = (;modelSettings...,slrObs = slrObs,sufficient_condMoments_IEKF = BetaSuffStatsCondMoments,sufficient_condJacobian = BetaSuffStatsJacobian)

#Random.seed!(1)
θpost_iekf, groupSizes_iekf, nFailure_iekf =GibbsTVGLM(dataSettings_iekf,priorSettings,modelSettings_iekf,algoSettings_iekf)
prcFailure_iekf =100 * nFailure_iekf[] /(algoSettings_iekf.nBurn + algoSettings_iekf.nIter)
println("$(algoSettings_iekf.stateSamplingMethod) failed at ","$(prcFailure_iekf)% of the simulated trajectories")

quant_paramtime_iekf = quantile_multidim( θpost_iekf,[0.025, 0.5, 0.975],dims=3)
plt_overlay = plot_param_path_betareg(β,γ)
PlotPostParamEvolution!(plt_overlay,quant_paramtime_iekf,"IEKF",groupSizes_iekf;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[2])
#PlotPostParamEvolution!(plt_overlay,quant_paramtime_iplf,"IPLF",groupSizes_iplf;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[4])
PlotPostParamEvolution!(plt_overlay,quant_paramtime_laplace,"Laplace",groupSizes_laplace;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[1])
display(plt_overlay)

#savefig(plt_overlay,joinpath(save_dir, "laplace_iplf_overlay_beta.pdf"))









