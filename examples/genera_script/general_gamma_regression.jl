# Beta regression with fixed path parameter evolution

#using Pkg
#Pkg.activate(joinpath(@__DIR__, "../.."))
#cd(joinpath(@__DIR__, "../.."))
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
# Simulate Gamma data with time-varying mean and precision
# No regression covariates
#
# Y_t | μ_t, ψ_t ~ Gamma(
#     shape = ψ_t,
#     scale = μ_t / ψ_t
# )
#
# E[Y_t]   = μ_t
# Var[Y_t] = μ_t^2 / ψ_t
# ==========================================================

T = 500
nCov = 0

# ----------------------------------------------------------
# Link functions
# ----------------------------------------------------------

link = (LogLinLink(), LogLinLink())
#invlink = (x -> linkinv(link[1], x),x -> linkinv(link[2], x))
#invlink = (x -> exp(x), x -> exp(x))
#invlink = (x -> linkinv(link[1], x),x -> linkinv(link[2], x))

# ==========================================================
# TRUE UNRESTRICTED STATE PATHS
# ==========================================================
Random.seed!(12345)

# ============================================================
# Simulate Gamma regression data
#
# Mean:
#   ημ,t = x_t' β_t
#   μ_t  = linkinv(link[1], ημ,t)
#
# Precision:
#   γ_t = log(κ_t)
#   κ_t = exp(γ_t)
#
# Observation:
#   Y_t ~ Gamma(shape = κ_t, scale = μ_t / κ_t)
# ============================================================

T = 500

β₀ = [2.0, 0.0]
p = length(β₀)

# ============================================================
# Generate covariates
# ============================================================

X_regressors = ones(T + 1)

X_regressors = hcat(
    X_regressors,
    simulateAR(T + 1, [0.7], 1.0)
)

# ============================================================
# Fixed time-varying regression coefficients for the mean
# ============================================================

function simulate_μ_gamma(
    X,
    T,
    p,
    link,
    β₀
)

    β = zeros(T + 1, p)
    β[1, :] = β₀

    μtime = zeros(T + 1)

    for t in 2:(T + 1)

        # Time-varying intercept
        β[t, 1] =
            sin(2π * t / T)

        # Time-varying regression coefficient
        if t < T / 3

            β[t, 2] = 0.0

        elseif t < (2 / 3) * T

            β[t, 2] = -1.0

        else

            β[t, 2] = 1.0
        end

        # Linear predictor
        ημ =
            dot(
                β[t, :],
                X[t, :]
            )

        # Positive Gamma mean
        μtime[t] =
            linkinv(
                link[1],
                ημ
            )
    end

    return β[2:end, :],
           μtime[2:end]
end


# ============================================================
# Mean link
# ============================================================

link_gamma = (
    LogLinLink(),
)


# ============================================================
# Generate mean path
# ============================================================

β, μtime =
    simulate_μ_gamma(
        X_regressors,
        T,
        p,
        link_gamma,
        β₀
    )


# ============================================================
# Precision path
#
# Same gamma path as in your Beta simulation
# ============================================================

γ = [
    t < T / 2 ? 1.0 : 3.0
    for t in 1:T
]

κtime = exp.(γ)


# ============================================================
# Gamma distribution parameterization
#
# Gamma(shape, scale)
#
# shape = κ
# scale = μ / κ
# ============================================================

GammaMeanPrecision(μ, κ) =
    Gamma(
        κ,
        μ / κ
    )


# ============================================================
# Generate observations
# ============================================================

y = [
    rand(
        GammaMeanPrecision(
            μtime[t],
            κtime[t]
        )
    )
    for t in 1:T
]


# Optional numerical safeguard
y = max.(y, 1e-16)


# plot the parameter evolution path of the regression coefficients
plt = plot_param_path_betareg(β, γ)
plot(y)

# ==========================================================
# VARIABLES KEPT FOR COMPATIBILITY WITH LATER CODE
# ==========================================================

X = X_regressors[2:end,:]
covSel = [[1,2], [1]]
p = length(covSel[1])
q = length(covSel[2])


# ==========================================================
# PRIOR FOR STATE AT t = 0
# ==========================================================# ============================================================
# Initial state prior for Gamma regression
# ============================================================

m = max(mean(y[1:20]), eps(Float64))
v = max(var(y[1:20]), eps(Float64))

# Gamma moment estimates:
# E[Y]   = μ
# Var[Y] = μ² / κ
κ_init = max(m^2 / v, 1e-3)
priorparam = [m, κ_init]

# ------------------------------------------------------------
# Initial unrestricted mean-state coefficients, precision-state coefficients
# ------------------------------------------------------------
f_μ(x) = priorparam[1] - linkinv(link[1], x)
β_m0 = [find_zero(f_μ, log(m)); zeros(p - 1)]

f_ϕ(x) = priorparam[2] - linkinv(link[2], x)
β_ϕ0 = [find_zero(f_ϕ, log(κ_init)); zeros(q - 1)]

# ------------------------------------------------------------
# Initial state
# ------------------------------------------------------------

μ₀ = [β_m0;β_ϕ0]

# Prior sample size used for inverse Fisher scaling
n₀ = 1.0

# Computed internally as, e.g.,
# Σ₀ = (1 / κ₀) * inv((1 / T) * Finfo)
Σ₀ = :fisherinfo

### Cond moments
@views function condMean(param, state, t)

    ημ = param.Z[1][t] * state[param.Zidx[1]]
    ημ = clamp.(ημ, -20.0, 20.0)

    μ = linkinv.(param.link[1], ημ)
    μ = max.(μ, 1e-3)
    return μ
end


@views function condCov(param, state, t)

    ημ = param.Z[1][t] * state[param.Zidx[1]]
    ηκ = param.Z[2][t] * state[param.Zidx[2]]

    ημ = clamp.(ημ, -20.0, 20.0)
    ηκ = clamp.(ηκ, -20.0, 20.0)

    μ = linkinv.(param.link[1], ημ)
    κ = linkinv.(param.link[2], ηκ)

    μ = max.(μ, 1e-3)
    κ = max.(κ, 1e-3)

    variance_y = μ.^2 ./ κ

    return Matrix(Diagonal(vec(variance_y)))
end

@views function observation(param, state, t)

    ημ =
        param.Z[1][t] *
        state[param.Zidx[1]]

    ηκ =
        param.Z[2][t] *
        state[param.Zidx[2]]

    μ =
        GLM.linkinv.(
            Ref(param.link[1]),
            ημ
        )

    κ =
        GLM.linkinv.(
            Ref(param.link[2]),
            ηκ
        )

    return product_distribution(
        Gamma.(
            vec(κ),
            vec(μ ./ κ)
        )
    )
end

dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=1)

priorSettings = (
    ϕ₀=0.5, κ₀=0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀= 3.0, ψ₀= 1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=μ₀, Σ₀=Σ₀,n₀ = 1, # Prior for βₜ at time t=0
);


#link = (ShiftedSoftplusLink(),ShiftedSoftplusLink())
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
    nBurn=2000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.00,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:none,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo= FisherInfoGamma,# Fisher info
    FisherInfoPrior= FisherInfoGamma,
    nCalibScale=1000,         # No. iter to calibrate the scaling matrix :fullfixed case
    fixed_scaling = false,     # Should the scaling matrix be fixed across Gibbs iter?
    verbose=true,             # Whether to print verbose output during sampling.
);

dateVec = 1:T
keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
results = []
interpMethod = :linear
scaling = :none
 FisherInfo= FisherInfoGamma
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
algoSettings_laplace = (;algoSettings...,scaling = scaling,nMaxIter=10, stateSamplingMethod = :ffbs_laplace, FisherInfo = FisherInfo)
dataSettings_laplace = (y = y,X = X,covSel = covSel,nPerGroup = nPerGroup)

θpost_laplace, groupSizes_laplace, nFailure_laplace =GibbsTVGLM(dataSettings_laplace,priorSettings,modelSettings,algoSettings_laplace)

prcFailure_laplace =100 * nFailure_laplace[] /(algoSettings_laplace.nBurn + algoSettings_laplace.nIter)
println("$(algoSettings_laplace.stateSamplingMethod) failed at ","$(prcFailure_laplace)% of the simulated trajectories")
quant_paramtime_laplace =quantile_multidim(θpost_laplace,[0.025, 0.5, 0.975],dims = 3)

quant_paramtime_laplace = quantile_multidim(θpost_laplace,[0.025, 0.5, 0.975],dims = 3)
plt_overlay          = plot_param_path_betareg(β,γ)
PlotPostParamEvolution!(plt_overlay,quant_paramtime_laplace,"Laplace",groupSizes_laplace;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[4])
display(plt_overlay)

# ============================================================
# 2. IPLF
# ============================================================

methodlabel = "IPLF"

obsTransform = GammaSuffStatsGrouped(min_mean = 1e-8,min_precision = 1e-3)
#obsTransform = IdentityTransform()

Y, _, _, groupSizes_iplf =splitEqualGroups(y,X,covSel,nPerGroup)
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
    joinpath(save_dir, "laplace_iplf_overlay_gamma_naive_iplf.pdf")
)




























##############################################
## Laplace approximation - none
methodlabel = "Laplace-None"
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_laplace);
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);

θpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime_lanone = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);
plt = plot_param_path_betareg(β, γ)
PlotPostParamEvolution!(plt, quant_paramtime_lanone, methodlabel,groupSizes; dateVec=dateVec, interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=2, c=colors[3])



##########
## IPLF 
##########

methodlabel = "IPLF"

# Gamma
#obsTransform = GammaSuffStatsAveraged()
obsTransform = GammaSuffStatsGrouped(min_mean = 1e-8,min_precision = 1e-3)
#obsTransform = IdentityTransform()
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
priorSettings = (; priorSettings..., n₀ = 1)

#Random.seed!(234)
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

#For large, E[logY] is only becomes weakly dependent on k