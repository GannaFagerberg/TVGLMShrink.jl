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
# Simulate data from the Von Mises regression model
# with time-varying mean direction and concentration

using Random
using Distributions
using SpecialFunctions

# ==========================================================
# Simulate Von Mises data with
# time-varying mean direction and concentration
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
# Fixed parameter paths for the mean direction
#
# μ_t = X_t' β_t
#
# Notice:
# μ_t is an angle, so we do NOT use exp() here.
# The Von Mises likelihood is periodic in μ.
# ----------------------------------------------------------

function simulate_μ_vonmises(X, T, p, β₀)

    β = zeros(T + 1, p)
    β[1, :] = β₀

    μtime = zeros(T + 1)

    for t in 2:(T + 1)

        # Time-varying intercept
        β[t, 1] = sin(2π * t / T)

        # Time-varying regression coefficient
        if t < T / 3

            β[t, 2] = 0.0

        elseif t < (2 / 3) * T

            β[t, 2] = -1.0

        else

            β[t, 2] = 1.0

        end

        ημ = dot(β[t, :], X[t, :])

        # Identity link for circular location
        μtime[t] = ημ
    end

    return β[2:end, :], μtime[2:end]
end


β, μtime = simulate_μ_vonmises(
    X_regressors,
    T,
    p,
    β₀
)


# ----------------------------------------------------------
# Time-varying concentration
#
# κ_t = exp(γ_t)
#
# κ = 0        -> uniform distribution on the circle
# larger κ     -> increasingly concentrated around μ
# ----------------------------------------------------------

γ = [
    t < T / 2 ? 1.0 : 3.0
    for t in 1:T
]

κtime = exp.(γ)


# ----------------------------------------------------------
# Von Mises distribution
#
# Distributions.jl:
# VonMises(mean direction, concentration)
# ----------------------------------------------------------

VonMisesMean(μ, κ) = VonMises(μ, κ)


# ----------------------------------------------------------
# Simulate observations
# ----------------------------------------------------------

y = [
    rand(VonMisesMean(μtime[t], κtime[t]))
    for t in 1:T
]


# ----------------------------------------------------------
# Principal-angle version of the mean direction
#
# maps μ into (-π, π]
#
# Useful for plotting only.
# ----------------------------------------------------------

μtime_wrapped = atan.(sin.(μtime), cos.(μtime))


# ----------------------------------------------------------
# Circular variance
#
# A(κ) = I₁(κ) / I₀(κ)
#
# Circular variance = 1 - A(κ)
# ----------------------------------------------------------

Atime = besseli.(1, κtime) ./ besseli.(0, κtime)

circvartime = 1 .- Atime


# ----------------------------------------------------------
# Parameter paths
# ----------------------------------------------------------

plt = plot_param_path_betareg(β, γ)


# ----------------------------------------------------------
# Plot observations
# ----------------------------------------------------------

link = (IdentityLink(), LogLinLink())
plot(y)


# Optional: compare observations and true mean direction
plot(
    y,
    label = "y",
    ylabel = "angle (radians)"
)

plot!(
    μtime_wrapped,
    label = "true mean direction",
    linewidth = 2
)



# ==========================================================
# VARIABLES KEPT FOR COMPATIBILITY WITH LATER CODE
# ==========================================================

X      = X_regressors[2:end, :]
covSel = [[1, 2], [1]]

p = length(covSel[1])   # mean-direction states
q = length(covSel[2])   # concentration states


# ==========================================================
# PRIOR MEAN FOR THE STATE AT t = 0
# ==========================================================

# Moment estimates from the first observations
#
# Von Mises:
#
#   E[cos(Y)] = A₁(κ) cos(μ)
#   E[sin(Y)] = A₁(κ) sin(μ)
#
# where
#
#   A₁(κ) = I₁(κ) / I₀(κ).
#
# Hence:
#
#   μ = atan(E[sin(Y)], E[cos(Y)])
#
# and κ is obtained from
#
#   A₁(κ) = R,
#
# where R is the mean resultant length.


y0 = y[1:20]

C = mean(cos.(y0))
S = mean(sin.(y0))

# Mean direction
m = atan(S, C)

# Mean resultant length
Rbar = sqrt(C^2 + S^2)


# ----------------------------------------------------------
# Recover concentration κ from
#
#     I₁(κ) / I₀(κ) = Rbar
#
# Use scaled Bessel functions for numerical stability.
# ----------------------------------------------------------

A1(κ) = besselix(1, κ) / besselix(0, κ)


function concentration_from_resultant(R)

    # numerical protection
    R = clamp(R, 0.0, 1.0 - 1e-10)

    if R < 1e-8
        return 1e-8
    end

    # Find an upper bracket
    κ_upper = 1.0

    while A1(κ_upper) < R
        κ_upper *= 2.0
    end

    return find_zero(
        κ -> A1(κ) - R,
        (0.0, κ_upper),
        Bisection()
    )
end


κhat = concentration_from_resultant(Rbar)


priorparam = [
    m,
    κhat
]


# ----------------------------------------------------------
# Mean-direction state
# ----------------------------------------------------------
#
# With IdentityLink:
#
#     μ = ημ
#
# so this is essentially just m.
# Keeping the generic form makes it compatible
# with your existing code.
# ----------------------------------------------------------

f_μ(x) = priorparam[1] - linkinv(link[1], x)

β_m0 = [
    find_zero(f_μ, priorparam[1]);
    zeros(p - 1)
]


# ----------------------------------------------------------
# Concentration state
# ----------------------------------------------------------
#
# With LogLinLink:
#
#     κ = exp(ηκ)
#
# so the initial state is log(κhat).
# ----------------------------------------------------------

f_κ(x) = priorparam[2] - linkinv(link[2], x)

β_κ0 = [
    find_zero(f_κ, log(priorparam[2]));
    zeros(q - 1)
]


# ----------------------------------------------------------
# Full initial state mean
# ----------------------------------------------------------

μ₀ = [β_m0; β_κ0]


# ==========================================================
# PRIOR COVARIANCE
# ==========================================================

Σ₀ = :fisherinfo


# ==========================================================
# MOMENTS
# ==========================================================
# Σ₀ is computed inside TVGLM_Gibbs()
# using the Inverse Gaussian Fisher information.


# ==========================================================
# MOMENTS FOR VON MISES SUFFICIENT STATISTICS
#
# T(Y) = [cos(Y), sin(Y)]
# ==========================================================

# Stable Bessel ratios
A1(κ) = besselix(1, κ) / besselix(0, κ)
A2(κ) = besselix(2, κ) / besselix(0, κ)


@views function condMean(param, state, t)

    ημ = param.Z[1][t] * state[param.Zidx[1]]
    ηκ = param.Z[2][t] * state[param.Zidx[2]]

    # Identity link for mean direction
    μ = linkinv.(Ref(param.link[1]), ημ)

    # Log link for concentration
    κ = linkinv.(Ref(param.link[2]), ηκ)

    κ = max.(κ, 1e-10)

    a1 = A1.(κ)

    # E[cos(Y)] and E[sin(Y)]
    h_cos = a1 .* cos.(μ)
    h_sin = a1 .* sin.(μ)

    return vcat(h_cos, h_sin)
end


@views function condCov(param, state, t)

    ημ = param.Z[1][t] * state[param.Zidx[1]]
    ηκ = param.Z[2][t] * state[param.Zidx[2]]

    μ = linkinv.(Ref(param.link[1]), ημ)
    κ = linkinv.(Ref(param.link[2]), ηκ)

    κ = max.(κ, 1e-10)

    a1 = A1.(κ)
    a2 = A2.(κ)

    g = length(μ)

    R = zeros(2g, 2g)

    for i in 1:g

        cμ = cos(μ[i])
        sμ = sin(μ[i])

        mcos = a1[i] * cμ
        msin = a1[i] * sμ

        # Second moments
        Ecos2 =
            0.5 * (1 + a2[i] * cos(2 * μ[i]))

        Esin2 =
            0.5 * (1 - a2[i] * cos(2 * μ[i]))

        Ecossin =
            0.5 * a2[i] * sin(2 * μ[i])

        # Covariance of [cos(Y), sin(Y)]
        var_cos =
            Ecos2 - mcos^2

        var_sin =
            Esin2 - msin^2

        cov_cos_sin =
            Ecossin - mcos * msin

        R[i, i] = var_cos
        R[g + i, g + i] = var_sin

        R[i, g + i] = cov_cos_sin
        R[g + i, i] = cov_cos_sin
    end

    return R
end

observation(param, state, t) =
    @views product_distribution(
        VonMises.(
            linkinv.(Ref(param.link[1]),
                     param.Z[1][t] * state[param.Zidx[1]]),

            max.(
                linkinv.(Ref(param.link[2]),
                         param.Z[2][t] * state[param.Zidx[2]]),
                1e-10
            )
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
    nIter=2000,              # Number of iterations in the Gibbs sampler
    nBurn=2000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.01,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:none,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfoBeta,# Fisher info
    FisherInfoPrior=FisherInfoBeta,
    nCalibScale=1000,         # No. iter to calibrate the scaling matrix :fullfixed case
    fixed_scaling = false,     # Should the scaling matrix be fixed across Gibbs iter?
    verbose=true,             # Whether to print verbose output during sampling.
);


dateVec = 1:T
keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
results = []
interpMethod = :linear
scaling      = :none
FisherInfo   = FisherInfoBeta
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
algoSettings_laplace = (;algoSettings...,scaling = scaling,nMaxIter=10, stateSamplingMethod = :ffbs_laplace, FisherInfo = FisherInfo)
dataSettings_laplace = (y = y,X = X,covSel = covSel,nPerGroup = nPerGroup)

θpost_laplace, groupSizes_laplace, nFailure_laplace =GibbsTVGLM(dataSettings_laplace,priorSettings,modelSettings,algoSettings_laplace)

prcFailure_laplace  = 100 * nFailure_laplace[] /(algoSettings_laplace.nBurn + algoSettings_laplace.nIter)
println("$(algoSettings_laplace.stateSamplingMethod) failed at ","$(prcFailure_laplace)% of the simulated trajectories")
quant_paramtime_laplace = quantile_multidim(θpost_laplace,[0.025, 0.5, 0.975],dims = 3)

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


#plot(y)
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









