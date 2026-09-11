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
pathof(TVGLMShrink)

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
# ============================================================
# Simulate Negative Binomial data
# with time-varying regression mean and precision/size
#
# Y_t | μ_t, r_t ~ NB(μ_t, r_t)
#
# E[Y_t]   = μ_t
# Var[Y_t] = μ_t + μ_t^2 / r_t
#
# r_t = exp(γ_t)
# ============================================================


Random.seed!(12345)

T = 500

covSel = [[1,2], [1]]
p = length(covSel[1])
q = length(covSel[2])

link = (LogLinLink(),LogLinLink())

# ============================================================
# Mean regression coefficients
# ============================================================

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

X = X_regressors[2:end,:]

# ============================================================
# Fixed time-varying regression coefficients for NB mean
#
# IDENTICAL to reference simulate_neg()
# ============================================================

function simulate_μ_nb(
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

        # Intercept
        β[t, 1] =
            sin(2π * t / T)

        # Regression coefficient:
        # IDENTICAL to reference NB simulation
        if t < T / 3

            β[t, 2] = 1.5

        elseif t < (2 / 3) * T

            β[t, 2] = 1.0

        else

            β[t, 2] = 2.0
        end

        # Linear predictor
        ημ =
            dot(
                β[t, :],
                X[t, :]
            )

        # Mean
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

link_nb = (LogLinLink(),)


# ============================================================
# Generate mean path
# ============================================================

β, μtime =
    simulate_μ_nb(
        X_regressors,
        T,
        p,
        link_nb,
        β₀
    )


# ============================================================
# Precision / size state
#
# IDENTICAL to reference NB simulation
# ============================================================

γ = [t < T / 2 ? -1.5 : 0.0 for t in 1:T]

#γ = [t < T / 2 ? 1.0 : 3.0 for t in 1:T]



# ============================================================
# Size parameter
#
# IDENTICAL to reference:
#
# ψtime = exp.(γ1)
# ============================================================

#rtime =exp.(γ)
rtime = linkinv.(Ref(link[2]), γ)

plot(rtime)
# ============================================================
# Negative Binomial parameterization
#
# E[Y]   = μ
# Var[Y] = μ + μ²/r
# ============================================================

function NBMeanPrecision(μ, r)
    p =r / (r + μ)
    return NegativeBinomial(r,p)
end


# ============================================================
# Generate observations
# ============================================================

y = [rand( NBMeanPrecision(μtime[t], rtime[t]))
    for t in 1:T
]


# ============================================================
# Check precision regimes
# ============================================================

@show minimum(rtime)
@show maximum(rtime)

# exp(-1.5) ≈ 0.22313
# exp(0.0)  = 1.0


# ============================================================
# Plot true state paths
# ============================================================

plt =
    plot_param_path_betareg(
        β,
        γ
    )

plot(y)

# ==========================================================
# VARIABLES KEPT FOR COMPATIBILITY WITH LATER CODE
# ==========================================================

# ==========================================================
# PRIOR FOR STATE AT t = 0
# ==========================================================
m = max(mean(y[1:20]), eps(Float64))
v = max(var(y[1:20]), eps(Float64))

# ============================================================
# Negative Binomial initial values
#
# Parameterization:
#
# E[Y]   = μ
# Var[Y] = μ + μ² / r
#
# Therefore:
#
# r = μ² / (Var[Y] - μ)
# ============================================================

# If v <= m, the first observations are not empirically
# overdispersed relative to Poisson. Use a large finite r.
r_init =
    if v > m
        max(m^2 / (v - m), 1e-3)
    else
        100.0
    end
priorparam = [m, r_init]

# ============================================================
# Initial unrestricted mean state
# ============================================================

f_μ(x) =priorparam[1] -linkinv(link[1], x)

β_m0 = [find_zero(f_μ, log(m));zeros(p - 1)]


# ============================================================
# Initial unrestricted size / precision state
# ============================================================

f_r(x) =priorparam[2] -linkinv(link[2], x)
β_r0 = [find_zero(f_r, log(r_init));zeros(q - 1)]

# ============================================================
# Initial state prior
# ============================================================

μ₀ = [β_m0;β_r0]
n₀ = 1.0
Σ₀ = :fisherinfo

### Cond moments

@views condMean(param, state, t) =
    GLM.linkinv.(
        param.link[1],
        param.Z[1][t] * state[param.Zidx[1]]
    )


@views function condCov(param, state, t)

    @views begin

        μ =
            GLM.linkinv.(
                param.link[1],
                param.Z[1][t] * state[param.Zidx[1]]
            )

        r =
            GLM.linkinv.(
                param.link[2],
                param.Z[2][t] * state[param.Zidx[2]]
            )
    end

    variance_y =
        μ .+
        μ.^2 ./ r

    return diagm(vec(variance_y))
end

@views function observation(param, state, t)

    ημ =
        param.Z[1][t] *
        state[param.Zidx[1]]

    ηr =
        param.Z[2][t] *
        state[param.Zidx[2]]

    μ =
        GLM.linkinv.(
            Ref(param.link[1]),
            ημ
        )

    r =
        GLM.linkinv.(
            Ref(param.link[2]),
            ηr
        )

    return product_distribution(
        NBMeanPrecision.(
            μ,
            r
        )
    )
end


dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=1)

priorSettings = (
    ϕ₀=0.5, κ₀=0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀= 3.0, ψ₀= 1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=zeros(p+q), Σ₀=I(p+q), n₀ = 1, # Prior for βₜ at time t=0
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
        nIter=1000,              # Number of iterations in the Gibbs sampler
        nBurn=1000,               # Number of burn-in iterations
        nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
        nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
        offsetMethod=eps(),       # Offset for log-volatility
        h_upper=Inf,              # Upper bound for log-volatility
        polyaoffset=0.01,          # Offset for Polya-Gamma variables in the update of h_t
        scaling=:none,            # Scaling of state innov, can be :full, :diagonal or :none
        FisherInfo=FisherInfoNB,# Fisher info
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

# ============================================================
# Common settings
# ============================================================

dateVec     = 1:T
keep_t0     = false
interpMethod = :linear

scaling     = :none
nPerGroup   = 5

# Where to save everything
save_dir = joinpath(
    pkgdir(TVGLMShrink),
    "plot_res"
)

mkpath(save_dir)

# ============================================================
# 2. IPLF
# ============================================================

# ============================================================
# 2. IPLF
# ============================================================

methodlabel = "IPLF"

obsTransform = NBFactorialStatsGrouped()

Y, _, _, groupSizes_iplf =
    splitEqualGroups(
        y,
        X,
        covSel,
        nPerGroup
    )

slrObs =
    prepare_observation_transform(
        obsTransform,
        Y,
        condMean,
        condCov,
        nPerGroup
    )

algoSettings_iplf = (
    ;
    algoSettings...,
    scaling = scaling,
    stateSamplingMethod = :ffbs_slr
)

dataSettings_iplf = (
    y = y,
    X = X,
    covSel = covSel,
    nPerGroup = nPerGroup
)

modelSettings_iplf = (
    ;
    modelSettings...,
    slrObs = slrObs
)

θpost_iplf, groupSizes_iplf, nFailure_iplf =
    GibbsTVGLM(
        dataSettings_iplf,
        priorSettings,
        modelSettings_iplf,
        algoSettings_iplf
    )

prcFailure_iplf =
    100 * nFailure_iplf[] /
    (algoSettings_iplf.nBurn + algoSettings_iplf.nIter)

println(
    "$(algoSettings_iplf.stateSamplingMethod) failed at ",
    "$(prcFailure_iplf)% of the simulated trajectories"
)

quant_paramtime_iplf =
    quantile_multidim(
        θpost_iplf,
        [0.025, 0.5, 0.975],
        dims = 3
    )

plt_overlay          = plot_param_path_betareg(β,γ)
PlotPostParamEvolution!(plt_overlay,quant_paramtime_iplf,"IPLF",groupSizes_iplf;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[1])


# ============================================================
# 1. LAPLACE
# ============================================================

methodlabel = "Laplace-None"

algoSettings_laplace = (
    ;
    algoSettings...,
    scaling = scaling,
    stateSamplingMethod = :ffbs_laplace
)

dataSettings_laplace = (
    y = y,
    X = X,
    covSel = covSel,
    nPerGroup = nPerGroup
)

θpost_laplace, groupSizes_laplace, nFailure_laplace =
    GibbsTVGLM(
        dataSettings_laplace,
        priorSettings,
        modelSettings,
        algoSettings_laplace
    )

prcFailure_laplace =
    100 * nFailure_laplace[] /
    (algoSettings_laplace.nBurn + algoSettings_laplace.nIter)

println(
    "$(algoSettings_laplace.stateSamplingMethod) failed at ",
    "$(prcFailure_laplace)% of the simulated trajectories"
)

quant_paramtime_laplace =
    quantile_multidim(
        θpost_laplace,
        [0.025, 0.5, 0.975],
        dims = 3
    )



# ============================================================
# 3. OVERLAY PLOT
#
# Start from truth, then add Laplace and IPLF to same plot
# ============================================================

plt_overlay = plot_param_path_betareg(β,γ)


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
    joinpath(save_dir, "laplace_iplf_overlay_NB2.pdf")
)



# ============================================================
# 5. Save everything together as well
# ============================================================

@save joinpath(save_dir, "laplace_iplf_all_results.jld2") \
    θpost_laplace \
    θpost_iplf \
    quant_paramtime_laplace \
    quant_paramtime_iplf \
    groupSizes_laplace \
    groupSizes_iplf \
    prcFailure_laplace \
    prcFailure_iplf




    ##########################################
    ##########################################
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



    ## IPLF 
    methodlabel = "IPLF"

    # NB
    #obsTransform = NBFactorialStatsAveraged()
    #obsTransform = IdentityTransform()
    obsTransform = NBFactorialStatsGrouped()
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

    #Random.seed!(678)
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
    #### Run from 5 seeds and plot 


    # ============================================================
# Save Laplace results
# ============================================================

@save joinpath(save_dir, "laplace_results.jld2") \
    θpost_laplace \
    quant_paramtime_laplace \
    groupSizes_laplace \
    prcFailure_laplace
# ============================================================
# Save IPLF results
# ============================================================

@save joinpath(save_dir, "iplf_results.jld2") \
    θpost_iplf \
    quant_paramtime_iplf \
    groupSizes_iplf \
    prcFailure_iplf