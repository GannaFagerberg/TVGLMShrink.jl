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
# ==========================================================
# Simulate Negative Binomial data with time-varying
# mean and dispersion/size
#
# Y_t | μ_t, r_t ~ NB(μ_t, r_t)
#
# E[Y_t]   = μ_t
# Var[Y_t] = μ_t + μ_t^2 / r_t
#
# Distributions.jl parameterization:
#
# NegativeBinomial(r, p)
#
# with
#
# p = r / (r + μ)
# ==========================================================

T = 500
nCov = 0

# Intercept/state only for mean and size
covSel = [[1], [1]]

p = length(covSel[1])     # mean state dimension
q = length(covSel[2])     # size/dispersion state dimension

nState = p + q


# ----------------------------------------------------------
# Link functions
# ----------------------------------------------------------

link = (LogLinLink(), LogLinLink())
invlink = (x -> linkinv(link[1], x),x -> linkinv(link[2], x))


# ==========================================================
# TRUE UNRESTRICTED STATE PATHS
# ==========================================================

# β[:,1] controls log μ_t
# γ[:,1] controls log r_t

β = zeros(T, p)
γ = zeros(T, q)


# ----------------------------------------------------------
# Mean state
# ----------------------------------------------------------

for t in 1:T
    β[t, 1] = 1.9 * sin(2.5π * t / 150)
end


# ----------------------------------------------------------
# Dispersion/size state
# ----------------------------------------------------------
#
# Larger r -> less overdispersion
# Smaller r -> more overdispersion

γ[1:250, 1]   .= -1.5
γ[251:end, 1] .=  0.0


# ==========================================================
# TRANSFORM TO NEGATIVE BINOMIAL MEAN AND SIZE
# ==========================================================

μtime = similar(β[:, 1])
rtime = similar(γ[:, 1])

for t in 1:T
    μtime[t] = invlink[1](β[t, 1])
    rtime[t] = invlink[2](γ[t, 1])
end


# ==========================================================
# CONVERT TO Distributions.jl PARAMETERIZATION
# ==========================================================
#
# NegativeBinomial(r, prob)
#
# E[Y] = r(1-p)/p = μ
#
# hence
#
# p = r / (r + μ)

ptime = rtime ./ (rtime .+ μtime)


# ==========================================================
# SIMULATE OBSERVATIONS
# ==========================================================

y = Vector{Int}(undef, T)

for t in 1:T

    y[t] = rand(
        NegativeBinomial(
            rtime[t],
            ptime[t]
        )
    )

end

plot(y)
plt = plot_param_path_betareg(β, γ)

# ==========================================================
# VARIABLES KEPT FOR COMPATIBILITY WITH LATER CODE
# ==========================================================

# There are no actual regressors.
# X contains only the intercept column.
X = ones(T, 1)

# Initial true unrestricted states, if needed later
β₀ = [β[1, 1]]
γ₀ = [γ[1, 1]]

# No covariate-generating AR processes
ρ  = Float64[]
σₑ = Float64[]
mₑ = Float64[]
# ==========================================================
# PRIOR FOR STATE AT t = 0
# ==========================================================

m = max(mean(y[1:20]), eps(Float64))
v = max(var(y[1:20]), eps(Float64))

# Negative Binomial:
# Var(Y) = μ + μ² / r
#
# therefore
# r = μ² / (Var(Y) - μ)

# If v <= m, the empirical sample is not overdispersed.
# This corresponds approximately to large r (Poisson limit).
r_init = if v > m
    max(m^2 / (v - m), 1e-3)
else
    100.0
end

priorparam = [m, r_init]


# Initial unrestricted mean state
f_μ(x) = priorparam[1] - invlink[1](x)

β_m0 = [
    find_zero(f_μ, log(m));
    zeros(p - 1)
]


# Initial unrestricted size/dispersion state
f_r(x) = priorparam[2] - invlink[2](x)

β_r0 = [
    find_zero(f_r, log(r_init));
    zeros(q - 1)
]


μ₀ = [β_m0; β_r0]

Σ₀ = :fisherinfo

### Cond moments@views function condMean(param, state, t)

    ημ = param.Z[1][t] * state[param.Zidx[1]]

    ημ = clamp.(ημ, -20.0, 20.0)

    μ = linkinv.(param.link[1], ημ)

    μ = max.(μ, 1e-8)

    return μ
end


@views function condCov(param, state, t)

    ημ = param.Z[1][t] * state[param.Zidx[1]]
    ηr = param.Z[2][t] * state[param.Zidx[2]]

    ημ = clamp.(ημ, -20.0, 20.0)
    ηr = clamp.(ηr, -20.0, 20.0)

    μ = linkinv.(param.link[1], ημ)
    r = linkinv.(param.link[2], ηr)

    μ = max.(μ, 1e-8)
    r = max.(r, 1e-3)

    variance_y = μ .+ μ.^2 ./ r

    return Matrix(Diagonal(vec(variance_y)))
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
        nIter=6000,              # Number of iterations in the Gibbs sampler
        nBurn=2000,               # Number of burn-in iterations
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

    ## IPLF 
    methodlabel = "IPLF"

    # NB
    obsTransform = NBFactorialStatsAveraged()
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
    #### Run from 5 seeds and plot 
