# Beta regression with fixed path parameter evolution

# If on SLURM cluster, get SLURM_ARRAY_TASK_ID, otherwise use ARGS for local testing
if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    slurm_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
else
    slurm_id = isempty(ARGS) ? 0 : parse(Int, ARGS[1])
end
println("slurm_id = $slurm_id")

using Pkg, Revise
Pkg.activate(@__DIR__) # Activate envir

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
cd(joinpath(@__DIR__, "../.."))
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using PDMats, LogExpFunctions
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim
using Utils: mvcolors as colors
includet("BetaModel.jl")  # Load simulator, Fisher info and plotting for BetaReg

gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=18, markerstrokecolor=:auto)

Random.seed!(slurm_id); # set seed for reproducibility, different seed for each slurm_id


# Simulate data from the Beta regression model with fixed parameter paths
T = 500;
nCov = 2;       # Total number of covariates, excluding the intercept
p = length(covSel[1])
q = length(covSel[2])
covSel = [[1, 2, 3], [1]] # covariates for mean and precision, first covariate is intercept
β₀ = [0.0, 0, -0.05]
γ₀ = [1]        # log precision intercept
logistic(x) = 1 / (1 + exp(-x))
invlinkmean(x) = logistic(x) # Inverse link function for Beta regression
invlinkprec(x) = exp(x) # Inverse link function for Beta regression
ρ = [0.7, 0.7];      # AR(1) coefficients for the covariate processes
σₑ = [1, 10];        # Noise std for the AR(1) processes that generate the covariates
mₑ = [0.0, 0.0];     # Mean for the AR(1) processes that generate the covariates

y, X, β, γ, μtime, ψtime, αtime, βtime = simulate_beta_reg_data(T, nCov, covSel,
    invlinkmean, invlinkprec, ρ, σₑ, mₑ, β₀, γ₀);

## Plot the true parameter paths and the time series

# plot the parameter evolution path of the regression coefficients
plt = plot_param_path_betareg(β, γ)

# plot the evolution of the Beta parameters over time
plot_betaparam_evolution(μtime, ψtime, αtime, βtime)

# plot the evolution of the Beta density over time and the time series
plot_betadensity_evolution(μtime, ψtime, y)


## The prior for the state at time t=0 using priors on intercepts and Fisher info
priorparam_μ = [0.5, 0.5] # Prior for μ ∼ BetaMean(0.5, 0.5)
inflateFactor_ϕ = 1.0 # Inflation factor for prior variance of ϕ
κ₀ = 1.0 # Prior sample size for the state at time t=0, used to scale InvFisher
μ₀, Σ₀ = prior_t0(priorparam_μ, inflateFactor_ϕ, FisherInfo, κ₀, p, q)

# Check that the prior 95% interval to see that they make sense
priorStd = sqrt.(diag(Σ₀))
println("Prior interval for the state at time t=0:")
[μ₀ .- 1.96 * priorStd μ₀ .+ 1.96 * priorStd]


## Set up the prior, model and algorithm settings

dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=1)
priorSettings = (
    ϕ₀=0.5, κ₀=0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀=3.0, ψ₀=1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=zeros(p + q), Σ₀=Diagonal(5 * ones(p + q)), # Prior for βₜ at time t=0
);

modelSettings = (
    observation=observation,
    staticParam=ParamBetaReg,
    condMean=condMean,
    condCov=condCov,
    α=1 / 2,
    β=1 / 2,
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);

algoSettings = (
    stateSamplingMethod=:ffbs_laplace, # Algorithm to sample the state
    nParticles=100,           # Number of particles if using PGAS
    nIter=2000,               # Number of iterations in the Gibbs sampler
    nBurn=2000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.0,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:full,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfo,    # Scaling for the state
);

keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
results = []
## Laplace approximation
algoSettings = (; algoSettings..., stateSamplingMethod=:ffbs_laplace)
for nPerGroup in [3, 5, 10]
    for scaling in [:full, :diagonal, :none]
        println("nPerGroup = $nPerGroup, scaling = $scaling")

        algoSettings = (; algoSettings..., scaling=scaling)
        dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup)

        θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
            priorSettings, modelSettings, algoSettings)

        # Parameter quantiles on the parameter time scale - this always includes t=0
        quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3)

        # Interp parameter quantiles to the observation time scale - potentially drop t=0
        quant_obstime, dateVec = interpParam2Obs(quant_paramtime, groupSizes;
            sample_t0=true, output_t0=keep_t0, interpMethod=:constant)

        scalingMethod = scaling == :full ? "F" : (scaling == :diagonal ? "D" : "N")

        push!(results, (
            name="Laplace$(nPerGroup)$(scalingMethod)",
            quant_paramtime=quant_paramtime,
            quant_obstime=quant_obstime,
            priorSettings=priorSettings,
            modelSettings=modelSettings,
            algoSettings=algoSettings,
            groupSizes=groupSizes,
            dateVec=dateVec,
            nFailure=nFailure[])
        )
    end
end

## IPLF approximation
TODO: # Implement IPLF and add to the `results` array, similar to Laplace approx above.