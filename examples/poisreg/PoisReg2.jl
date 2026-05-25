# Poisson regression with Dynamic Shrinkage Process parameter evolution

# If on SLURM cluster, get SLURM_ARRAY_TASK_ID, otherwise use ARGS for local testing
if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    slurm_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
else
    slurm_id = isempty(ARGS) ? 0 : parse(Int, ARGS[1])
end
println("slurm_id = $slurm_id")

using Pkg
Pkg.activate(@__DIR__) # Activate envir

# First we load the required packages and set some plotting parameters:
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using JLD2, PDMats
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim
using Utils: mvcolors as colors
includet("PoisRegModel.jl")  # Load simulator, Fisher info and plotting 

## Settings

# Simulation of DGP settings
T = 500;
σₑ = [1, 10]    # Noise std for the AR(1) processes that generate the covariates
mₑ = [0, 0]     # Mean for the AR(1) processes that generate the covariates 
ρ = [0.7, 0.7]  # AR(1) coefficients for the covariate processes
β₀ = [0.0, 0.0, 0.05] # Initial parameter values for regr coeffs, including intercept
invlink_dgp(x) = exp_lin(x) # Inverse link function for the DGP, can be exp or exp_lin

# Model settings    
invlink(x) = exp_lin(x) # inverse link function for fitted Poisson regression

# Grouping settings
nPerGroup = 5            # Number of observations per group
interpMethod = :constant # interp from parameter to obs time scale. :constant or :linear
keep_t0 = false          # keep t=0 in the plots and storage

# Scaling settings for the parameter evolution
scalingType = :diagonal # can be :full, :diagonal or :none

gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=18, markerstrokecolor=:auto)

# Set up folders for saving results and figures - create if they don't exist
mainFolder = @__DIR__
figFolder = joinpath(mainFolder, "figures/")
if !isdir(figFolder)
    mkdir(figFolder)
end
resFolder = joinpath(mainFolder, "results/")
if !isdir(resFolder)
    mkdir(resFolder)
end

Random.seed!(slurm_id); # set seed for reproducibility, different seed for each slurm_id

results = []       # Store results for all methods/models/options in this array

# ### Simulate data from the Poisson regression model with fixed parameter paths
p = 3;      # Number of parameters, including intercept
y, X, β = simulate_poisson_reg_data(T, p, invlink_dgp, ρ, σₑ, mₑ, β₀)


# ### Plot the parameter evolution path of βₜ and the time series
keep_t0 = false # plot β₀ or not
plt = []
for j = 1:p
    push!(plt, plot(!keep_t0:T, β[(!keep_t0+1):end, j], label="true",
        xlabel="time, " * L"t", ylabel="", title=L"\beta_{%$(j-1)}", color=:black, lw=2))
end
plot(plt..., layout=(p, 1), size=(1200, 1000), xguidefontsize=12,
    yguidefontsize=14, titlefontsize=20,
    legend=:bottomleft, margin=5mm)

# ### Set up the prior, model and algorithm settings
priorSettings = (
    ϕ₀=0.5, κ₀=0.3,         # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,       # Prior for μ ~ N(m₀, σ₀²)
    ν₀=3.0, ψ₀=1.0,         # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=zeros(p), Σ₀=5 * I(p),# Prior for βₜ at time t=0
);


## Setting up data as grouped data
Y, Z, groupSizes = splitEqualGroups(y, X, nPerGroup)

# Instantiate model parameters (Σᵥ = I for all t), overwritten at each Gibbs iteration
param = ParamTvReg(LogVol2Covs(zeros(length(groupSizes), p)), Z)

## Scaling 
scaling = sqrt(Σ₀)
scaling = I(p)
function PoisFisherInfo(θ, μ, t, scalingType, Xm)
    if scalingType == :none
        return I(length(μ))
    end
    T = size(Xm, 1)
    if t > 1
        S = sqrt(inv((Xm' * Diagonal(exp.(Xm * μ)) * Xm) / T))
    else
        S = scaling
    end
    if scalingType == :diagonal
        return Diagonal(diag(S))
    end
    return S
end
PoisFisherInfo(θ, μ, t) = PoisFisherInfo(θ, μ, t, scalingType, vcat(θ.Z...))

## Model and Algorithms settings
modelSettings = (
    observation=observation,
    param=param,
    condMean=condMean,
    condCov=condCov,
    α=1 / 2,          # First shape param in Z distribution
    β=1 / 2,          # Second shape param in Z distribution
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);

algoSettings = (
    stateSamplingMethod=:pgas, #:ffbs_laplace, # Algorithm to sample the state
    nParticles=200,           # Number of particles if using PGAS
    nIter=2000,              # Number of iterations in the Gibbs sampler
    nBurn=1000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.0,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:full,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=PoisFisherInfo,# Scaling for the state
);


## Laplace approximation 
algoSettings = (; algoSettings..., stateSamplingMethod=:pgas)

θpost, Hpost, ϕpost, σ²ₙpost, μpost, nFailure = GibbsTVGLM(Y, priorSettings, modelSettings,
    algoSettings);

println("Laplace failed at $(100*nFailure[]/(algoSettings.nBurn+algoSettings.nIter))% of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);

# Interpolate parameter quantiles to the observation time scale - potentially drop t=0 here
quant_obstime, dateVec = interpParam2Obs(quant_paramtime, groupSizes;
    sample_t0=true, output_t0=keep_t0, interpMethod=:constant)

push!(results, (
    name="Laplace scaling",
    quant_paramtime=quant_paramtime,
    quant_obstime=quant_obstime,
    priorSettings=priorSettings,
    modelSettings=modelSettings,
    algoSettings=algoSettings,
    groupSizes=groupSizes,
    dateVec=dateVec,
    nFailure=nFailure[])
)

PlotPostParamEvolution!(plt, quant_paramtime, "PGAS-F", groupSizes;
    dateVec=nothing, interpMethod=:constant, plot_t0=keep_t0, interval_style=:dash, lw=3, c=colors[3])
plot(plt..., layout=(3, 1), size=(1400, 1000), xlabel="time",
    xguidefontsize=14, titlefontsize=20, bottommargin=5mm, legend=:bottomleft)



## Store data and parameter paths last in the results vector. Save plots.
push!(results, (
    name="TruePathAndData",
    paramPaths=β[(!keep_t0+1):end, :],
    y=y,
    X=X)
)


@save joinpath(resFolder, "PoisSimGroup$(nPerGroup)_$(slurm_id).jld2") results
savefig(figFolder * "PoisSimGroup$(nPerGroup)_$(slurm_id).pdf")
