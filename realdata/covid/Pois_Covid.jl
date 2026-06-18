# Poisson Regression for Covid data

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
using Roots
includet("../../examples/poisreg/PoisModel.jl")  # Load Fisher info and plotting for PoisReg

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
using CSV, DataFrames
using Dates

CovidData = CSV.read("/Users/niuyijie/Desktop/CovidDataU.csv", DataFrame)

T = size(CovidData, 1);
nCov = 1;       # Total number of covariates, excluding the intercept
covSel = [[1, 2, 3]] # covariates for mean and precision, first covariate is intercept
p = length(covSel[1])
invlink_dgp(x) = exp_lin(x) # Inverse link function for Poisson regression
y = CovidData.case
X = hcat(ones(T), CovidData.wastewater, CovidData.temp) # Add intercept to covariates

lag = 7
y = y[lag+1:end]
X = X[1:end-lag, :]


plot(y)
plot(X[:, 2])
plot(X[:, 3])
## The prior for the state at time t=0 using priors on intercepts and Fisher info
priorparam_λ = mean(y[1:30]) # Prior for λ ∼ lognormal(m, σ²)
κ₀ = 1.0 # Prior sample size for the state at time t=0, used to scale InvFisher
μ₀, Σ₀ = prior_t0(priorparam_λ, FisherInfo, κ₀, p)

# Check that the prior 95% interval to see that they make sense
priorStd = sqrt.(diag(Σ₀))
println("Prior interval for the state at time t=0:")
[μ₀ .- 1.96 * priorStd μ₀ .+ 1.96 * priorStd]

plt_full = [plot() for i in 1:p]
plt_none = [plot() for i in 1:p]
plt_IPLF = [plot() for i in 1:p]
plt_laplace = [plot() for i in 1:p]
## Set up the prior, model and algorithm settings

dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=1)
priorSettings = (
    ϕ₀=0.5, κ₀=0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀=3.0, ψ₀=1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=μ₀, Σ₀=Σ₀, # Prior for βₜ at time t=0
);

modelSettings = (
    observation=observation,
    staticParam=ParamPoisReg,
    condMean=condMean,
    condCov=condCov,
    innovModel=:dsp,
    α=1 / 2,
    β=1 / 2,
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);



keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
results = []
## Laplace approximation - full Fisher scaling
#algoSettings = (; algoSettings..., stateSamplingMethod=:ffbs_laplace);
nPerGroup = 14
dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup);

function FisherInfo(θ, μ, t, Xm)
    S = Xm' * (Diagonal(invlink_dgp.(Xm * μ))) * Xm # Add a small ridge for numerical stability
    #eig = eigen(S).values
    # S = S +  0.01 * mean(eig) * I  # Add a small ridge for numerical stability
    return S
end
FisherInfo(θ, μ, t) = FisherInfo(θ, μ, t, X[:, covSel[1]])


scalingVec = [:full]
methods = [:ffbs_laplace]#, :ffbs_slr]
legendLabels = ["Laplace"]#, "IPLF"]
plt = Dict(
    #:full => plt_full,
    #:none => plt_none
    :ffbs_slr => plt_IPLF,
    :ffbs_laplace => plt_laplace
)
function main()
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
    )
    for i in 1:length(scalingVec)
        scaling = scalingVec[i]
        algoSettings = (; algoSettings..., scaling=scaling)

        for j in 1:length(methods)
            method = methods[j]
            methodlabel = legendLabels[j]
            algoSettings = (; algoSettings..., stateSamplingMethod=method)
            θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
                priorSettings, modelSettings, algoSettings)
            println("$methodlabel failed at $(100*nFailure[]/(algoSettings.nBurn+algoSettings.nIter))% of the simulated trajectories")

            # Parameter quantiles on the parameter time scale - this always includes t=0
            quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3)

            #PlotPostParamEvolution!(plt[scaling], quant_paramtime,
            #    "$methodlabel ($(nPerGroup))$(scalingLabel(scaling))",
            #    groupSizes; dateVec=nothing, interpMethod=:linear, plot_t0=keep_t0, interval_style=:dash, lw=3, c=colors[2+j])

            PlotPostParamEvolution!(plt[method], quant_paramtime,
                "$methodlabel ($(nPerGroup))$(scalingLabel(scaling))",
                groupSizes; dateVec=nothing, interpMethod=:constant, plot_t0=keep_t0, interval_style=:dash, lw=3, c=colors[2+i])

            # Interpolate parameter quantiles to the observation time scale - potentially drop t=0 here
            quant_obstime, dateVec = interpParam2Obs(quant_paramtime, groupSizes;
                sample_t0=true, output_t0=keep_t0, interpMethod=:linear)

            push!(results, (
                name="$methodlabel scaling $(scaling)",
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
    ## Store data and parameter paths last in the results vector. Save plots.
    push!(results, (
        name="TruePathAndData",
        paramPaths=β[(!keep_t0+1):end, :],
        y=y,
        X=X)
    )
end
main()

#PlotPostParamEvolution!(plt, quant_paramtime, "PGAS-F", groupSizes;
#  dateVec=nothing, interpMethod=:constant, plot_t0=keep_t0, interval_style=:dash, lw=3, c=colors[3])
plot(plt_laplace..., layout=(2, 2), size=(1400, 1000), xlabel="time",
    xguidefontsize=24, titlefontsize=12, bottommargin=5mm, legend=:outerright)
plot(plt_IPLF..., layout=(2, 1), size=(1400, 1000), xlabel="time",
    xguidefontsize=24, titlefontsize=12, bottommargin=5mm, legend=false)

p = plot(
    plt_laplace..., plt_IPLF...,
    layout=(2, 2),
    size=(2000, 1200),
    xtickfontsize=14,
    ytickfontsize=14,
    xguidefontsize=16,
    yguidefontsize=16,
    titlefontsize=14,
    legend=:outerright)
@save joinpath(resFolder, "CovidGroup$(nPerGroup)_$(slurm_id).jld2") results
savefig(p, figFolder * "CovidGroup$(nPerGroup)_$(slurm_id).pdf")
