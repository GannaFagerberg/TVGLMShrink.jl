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
using CSV, DataFrames, Dates, JLD2

slurm_id = get_slurm_id() # get slurm ID, if on cluster


include(joinpath(@__DIR__, "../..") * "/examples/betareg/BetaModel.jl") # BetaReg stuff
include(joinpath(@__DIR__, "../..") * "/examples/betareg/BetaModelUtils.jl") # BetaReg stuff

gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=18, markerstrokecolor=:auto)
mainFolder = @__DIR__
dataFolder = joinpath(@__DIR__, "data")
figFolder = joinpath(@__DIR__, "figs/")
resFolder = joinpath(@__DIR__, "results/")



Random.seed!(slurm_id); # set seed for reproducibility, different seed for each slurm_id

# Simulate data from the Beta regression model with fixed parameter paths
using Random
using Distributions

# ==========================================================
# Simulate Beta data with time-varying mean and precision
# No regression covariates
# ==========================================================

## observation data
#applName = "layoff_vix" # all save files with this prefix
#df = CSV.read(dataFolder * "/layoff_data.csv", DataFrame)

#maxlag = 1 # maximum lag for the covariates, so we remove nLags first rows
#df = df[(maxlag+1):end, :]
println("$(sum(ismissing.(eachrow(df)))) observations with missing values")
y = df.layoff_share # df.layoff_share
T = size(y, 1)
dates = df.date
X = [ones(T) df.credit_spr_lag1 df.vix_lag1 df.cpi_infl df.sentiment_lag1 df.log_oil_lag1]
# standardize covariates
#X[:, 2:end] = (X[:, 2:end] .- mean(X[:, 2:end], dims=1))

X = Matrix{Float64}(X)
dateVec = year.(df.date) .+ (month.(df.date) .- 1) ./ 12
covSel = [[1, 3], [1]]
p = length(covSel[1])
q = length(covSel[2])

# Link function for the mean and precision
link = (LogitLink(), LogLinLink())
#link = (CauchitLink(),LogLinLink()) # does not work 
#link = (LogitLink(),PositiveHardLink(1e-6))

## The prior for the state at time t=0 using priors on intercepts and Fisher info
priorparam = [0.15, exp(6)] # Best guess for y₀ ∼ BetaMean(priorparam_μ[1], priorparam_μ[2])

f(x) = priorparam[1] - linkinv(link[1], x)
g(x) = priorparam[2] - linkinv(link[2], x)

m = find_zero(f, 0.0)
s = find_zero(g, 0.0)
μ₀ = [m; zeros(p - 1); s; zeros(q - 1)]

n₀ = 1.0 # Prior sample size for the state at time t=0, used to scale InvFisher
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
    μ₀=μ₀, 
    Σ₀=Σ₀,
    #Σ₀=I(p+q),
    n₀ = 1, # Prior for βₜ at time t=0
);

modelSettings = (
    observation=observation,
    link=link,

    # New for ILF
    condMean = condMean, 
    condCov=condCov,
    slrObs = nothing,
    #end new

    innovModel=:dsp,   # choices: :dsp, :homogaussuniv
    α=1/2,
    β=1/2,
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);

algoSettings = (
    stateSamplingMethod=:ffbs_laplace, # Algorithm to sample the state
    nParticles=100,           # Number of particles if using PGAS
    nIter=10000,              # Number of iterations in the Gibbs sampler
    nBurn=3000,               # Number of burn-in iterations
    nMaxIter=1,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.00,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:none,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfoBeta,# Fisher info
    nCalibScale=1000,         # No. iter to calibrate the scaling matrix :fullfixed case
    fixed_scaling = false,     # Should the scaling matrix be fixed across Gibbs iter?
    verbose=true,             # Whether to print verbose output during sampling.
);

#dateVec = 1:T
dateVec = year.(df.date) .+ (month.(df.date) .- 1) ./ 12
keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
results = []
interpMethod = :linear
scaling = :none
nPerGroup = 5

# Where to save everything
save_dir = joinpath(pkgdir(TVGLMShrink),"plot_res")
mkpath(save_dir)

# ============================================================
# 1. LAPLACE
# ============================================================
methodlabel = "Laplace-None"
algoSettings_laplace = (;algoSettings...,scaling = scaling,stateSamplingMethod = :ffbs_laplace)
dataSettings_laplace = (y = y,X = X,covSel = covSel,nPerGroup = nPerGroup)

Random.seed!(1)
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
algoSettings_iplf = (;algoSettings...,scaling = scaling,stateSamplingMethod = :ffbs_slr)
dataSettings_iplf = (y = y,X = X,covSel = covSel,nPerGroup = nPerGroup)
modelSettings_iplf = (;modelSettings...,slrObs = slrObs)

Random.seed!(1)
θpost_iplf_tricks, groupSizes_iplf, nFailure_iplf =GibbsTVGLM(dataSettings_iplf,priorSettings,modelSettings_iplf,algoSettings_iplf)
prcFailure_iplf =100 * nFailure_iplf[] /(algoSettings_iplf.nBurn + algoSettings_iplf.nIter)
println("$(algoSettings_iplf.stateSamplingMethod) failed at ","$(prcFailure_iplf)% of the simulated trajectories")


θpost_iplf = copy(θpost_iplf_tricks)
#@load joinpath(save_dir, "θpost_iplf.jld2") θpost_iplf2
#quant_paramtime_iplf3 =quantile_multidim(θpost_iplf,[0.025, 0.5, 0.975],dims = 3)
#quant_paramtime_iplf2 =quantile_multidim(θpost_iplf,[0.025, 0.5, 0.975],dims = 3)

plt_overlay = plot(layout = (3, 1),size = (900, 650),legend = :topright)
PlotPostParamEvolution!(plt_overlay,quant_paramtime_iplf,"IPLF",groupSizes_iplf;
                        dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[4])
display(plt_overlay)

# ============================================================
# 3. OVERLAY PLOT
# ============================================================
plt_overlay = plot(layout = (3, 1),size = (900, 650),legend = :topright)

# ------------------------------------------------------------
# Laplace
# ------------------------------------------------------------
PlotPostParamEvolution!(plt_overlay,quant_paramtime_laplace,"Laplace",groupSizes_laplace;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[1])

# ------------------------------------------------------------
# IPLF
# ------------------------------------------------------------
PlotPostParamEvolution!(plt_overlay,quant_paramtime_iplf2,"IPLF",groupSizes_iplf;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[2])
PlotPostParamEvolution!(plt_overlay,quant_paramtime_iplf3,"SLR",groupSizes_iplf;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = "black")
display(plt_overlay)







# ============================================================
# 4. Save overlay
# ============================================================

#savefig(plt_overlay,joinpath(save_dir, "laplace_iplf_overlay_layoff_1.pdf"))
#θpost_iplf1=copy(θpost_iplf)#centered, alpga=0.5, beta=2, kappa=0

#savefig(plt_overlay,joinpath(save_dir, "laplace_iplf_overlay_layoff_2.pdf"))
#θpost_iplf1=copy(θpost_iplf)#centered, alpga=1, beta=0, kappa=0

#θpost_iplf3 = copy(θpost_iplf2)
#savefig(plt_overlay,joinpath(save_dir, "laplace_iplf_overlay_layoff_slr_maxiter=1.pdf"))

#savefig(plt_overlay,joinpath(save_dir, "laplace_iplf_slr_laplace_layoff.pdf"))

using JLD2
#@save joinpath(save_dir, "θpost_iplf.jld2") θpost_iplf2
#@save joinpath(save_dir, "θpost_laplace.jld2") θpost_laplace



#abspath("θpost_iplf.jld2")

