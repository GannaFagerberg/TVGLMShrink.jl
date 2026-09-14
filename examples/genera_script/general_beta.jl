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
# Simulate Beta data with time-varying mean and precision
# No regression covariates
# ==========================================================

T = 500
nCov = 0

# Intercept/state only for mean and precision
covSel = [[1], [1]]

p = length(covSel[1])     # = 1, mean state
q = length(covSel[2])     # = 1, precision state

nState = p + q            # = 2

# ----------------------------------------------------------
# Link functions
# ----------------------------------------------------------

link = (LogitLink(), LogLink())
#link = (LogitLink(), ()) # best so far
#invlink2 = (x -> logistic(x), x -> exp(x))

# ==========================================================
# TRUE UNRESTRICTED STATE PATHS
# ==========================================================

# β[:,1] = unrestricted state controlling the Beta mean
# γ[:,1] = unrestricted state controlling the precision

β = zeros(T, p)
γ = zeros(T, q)

# Example time-varying paths
for t in 1:T

    # unrestricted mean state
    β[t, 1] = 0.8 * sin(2.5π * t / 150)
    γ[t, 1] = 0.8 * sin(2.5π * t / 150)

    # unrestricted log-precision state
    #γ[t, 1] =1.5 + 0.5 * cos(2π * t / 200)
    # Precision state: step change
end

    #β[1:250, 1]  .=  -3.1
    #β[251:end, 1] .=  -4.1

    #γ[1:250, 1]   .= log(10.0)
    #γ[251:end, 1] .= log(5.0)

    #γ[1:250, 1]   .= log(0.5)
    #γ[251:end, 1] .= log(1.5)  
    
    #γ[1:250, 1]   .= 3.5
    #γ[251:end, 1] .=  3.5

    #γ[1:250, 1]   .= 1.5
    #γ[251:end, 1] .= 3.5   

# ==========================================================
# TRANSFORM TO BETA MEAN AND PRECISION
# ==========================================================

μtime = similar(β[:, 1])
ψtime = similar(γ[:, 1])

for t in 1:T
    μtime[t]  = linkinv(link[1], β[t, 1])
    ψtime[t] = linkinv(link[2], γ[t, 1])
end

# Beta shape parameters
αtime = μtime .* ψtime
βtime = (1 .- μtime) .* ψtime

plot(αtime )
plot!(βtime )
# ==========================================================
# SIMULATE OBSERVATIONS
# ==========================================================

Random.seed!(564)
y = Vector{Float64}(undef, T)
for t in 1:T
    y[t] = rand(Beta(αtime[t], βtime[t]))
end

# Numerical protection for log(y), log(1-y), etc.
y = clamp.(y, 1e-16, 1 - 1e-16)
plot(y)

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

# plot the parameter evolution path of the regression coefficients
plt = plot_param_path_betareg(β, γ)

# plot the evolution of the Beta distribution parameters over time
#plot_betaparam_evolution(μtime, ψtime, αtime, βtime)

# plot the evolution of the Beta density over time and the time series
plot_betadensity_evolution(μtime, ψtime, y)

## The prior for the state at time t=0 using priors on intercepts and Fisher info
m = mean(y[1:20])
v = max(var(y[1:20]), eps(Float64))

κ_init = max(m * (1 - m) / v - 1,1e-3)

priorparam = [m, κ_init]

f_μ(x) = priorparam[1] - linkinv(link[1], x)
β_m0 = [find_zero(f_μ, 0.0); zeros(p - 1)]

f_ϕ(x) =  priorparam[2] - linkinv(link[2], x)
β_ϕ0 = [find_zero(f_ϕ, log(κ_init)); zeros(q - 1)]

μ₀ = [β_m0; β_ϕ0]

Σ₀ = :fisherinfo

## Set up the prior, model and algorithm settings


@views function condMean(param, state, t)

    ημ = param.Z[1][t] * state[param.Zidx[1]]
    μ = linkinv.(param.link[1], ημ)

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


dataSettings = (y=y, X=X, covSel=covSel, nPerGroup=1)

priorSettings = (
    ϕ₀=0.5, κ₀=0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀= 3.0, ψ₀= 1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=μ₀, Σ₀=Σ₀,n₀ = 1, # Prior for βₜ at time t=0
);

#link = (LogitLink(), LogLinLink())
modelSettings = (
    observation=observation,
    link=link,
 
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
    FisherInfo=FisherInfoBeta,# Fisher info
    nCalibScale=1000,         # No. iter to calibrate the scaling matrix :fullfixed case
    fixed_scaling = true,     # Should the scaling matrix be fixed across Gibbs iter?
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

# Beta
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

#obsTransform = IdentityTransform()
Y, _, _, groupSizes = splitEqualGroups(y, X, covSel, nPerGroup)

slrObs = prepare_observation_transform(obsTransform,Y,condMean,condCov,nPerGroup)
algoSettings = (; algoSettings..., scaling=scaling, stateSamplingMethod=:ffbs_slr);
dataSettings = (y= y, X=X, covSel=covSel, nPerGroup=nPerGroup);
modelSettings = (;modelSettings...,slrObs = slrObs)

Random.seed!(678)
θpost, groupSizes, nFailure = GibbsTVGLM(dataSettings, priorSettings, modelSettings, algoSettings);
prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% of the simulated trajectories")

#size(θpost)
quant_paramtime_iplf = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);
plt_iplf = plot_param_path_betareg(β, γ)
PlotPostParamEvolution!(plt_iplf,quant_paramtime_iplf,"IPLF",groupSizes;dateVec=dateVec,interpMethod=interpMethod,
                        plot_t0=keep_t0,interval_style=:solid,lw=2,c=colors[4])
display(plt_iplf)

### in some cases letting delta gamma be 0.3 helps
### for iplf shoulders 0.01 help prevent some excursions into unstable regions

#### Run from 5 seeds and plot 

####################
# IEKF
####################


algoSettings = (
    stateSamplingMethod=:ffbs_laplace, # Algorithm to sample the state
    nParticles=100,           # Number of particles if using PGAS
    nIter=3000,              # Number of iterations in the Gibbs sampler
    nBurn=3000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.0,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:none,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfoBeta,# Fisher info
    nCalibScale=1000,         # No. iter to calibrate the scaling matrix :fullfixed case
    fixed_scaling = true,     # Should the scaling matrix be fixed across Gibbs iter?
    verbose=true,             # Whether to print verbose output during sampling.
);

scaling = :none
nPerGroup = 5

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

if obsChoice === :logy
    ieMoments  = BetaLogYCondMoments
    ieJacobian = BetaLogYJacobian
elseif obsChoice === :log1my
    ieMoments  = BetaLog1mYCondMoments
    ieJacobian = BetaLog1mYJacobian
elseif obsChoice === :both
    ieMoments  = BetaSuffStatsCondMoments
    ieJacobian = BetaSuffStatsJacobian
end

# Same transformed/grouped observation setup as IPLF
Y, _, _, groupSizes_iekf =splitEqualGroups(y, X, covSel, nPerGroup)
slrObs = prepare_observation_transform(obsTransform,Y,condMean,condCov,nPerGroup)

# IEKF algorithm
algoSettings_iekf = (;algoSettings...,scaling = scaling, nMaxIter=10, stateSamplingMethod = :ffbs_iekf)
dataSettings_iekf = (y = y,X = X,covSel = covSel,nPerGroup = nPerGroup)

# Add IEKF-specific functions ONLY to this modelSettings object
modelSettings_iekf = (;modelSettings...,slrObs = slrObs,
                      sufficient_condMoments_IEKF = ieMoments,
                      sufficient_condJacobian = ieJacobian)

Random.seed!(1)
θpost_iekf, groupSizes_iekf, nFailure_iekf =GibbsTVGLM(dataSettings_iekf,priorSettings,modelSettings_iekf,algoSettings_iekf)
prcFailure_iekf =100 * nFailure_iekf[] /(algoSettings_iekf.nBurn + algoSettings_iekf.nIter)
println("$(algoSettings_iekf.stateSamplingMethod) failed at ","$(prcFailure_iekf)% of the simulated trajectories")

quant_paramtime_iekf = quantile_multidim(θpost_iekf,[0.025, 0.5, 0.975],dims=3)
plt_overlay = plot_param_path_betareg(β, γ)
PlotPostParamEvolution!(plt_overlay,quant_paramtime_iekf,"IEKF (new constraints)",groupSizes_iekf;dateVec = dateVec,interpMethod = interpMethod,plot_t0 = keep_t0,interval_style = :solid,lw = 2,c = colors[2])
display(plt_overlay)#
#savefig(plt_overlay,joinpath(save_dir, "laplace_iplf_iekf_new_constraints.pdf"))
