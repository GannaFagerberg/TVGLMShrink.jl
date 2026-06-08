using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
cd(joinpath(@__DIR__, "../.."))
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using PDMats#, LogExpFunctions
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim
using Utils: mvcolors as colors
using CSV, DataFrames, Dates

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize = 18, markerstrokecolor = :auto)

Random.seed!(12345);

#figFolder = joinpath(@__DIR__,"figs/")

BetaMean(μ, ψ) = Beta(1e-5+μ*ψ,1e-5+(1-μ)*ψ)

cpipc1 = CSV.read("/Users/niuyijie/Downloads/CPIAUCSL_PC1.csv", DataFrame)
cpipc1[findall(ismissing, cpipc1[:,2]),2] .= cpipc1[findall(ismissing, cpipc1[:,2]) .- 1,2]

## observation data
bdf2 = CSV.read("/Users/niuyijie/Downloads/LNS13023654.csv", DataFrame)

bdf2[findall(ismissing, bdf2[:,2]),2] .= bdf2[findall(ismissing, bdf2[:,2]) .- 1,2]


x21 = cpipc1[findfirst(cpipc1[:,1] .== Date(1966,4,1)):end-11, 2] ./ 100

y = bdf2[:,2]./100

T = size(y, 1)

q = 1; # Number of covariates in ψ, including intercept
Xmean = [ones(T) x21]; # Design matrix
p = size(Xmean,2)
Xprec = [ones(T) randn(T, q-1)];                 # Design matrix for precision
logistic(x) = 1/(1+exp(-x))
invlinkmean_dgp(x) = logistic(x) # Inverse link function for Beta regression
invlinkprec_dgp(x) = exp(x) # Inverse link function for Beta regression


#### scaling
logit(x) = log(x ./ (1 .- x))
E_μ = mean(y)
ϕ_μ = 0.5 
μ_sim = rand(BetaMean(E_μ, ϕ_μ), 10000)
β_m0 = [mean(logit.(μ_sim)), zeros(p-1)...]


#### ϕ
E_ϕ = E_μ*(1-E_μ)/var(y) - 1
var_ϕ = 2 * E_ϕ

s² = log(var_ϕ/(E_ϕ^2) + 1)
m = log(E_ϕ) - s²/2

#s₀² = log(s²/(m^2) + 1)
#m₀ = log(m) - s₀²/2
β_ϕ0 = [log(m) - s₀²/2, zeros(q-1)...]


Scaling_matrix = fisher_beta_blocks(Xmean, Xprec, β_m0, β_ϕ0)
Σₘ = Scaling_matrix ^(2)
Scaling_matrix = [var(logit.(μ_sim)) 0 0; 0 Σₘ[2,2] 0; 0 0 s²]
Scaling_matrix = 0.5 * (Scaling_matrix + Scaling_matrix')
scaling = (Scaling_matrix)^(1/2)


# ### Set up the Beta regression model
mutable struct ParamTvReg{T, S<:AbstractMatrix{T}}
    Σᵥ::Vector{PDMat{T,S}}
    Zmean::Vector{Matrix{T}}
    Zprec::Vector{Matrix{T}}
end

invlinkmean(x) = logistic(x) # inverse link function for μ in Beta regression
invlinkprecision(x) = exp(x) # inverse link function for ψ in Beta regression
observation(param, state, t) = 
    product_distribution(
        BetaMean.(
            invlinkmean.(param.Zmean[t] * state[1:p]), 
            invlinkprecision.(param.Zprec[t] * state[(p+1):(p+q)])
        )
    )
condMean(param, state, t) = invlinkmean.(param.Zmean[t] * state[1:p])
function condCov(param, state, t) 
    μ = invlinkmean.(param.Zmean[t] * state[1:p])
    ψ = invlinkprecision.(param.Zprec[t] * state[(p+1):(p+q)])
    return diagm(μ .* (1 .- μ) ./ (1 .+ ψ))
end

# #### Setting up data as grouped data
nPerGroup = 5
X = [Xmean, Xprec]
Y, Z, groupSizes = splitEqualGroups(y, X, nPerGroup)
Zmean = Z[1]
Zprec = Z[2]

# Instantiate model parameters (Σᵥ = I for all t), overwritten at each Gibbs iteration
param = ParamTvReg(LogVol2Covs(zeros(length(groupSizes),p+q)), Zmean, Zprec) 

function FisherInfo(θ, μ, t)
    Xm = vcat(θ.Zmean...)
    Xp = vcat(θ.Zprec...)
    if t > 1
        S = fisher_beta_blocks(Xm, Xp, μ[1:2], μ[3:end])
    else 
        S = scaling
    end
   # S = Diagonal(diag(S))
    return S
end
# ### Set up the prior, model and algorithm settings
priorSettings = (
    ϕ₀ = 0.5, κ₀ = 0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀ = -15.0, σ₀ = 3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀ = 3.0, ψ₀ = 1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀ = zeros(p+q), Σ₀ = 5*I(p+q), # Prior for βₜ at time t=0
); 

modelSettings = (
    observation = observation,
    param = param,
    condMean  = condMean,
    condCov   = condCov,
    α = 1/2,          
    β = 1/2,          
    updateσₙ = false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp = 10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);

algoSettings = (
    stateSamplingMethod = :pgas,# Algorithm to sample the state
    nParticles = 200,           # Number of particles if using PGAS
    nIter = 5000,               # Number of iterations in the Gibbs sampler
    nBurn = 3000,               # Number of burn-in iterations
    nMaxIter = 10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS = 500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod = eps(),       # Offset for log-volatility
    h_upper = Inf,              # Upper bound for log-volatility
    polyaoffset = 0.0,           # Offset for Polya-Gamma variables in the update of h_t
    FisherInfo = FisherInfo,      # Scaling for the state
    );

# ### PGAS 
θpost, Hpost, ϕpost, σ²ₙpost, μpost = GibbsTVGLM(Y, priorSettings, modelSettings, 
    algoSettings);

PGAS_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims = 3);
plt = [plot() for i in 1:(p+q)]
PlotPostParamEvolution!(plt, PGAS_quantiles, "PGAS($(algoSettings.nParticles))",    
    groupSizes; 
    dateVec = nothing, interval_style = :shaded, lw = 2, c = :gray)
plot(plt..., layout = (2,2), size = (1400, 1000), xlabel = "time", 
    bottommargin = 5mm, legend = :bottomleft)


    # ### Laplace approximation
algoSettings = (; algoSettings..., stateSamplingMethod = :ffbs_laplace)
θpost, Hpost, ϕpost, σₙpost, μpost = GibbsTVGLM(Y, priorSettings, modelSettings, 
    algoSettings);

plt = [plot() for i in 1:(p+q)]
Laplace_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims = 3);
PlotPostParamEvolution!(plt, Laplace_quantiles, "Laplace", groupSizes; 
    dateVec = nothing, interval_style = :dash, lw = 2, c = colors[3])
plot(plt..., layout = (2,2), size = (1400, 1000), xlabel = "time", 
    bottommargin = 5mm, legend = :bottomleft)


plot(θpost[15,2,:])
# ### Iterated Posterior linearization filter
algoSettings = (; algoSettings..., stateSamplingMethod = :ffbs_slr)

θpost, Hpost, ϕpost, σₙpost, μpost = GibbsTVGLM(Y, priorSettings, modelSettings, 
    algoSettings);

IPLF_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims = 3);
PlotPostParamEvolution!(plt, IPLF_quantiles, "IPLF($(algoSettings.nMaxIter))", 
    groupSizes; 
    dateVec = nothing, interval_style = :solid, lw = 2, c = colors[1])
plot(plt..., layout = (2,2), size = (1400, 1000), xlabel = "time", 
    bottommargin = 5mm, legend = :bottomleft)




