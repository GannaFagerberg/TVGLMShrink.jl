using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
cd(joinpath(@__DIR__, "../.."))
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using PDMats, LogExpFunctions
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim, get_slurm_id
using Utils: mvcolors as colors
using CSV, DataFrames, Dates

slurm_id = get_slurm_id() # get slurm ID, if on cluster

Random.seed!(slurm_id);

includet("BetaModel.jl")  # Load simulator, Fisher info and plotting for BetaReg

gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=18, markerstrokecolor=:auto)

#figFolder = joinpath(@__DIR__,"figs/")

indpro = CSV.read("/Users/niuyijie/Downloads/INDPRO (2).csv", DataFrame)
indpc1 = CSV.read("/Users/niuyijie/Downloads/INDPRO_PC1.csv", DataFrame)

cpi = CSV.read("/Users/niuyijie/Downloads/CPIAUCSL (6).csv", DataFrame)
cpipc1 = CSV.read("/Users/niuyijie/Downloads/CPIAUCSL_PC1.csv", DataFrame)
cpipc1[findall(ismissing, cpipc1[:, 2]), 2] .= cpipc1[findall(ismissing, cpipc1[:, 2]).-1, 2]

ppi = CSV.read("/Users/niuyijie/Downloads/PPIACO_PC1.csv", DataFrame)

interest = CSV.read("/Users/niuyijie/Downloads/IR3TIB01USM156N (1).csv", DataFrame)

civpart = CSV.read("/Users/niuyijie/Downloads/CIVPART (1).csv", DataFrame)
unrate = CSV.read("/Users/niuyijie/Downloads/UNRATE (1).csv", DataFrame)
epr = CSV.read("/Users/niuyijie/Downloads/EMRATIO.csv", DataFrame)
unrate[findall(ismissing, unrate[:, 2]), 2] .= unrate[findall(ismissing, unrate[:, 2]).-1, 2]

bdf1 = CSV.read("/Users/niuyijie/Downloads/LNS13026511 (2).csv", DataFrame)
bdf2 = CSV.read("/Users/niuyijie/Downloads/LNS13023654.csv", DataFrame)
bdf3 = CSV.read("/Users/niuyijie/Downloads/LNS13023622.csv", DataFrame)

bdf1[findall(ismissing, bdf1[:, 2]), 2] .= bdf1[findall(ismissing, bdf1[:, 2]).-1, 2]
bdf2[findall(ismissing, bdf2[:, 2]), 2] .= bdf2[findall(ismissing, bdf2[:, 2]).-1, 2]
bdf3[findall(ismissing, bdf3[:, 2]), 2] .= bdf3[findall(ismissing, bdf3[:, 2]).-1, 2]

civpart[findall(ismissing, civpart[:, 2]), 2] .= civpart[findall(ismissing, civpart[:, 2]).-1, 2]
interest[findall(ismissing, interest[:, 2]), 2] .= interest[findall(ismissing, interest[:, 2]).-1, 2]


y1 = bdf1[1:findfirst(bdf1[:, 1] .== Date(2025, 4, 1)), 2] ./ 100
y2 = bdf2[1:findfirst(bdf2[:, 1] .== Date(2025, 4, 1)), 2] ./ 100
y3 = bdf3[1:findfirst(bdf3[:, 1] .== Date(2025, 4, 1)), 2] ./ 100


y1 = bdf1[:, 2] ./ 100
y2 = bdf2[:, 2] ./ 100
y3 = bdf3[:, 2] ./ 100


x1 = log.(indpro[findfirst(indpro[:, 1] .== Date(1966, 10, 1)):end-2, 2]) - log.(indpro[findfirst(indpro[:, 1] .== Date(1966, 09, 1)):end-3, 2])
x11 = indpc1[findfirst(indpc1[:, 1] .== Date(1966, 10, 1)):end-4, 2] ./ 100

x2 = log.(cpi[findfirst(cpi[:, 1] .== Date(1966, 07, 1)):end-16, 2]) - log.(cpi[findfirst(cpi[:, 1] .== Date(1966, 06, 1)):end-17, 2])
x21 = cpipc1[findfirst(cpipc1[:, 1] .== Date(1966, 7, 1)):end-8, 2] ./ 100
x21 = Float64.(vec(x21))


x22 = cpipc1[findfirst(cpipc1[:, 1] .== Date(1966, 12, 1)):end-1, 2] ./ 100
x22 = Float64.(vec(x22))


x3 = (interest[findfirst(interest[:, 1] .== Date(1966, 10, 1)):end-2, 2]) ./ 100

x41 = ppi[findfirst(ppi[:, 1] .== Date(1966, 12, 1)):end-10, 2]

x5 = unrate[findfirst(unrate[:, 1] .== Date(1966, 12, 1)):end-11, 2] ./ 100
x6 = civpart[findfirst(civpart[:, 1] .== Date(1966, 12, 1)):end-1, 2] ./ 100
#x7 = epr[findfirst(epr[:,1] .== Date(1966,12,1)):end-11, 2]./100

bdf2[200, 1]
y = y2
plot(y)
T = size(y, 1)
x11 = indpc1[findfirst(indpc1[:, 1] .== Date(1966, 6, 1)):end-6, 2] ./ 100
x21 = cpipc1[findfirst(cpipc1[:, 1] .== Date(1966, 4, 1)):end-11, 2] ./ 100
x22 = cpipc1[findfirst(cpipc1[:, 1] .== Date(1966, 11, 1)):end-4, 2] ./ 100

x5 = unrate[findfirst(unrate[:, 1] .== Date(1966, 6, 1)):end-9, 2] ./ 100

#p = 2; # Number of covariate in μ, including intercept
q = 1; # Number of covariates in ψ, including intercept
Xmean = [ones(T) x21]; # Design matrix
p = size(Xmean, 2)
Xprec = [ones(T) randn(T, q - 1)];                 # Design matrix for precision
logistic(x) = 1 / (1 + exp(-x))
invlinkmean_dgp(x) = logistic(x) # Inverse link function for Beta regression
invlinkprec_dgp(x) = exp(x) # Inverse link function for Beta regression


#### scaling
logit(x) = log(x ./ (1 .- x))
E_μ = mean(y)
c = exp(-8) #mean(1 ./ (ψtime .+ 1))
V_μ = (var(y) - c * (E_μ - E_μ^2)) / (1 - c)
phi = (E_μ * (1 - E_μ) / V_μ - 1)
μ_sim = rand(BetaMean(E_μ, phi), 1000000)
mean(y)
βₘ = [mean(logit.(μ_sim)), zeros(p - 1)...]
Σₘ = inv(1 / T * Xmean[:, 2:end]' * Diagonal(invlinkmean_dgp.(Xmean * βₘ)) * Xmean[:, 2:end])
Σₒ = [var(logit.(μ_sim)) zeros(1, size(Σₘ, 2));
    zeros(size(Σₘ, 1), 1) Σₘ]
Σₒ = 0.5 * (Σₒ + Σₒ')
scaling_mean = inv(sqrt(Σₒ))
vphi = exp(1)
scaling = [inv(sqrt(Σₒ)) zeros(size(Σₒ, 1), q);
    zeros(q, size(Σₒ, 1)) vphi]

scaling = I(p + q)
scaling_mean = I(p)
vphi = 1
# ### Set up the Beta regression model
mutable struct ParamTvReg{T,S<:AbstractMatrix{T}}
    Σᵥ::Vector{PDMat{T,S}}
    Zmean::Vector{Matrix{T}}
    Zprec::Vector{Matrix{T}}
end

invlinkmean(x) = logistic(x) # inverse link function for μ in Beta regression
invlinkprecision(x) = exp(x) # inverse link function for ψ in Beta regression
observation(param, state, t) =
    product_distribution(
        BetaMean.(
            invlinkmean.(param.Zmean[t] * inv(scaling_mean) * state[1:p]),
            invlinkprecision.(1 / vphi * param.Zprec[t] * state[(p+1):(p+q)])
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
param = ParamTvReg(LogVol2Covs(zeros(length(groupSizes), p + q)), Zmean, Zprec)

# ### Set up the prior, model and algorithm settings
priorSettings = (
    ϕ₀=0.5, κ₀=0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀=3.0, ψ₀=1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=zeros(p + q), Σ₀=5 * I(p + q), # Prior for βₜ at time t=0
);

modelSettings = (
    observation=observation,
    param=param,
    condMean=condMean,
    condCov=condCov,
    α=1 / 2,
    β=1 / 2,
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);

algoSettings = (
    stateSamplingMethod=:pgas,# Algorithm to sample the state
    nParticles=200,           # Number of particles if using PGAS
    nIter=3000,               # Number of iterations in the Gibbs sampler
    nBurn=4000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.0,           # Offset for Polya-Gamma variables in the update of h_t
    scaling=scaling          # Scaling for the state
);

# ### PGAS 
θpost, Hpost, ϕpost, σ²ₙpost, μpost = GibbsTVGLM(Y, priorSettings, modelSettings,
    algoSettings);

PGAS_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);
plt = [plot() for i in 1:(p+q)]
PlotPostParamEvolution!(plt, PGAS_quantiles, "PGAS($(algoSettings.nParticles))",
    groupSizes;
    dateVec=nothing, interval_style=:shaded, lw=2, c=:gray)
plot(plt..., layout=(2, 2), size=(1400, 1000), xlabel="time",
    bottommargin=5mm, legend=:bottomleft)

# j=1
#bdf[:,1] = Date.(bdf[:,1])
#n_ticks = 10
#total_dates = length(bdf[:,1])
#xtick_idx = round.(Int, range(1, stop=total_dates, length=n_ticks))
#xticks_vals = bdf[xtick_idx,1]
#for j = 1:p
#   push!(plt, plot(bdf[:,1],PGAS_quantiles_expand[:,j,2], xticks = (xticks_vals, string.(year.(xticks_vals))),label = "true", xlabel = "time, "*L"t", 
#       ylabel = "", title = L"\beta_{%$(j-1)}", color = :black, lw = 2))
#end
#for j = 1:q
#   push!(plt, plot(bdf[:,1],PGAS_quantiles_expand[:,(p+j),2], xticks = (xticks_vals, string.(year.(xticks_vals))), label = "true", xlabel = "time, "*L"t", 
#       ylabel = "", title = L"\gamma_{%$(j-1)}", color = :black, lw = 2))
#end
#PlotPostParamEvolution!(plt, PGAS_quantiles_expand, "PGAS($(algoSettings.nParticles))"; 
#   dateVec = bdf[:,1], interval_style = :shaded, lw = 2, c = :gray, sample_t0 = false)
#plot(plt..., layout = (3,1), size = (1400, 1000), xlabel = "time", 
#   bottommargin = 5mm, legend = :bottomleft)


# ### Laplace approximation
algoSettings = (; algoSettings..., stateSamplingMethod=:ffbs_laplace)
θpost, Hpost, ϕpost, σₙpost, μpost = GibbsTVGLM(Y, priorSettings, modelSettings,
    algoSettings);

plt = [plot() for i in 1:(p+q)]
Laplace_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);
PlotPostParamEvolution!(plt, Laplace_quantiles, "Laplace", groupSizes;
    dateVec=nothing, interval_style=:dash, lw=2, c=colors[3])
plot(plt..., layout=(2, 2), size=(1400, 1000), xlabel="time",
    bottommargin=5mm, legend=:bottomleft)

plot(plt[2])
hline!([0])
plot(θpost[100, 1, :])
# ### Iterated Posterior linearization filter
algoSettings = (; algoSettings..., stateSamplingMethod=:ffbs_slr)

θpost, Hpost, ϕpost, σₙpost, μpost = GibbsTVGLM(Y, priorSettings, modelSettings,
    algoSettings);

IPLF_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);
PlotPostParamEvolution!(plt, IPLF_quantiles, "IPLF($(algoSettings.nMaxIter))",
    groupSizes;
    dateVec=nothing, interval_style=:solid, lw=2, c=colors[1])
plot(plt..., layout=(2, 2), size=(1400, 1000), xlabel="time",
    bottommargin=5mm, legend=:bottomleft)




e = [0.5241171288681834, -0.12025572602569669, 0.004281294169531033]
g = [1.0 -0.7801022036385404 18.66683806041292; 1.0 -0.2907888115596499 -43.741171238325165; 1.0 -0.597761124088303 173.9634990407811; 1.0 0.1548091262400686 85.9696622515452; 1.0 -0.37266588111728444 -9.474215932607654; 1.0 1.0770174488462274 -33.1601314901109; 1.0 -0.7657527159695439 97.64275035023564; 1.0 -1.5340243720661564 42.99749975360986; 1.0 0.5418157863580144 -92.89671333522764; 1.0 1.155571132545481 94.51913664173681]
g = [1.0 -0.7801022036385404 18.66683806041292; 1.0 -0.2907888115596499 -43.741171238325165]
fi = (g' * Diagonal(exp.(g * e))) * g
fi^(-1 / 2)

e = [0.5203285469386031, -0.26549142359939093, -0.06686266910272463]
g = [1.0 1.7451720974538483 -0.6353798067979661; 1.0 0.3046697147846464 0.4257409416165618]