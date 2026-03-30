# # Poisson regression with Dynamic Shrinkage Process parameter evolution

# In this example we explore the joint posterior in the Poisson regression model with parameters following independent dynamic shrinkage process priors.
#
# ```math
# \begin{align*}
#   y_t \vert \boldsymbol{x}_t &\sim \mathrm{Poisson}\big( \exp(\boldsymbol{x}_t^\top \boldsymbol{\beta}_t) \big) \\
# \boldsymbol{\beta}_t &= \boldsymbol{\beta}_{t-1} + \boldsymbol{\nu}_t, \quad \boldsymbol{\nu}_t \sim N\Big(\boldsymbol{0},\mathrm{Diag}(\exp(\boldsymbol{h}_t/2))\Big) \\
#   \boldsymbol{h}_t &= \boldsymbol{\mu} + \phi(\boldsymbol{h}_{t-1} -\boldsymbol{\mu}) + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim Z(\alpha,\alpha, 0, \sigma_\eta) \\
# \end{align*}
# ```
#

# First we load the required packages and set some plotting parameters:
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using PDMats
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim
using Utils: mvcolors as colors

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize = 18, markerstrokecolor = :auto)

figFolder = joinpath(@__DIR__)

Random.seed!(12345);


# ### Simulate data from the Poisson regression model with fixed parameter paths
T = 500;
p = 3;      # Number of parameters, including intercept
X = ones(T+1); # Design matrix
for i = 1:(p-1)
    X = hcat(X, simulateAR(T+1, [0.7], 1, 0))
end
y = zeros(Int, T+1)
β = zeros(T+1,p) # Store the regression parameters
β[1,:] = [0.0, 0.0, 0.5]
invlink_dgp(x) = exp_lin(x) # Inverse link function for Poisson regression
for t in 2:(T+1)
    β[t,1] = 1*sin(2π*t/T)
    if t < T/3
        β[t,2] = 0
    else 
        if t < ((2/3)*T)
            β[t,2] = -1
        else
            β[t,2] = 1
        end
    end
    β[t,3] = 0.5
    y[t] = rand(Poisson(invlink_dgp(X[t,:]⋅β[t,:])))
end
y = y[2:end];
X = X[2:end,:];

# ### Plot the parameter evolution path of βₜ and the time series
plt = []
for j = 1:p
    push!(plt, plot(β[:,j], label = "true", xlabel = "time, "*L"t", 
        ylabel = "", title = L"\beta_{%$(j-1)}", color = :black, lw = 2))
end
plot(plt..., layout = (p,1), size = (1200, 1000), xguidefontsize = 12, 
    ylim = [-1.5,1.5], yguidefontsize = 14, titlefontsize=20, 
    legend = :bottomleft, margin = 5mm) 

# Plot the time series
plot(y, xlabel = "time, "*L"t", ylabel = L"y_t", lw = 1,  
    color = colors[1], legend = nothing)
scatter!(y, markersize = 2, color = colors[1])

# ### Set up the prior, model and algorithm settings
priorSettings = (
    ϕ₀ = 0.5, κ₀ = 0.3,         # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀ = -15.0, σ₀ = 3.0,       # Prior for μ ~ N(m₀, σ₀²)
    ν₀ = 3.0, ψ₀ = 1.0,         # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀ = zeros(p), Σ₀ = 10*I(p),# Prior for βₜ at time t=0
); 

# ### Set up the Poisson regression model
mutable struct ParamTvReg{T, S<:AbstractMatrix{T}}
    Σᵥ::Vector{PDMat{T,S}}
    Z::Vector{Matrix{T}}
end

invlink(x) = exp_lin(x) # inverse link function for Poisson regression
observation(param, state, t) = product_distribution(Poisson.(invlink.(param.Z[t] * state)))
condMean(param, state, t) = invlink.(param.Z[t] * state)
condCov(param, state, t) = diagm(invlink.(param.Z[t] * state))

# #### Setting up data as grouped data
nPerGroup = 1
Y, Z, groupSizes = splitEqualGroups(y, X, nPerGroup)

# Instantiate model parameters (Σᵥ = I for all t), overwritten at each Gibbs iteration
param = ParamTvReg(LogVol2Covs(zeros(length(groupSizes), p)), Z) 

modelSettings = (
    observation = observation,
    param = param,
    condMean  = condMean,
    condCov   = condCov,
    α = 1/2,          # First shape param in Z distribution
    β = 1/2,          # Second shape param in Z distribution
    updateσₙ = false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp = 10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);

algoSettings = (
    stateSamplingMethod = :pgas, #:ffbs_laplace, # Algorithm to sample the state
    nParticles = 200,           # Number of particles if using PGAS
    nIter = 5000,               # Number of iterations in the Gibbs sampler
    nBurn = 1000,               # Number of burn-in iterations
    nMaxIter = 10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS = 500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod = eps(),       # Offset for log-volatility
    h_upper = Inf,               # Upper bound for log-volatility
    polyaoffset = 0.0           # Offset for Polya-Gamma variables in the update of h_t
);

# ### PGAS 
θpost, Hpost, ϕpost, σ²ₙpost, μpost, nFailure = GibbsTVGLM(Y, priorSettings, modelSettings, 
    algoSettings);

println("PGAS failed at $(100*nFailure[]/(algoSettings.nBurn+algoSettings.nIter))% of the simulated trajectories") 

PGAS_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims = 3);
PlotPostParamEvolution!(plt, PGAS_quantiles, "PGAS($(algoSettings.nParticles))",
    groupSizes; dateVec = nothing, interval_style = :shaded, lw = 2, c = :gray);
plot(plt..., layout = (3,1), size = (1400, 1000), xlabel = "time", 
    bottommargin = 5mm, ylims = [-1.5,1.5], legend = :bottomleft)

# ### Laplace approximation
algoSettings = (; algoSettings..., stateSamplingMethod = :ffbs_laplace)

θpost, Hpost, ϕpost, σ²ₙpost, μpost, nFailure = GibbsTVGLM(Y, priorSettings, modelSettings, 
    algoSettings);

println("Laplace failed at $(100*nFailure[]/(algoSettings.nBurn+algoSettings.nIter))% of the simulated trajectories") 

Laplace_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims = 3);
PlotPostParamEvolution!(plt, Laplace_quantiles, "Laplace", groupSizes; 
    dateVec = nothing, interval_style = :dash, lw = 2, c = colors[3])
plot(plt..., layout = (3,1), size = (1400, 1000), xlabel = "time", 
    bottommargin = 5mm, ylims = [-1.5,1.5], legend = :bottomleft)



# ### Iterated Posterior linearization filter
algoSettings = (; algoSettings..., stateSamplingMethod = :ffbs_slr)

θpost, Hpost, ϕpost, σ²ₙpost, μpost, nFailure = GibbsTVGLM(Y, priorSettings, modelSettings, 
    algoSettings);

println("IPLF failed at $(100*nFailure[]/(algoSettings.nBurn+algoSettings.nIter))% of the simulated trajectories") 

IPLF_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims = 3);
PlotPostParamEvolution!(plt, IPLF_quantiles, "IPLF($(algoSettings.nMaxIter))",
    groupSizes; 
    dateVec = nothing, interval_style = :solid, lw = 2, c = colors[1])
plot(plt..., layout = (3,1), size = (1400, 1000), xlabel = "time", 
    bottommargin = 5mm, ylims = [-1.5,1.5], legend = :bottomleft)



# ### Monte Carlo sampling
runMonteCarlo = false # slow
if runMonteCarlo
    algoSettings = (; algoSettings..., stateSamplingMethod = :montecarlo)

    θpost, Hpost, ϕpost, σ²ₙpost, μpost, nFailure = GibbsTVGLM(Y, priorSettings, modelSettings, 
        algoSettings);

    println("Monte Carlo failed at $(100*nFailure[]/(algoSettings.nBurn+algoSettings.nIter))% of the simulated trajectories") 

    MC_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims = 3);
    PlotPostParamEvolution!(plt, MC_quantiles, "MC($(algoSettings.nMaxIter))", 
        groupSizes; dateVec = nothing, interval_style = :solid, lw = 2, c = colors[4])
    plot(plt..., layout = (2,2), size = (1400, 1000), xlabel = "time", 
        bottommargin = 5mm, legend = :bottomleft)
end

savefig(figFolder*"PoisSimGroup1.pdf")