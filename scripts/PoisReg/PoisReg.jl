using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
cd(joinpath(@__DIR__, "../.."))
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using PDMats
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim
using Utils: mvcolors as colors

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize = 18, markerstrokecolor = :auto)

Random.seed!(12345);

figFolder = joinpath(@__DIR__,"figs/")

# ### Simulate data from the Poisson regression model with fixed parameter paths
T = 500;
p = 3; # Number of parameters, including intercept
X = [ones(T+1) randn(T+1, p-1)]; # Design matrix
y = zeros(Int, T+1)
β = zeros(T+1,p) # Store the regression parameters
β[1,:] = [0.0, 0.0, 0.5]
invlink_dgp(x) = exp(x) # Inverse link function for Poisson regression
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

# ### Plot the parameter evolution path of βₜ
plt = []
for j = 1:p
    push!(plt, plot(β[:,j], label = "true", xlabel = "time, "*L"t", 
        ylabel = "", title = L"\beta_{%$(j-1)}", color = colors[3], lw = 3))
end
plot(plt..., layout = (p+1,1), size = (1200, 1000), xguidefontsize = 12, 
    ylim = (-1.2,1.2), yguidefontsize = 14, titlefontsize=16, legend = :bottomleft, 
    margin = 5mm) 

# Plot the time series
p3 = plot(y, xlabel = "time, "*L"t", ylabel = L"y_t", lw = 1,  
    color = colors[1], legend = nothing)
p3 = scatter!(y, markersize = 2, color = colors[1])
savefig(figFolder*"PoisReg_dataNew.pdf")

p2 = plot(X[:,2], log.(y .+ 1), seriestype = :scatter, markersize = 2,    
    xlabel = L"x_{1t}", ylabel = L"y_t", color = colors[3], legend = nothing)

p3 = plot(X[:,3], log.(y .+ 1), seriestype = :scatter, markersize = 2,    
    xlabel = L"x_{2t}", ylabel = L"y_t", color = colors[3], legend = nothing)


plot(p2, p3, layout = (3,1), size = (600, 800), 
    xguidefontsize = 12, yguidefontsize = 14, titlefontsize=16, margin = 2mm)

# ### Set up the prior, model and algorithm settings
priorSettings = (
    ϕ₀ = 0.5, κ₀ = 0.3,         # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀ = -12.0, σ₀ = 3.0,       # Prior for μ ~ N(m₀, σ₀²)
    ν₀ = 3.0, ψ₀ = 1.0,         # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀ = zeros(p), Σ₀ = 1*I(p),# Prior for βₜ at time t=0
); 

# ### Set up Poisson model
mutable struct ParamTvReg
    Σᵥ::Vector{PDMat{Float64}}
    Z::Vector{Matrix{Float64}} # Covariates for each group
end

# Setting up data as grouped data, here trivial grouping with 1 obs per group
Z = Vector{Matrix{Float64}}(undef, T)
for i = 1:T
    Z[i] = X[i,:]'
end
Y = Vector{Vector{Float64}}(undef, T) 
for i = 1:T
    Y[i] = [y[i]]
end

param = ParamTvReg(LogVol2Covs(zeros(T,p)), Z)
invlink(x) = exp(x) # inverse link function for Poisson regression
observation(param, state, t) = product_distribution(Poisson.(invlink.(param.Z[t] ⋅ state)))
condMean(param, state, t) = exp.(param.Z[t] * state)
condCov(param, state, t) = diagm(exp.(param.Z[t] * state))

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
    nParticles = 100,           # Number of particles if using PGAS
    nIter = 1000,               # Number of iterations in the Gibbs sampler
    nBurn = 1000,               # Number of burn-in iterations
    offsetMethod = eps()        # Offset for log-volatility
);


# ### Run the Gibbs sampler
θpost, Hpost, ϕpost, σₙpost, μpost = GibbsTVGLM(Y, priorSettings, modelSettings, 
    algoSettings);

Laplace_median = median(θpost, dims = 3)[:,:,1]
Laplace_quantiles = quantile_multidim(θpost, [0.025, 0.975], dims = 3);
for j = 1:p
    plot!(plt[j], 0:T, Laplace_median[:,j], lw = 3, c = colors[1], 
        linestyle = :solid, label = "Laplace")
    plot!(plt[j], 0:T, Laplace_quantiles[:,j,1], lw = 3, c = colors[1], 
        linestyle = :dash, label = nothing)
    plot!(plt[j], 0:T, Laplace_quantiles[:,j,2], lw = 3, c = colors[1], 
        linestyle = :dash, label = nothing)
end
plot(plt..., layout = (3,1), size = (1400, 1000), xlabel = "time", 
    bottommargin = 5mm, ylims = [-1.5,1.5], legend = :bottomleft)
savefig(figFolder*"PoisReg_Laplace.pdf")

# IPLF
