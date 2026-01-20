using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
cd(joinpath(@__DIR__, "../.."))
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using PDMats, LogExpFunctions
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim
using Utils: mvcolors as colors

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize = 18, markerstrokecolor = :auto)

Random.seed!(12345);

figFolder = joinpath(@__DIR__,"figs/")

BetaMean(μ, ψ) = Beta(eps() + μ*ψ, eps() + (1-μ)*ψ)

# ### Simulate data from the Beta regression model with fixed parameter paths
T = 500;
p = 3; # Number of parameters, including intercept
X = [ones(T+1) randn(T+1, p-1)]; # Design matrix
y = zeros(Float64, T+1)
β = zeros(T+1,p) # Store the regression parameters
β[1,:] = [0.0, 0.0, 0.5]
μtime = zeros(T+1)
ψtime = zeros(T+1)
ψfunc(t) = t/T < 0.4 ? 1 : 50 - 40*(t/T) # Precision parameter
invlinkmean_dgp(x) = logistic(x) # Inverse link function for Beta regression
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
    μtime[t] = invlinkmean_dgp(X[t,:]⋅β[t,:])
    ψtime[t] = ψfunc(t)
    y[t] = rand(BetaMean(μtime[t], ψtime[t]))
end
y = y[2:end];
X = X[2:end,:];
μtime = μtime[2:end];
ψtime = ψtime[2:end];
xgrid = 0.001:0.001:0.999
pdfvals = zeros(T,length(xgrid))
for t in 1:T
    pdfvals[t, :] = pdf.(BetaMean(μtime[t], ψtime[t]), xgrid)
end
pdfvals = pdfvals ./ maximum(pdfvals, dims = 2) # Normalize for better color scale
# plot a heatmap of the pdf evolution with logpdf scale for the colors
heatmap(xgrid, 1:T, pdfvals, clims = (0,1),
    xlabel = L"x", ylabel = "time, "*L"t", 
    title = "Evolution of Beta PDF over time", colorbar_title = "log PDF", 
    size = (800,600))


plot(quants[:,2], fillrange = quants[:,1], alpha = 0.2, color = :gray, lw = 0,
    label = "", xlabel = "time, "*L"t")
plot!(quants[:,2], fillrange = quants[:,3], alpha = 0.2, color = :gray, lw = 0,
    label = "")
#plot!(quants[:,2], color = :gray, lw = 1, label = "")
scatter!(y, markersize = 2, color = colors[3], label = "observations")

# ### Plot the parameter evolution path of βₜ
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
savefig(figFolder*"BetaReg_data.pdf")

# ### Set up the prior, model and algorithm settings
priorSettings = (
    ϕ₀ = 0.5, κ₀ = 0.3,         # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀ = -15.0, σ₀ = 3.0,       # Prior for μ ~ N(m₀, σ₀²)
    ν₀ = 3.0, ψ₀ = 1.0,         # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀ = zeros(p), Σ₀ = 10*I(p),# Prior for βₜ at time t=0
); 

# ### Set up the Poisson regression model
mutable struct ParamTvReg
    ψ::Float64
    Σᵥ::Vector{PDMat{Float64}}
    Z::Vector{Matrix{Float64}} # Covariates for each group
end

invlinkmean(x) = logistic(x) # inverse link function for μ in Beta regression
observation(param, state, t) = 
    product_distribution(BetaMean.(invlinkmean.(param.Z[t] ⋅ state), param.ψ))
condMean(param, state, t) = invlinkmean.(param.Z[t] * state)
function condCov(param, state, t) 
    μ = invlinkmean.(param.Z[t] * state)
    return diagm(μ .* (1 .- μ) ./ (1 + param.ψ))
end

# #### Setting up data as grouped data, here trivial grouping with 1 obs per group
Z = Vector{Matrix{Float64}}(undef, T)
for i = 1:T
    Z[i] = X[i,:]'
end
Y = Vector{Vector{Float64}}(undef, T) 
for i = 1:T
    Y[i] = [y[i]]
end

# Instantiate model parameters (Σᵥ = I for all t), overwritten at each Gibbs iteration
param = ParamTvReg(ψ, LogVol2Covs(zeros(T,p)), Z) 

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
    nIter = 1000,               # Number of iterations in the Gibbs sampler
    nBurn = 1000,               # Number of burn-in iterations
    nMaxIter = 10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS = 500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod = eps()        # Offset for log-volatility
);

# ### PGAS 
θpost, Hpost, ϕpost, σₙpost, μpost = GibbsTVGLM(Y, priorSettings, modelSettings, 
    algoSettings);

PGAS_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims = 3);
PlotPostParamEvolution!(plt, PGAS_quantiles, "PGAS($(algoSettings.nParticles))"; 
    dateVec = nothing, interval_style = :shaded, lw = 2, c = :gray)
plot(plt..., layout = (3,1), size = (1400, 1000), xlabel = "time", 
    bottommargin = 5mm, ylims = [-1.5,1.5], legend = :bottomleft)



# ### Laplace approximation
algoSettings = (; algoSettings..., stateSamplingMethod = :ffbs_laplace)

θpost, Hpost, ϕpost, σₙpost, μpost = GibbsTVGLM(Y, priorSettings, modelSettings, 
    algoSettings);

Laplace_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims = 3);
PlotPostParamEvolution!(plt, Laplace_quantiles, "Laplace"; 
    dateVec = nothing, interval_style = :dash, lw = 2, c = colors[3])
plot(plt..., layout = (3,1), size = (1400, 1000), xlabel = "time", 
    bottommargin = 5mm, ylims = [-1.5,1.5], legend = :bottomleft)



# ### Iterated Posterior linearization filter
algoSettings = (; algoSettings..., stateSamplingMethod = :ffbs_slr)

θpost, Hpost, ϕpost, σₙpost, μpost = GibbsTVGLM(Y, priorSettings, modelSettings, 
    algoSettings);

IPLF_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims = 3);
PlotPostParamEvolution!(plt, IPLF_quantiles, "IPLF($(algoSettings.nMaxIter))"; 
    dateVec = nothing, interval_style = :solid, lw = 2, c = colors[1])
plot(plt..., layout = (3,1), size = (1400, 1000), xlabel = "time", 
    bottommargin = 5mm, ylims = [-1.5,1.5], legend = :bottomleft)

savefig(figFolder*"PoisReg3.pdf")
