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

BetaMean(μ, ψ) = Beta(5.0e-5 + μ*ψ, 5.0e-5 + (1-μ)*ψ)

# ### Simulate data from the Beta regression model with fixed parameter paths
T = 500;
p = 3; # Number of covariate in μ, including intercept
q = 1; # Number of covariates in ψ, including intercept
Xmean = [ones(T+1) randn(T+1, p-1)]; # Design matrix
Xprec = [ones(T+1) randn(T+1, q-1)];                 # Design matrix for precision
y = zeros(Float64, T+1)
β = zeros(T+1,p) # Store the regression parameters
γ = zeros(T+1,q) # Store the precision parameters
β[1,:] = [0, 0, -0.5]
γ[1,:] = [1] # log precision intercept
μtime = zeros(T+1)
ψtime = zeros(T+1)
invlinkmean_dgp(x) = logistic(x) # Inverse link function for Beta regression
invlinkprec_dgp(x) = exp(x) # Inverse link function for Beta regression
for t in 2:(T+1)
    β[t,1] = 0.5*sin(2π*t/T)
    β[t,2] = (t/T)^2
    β[t,3] = -0.5
    γ[t,1] = 1.0
    μtime[t] = invlinkmean_dgp(Xmean[t,:]⋅β[t,:])
    ψtime[t] = invlinkprec_dgp(Xprec[t,:]⋅γ[t,:])
    y[t] = rand(BetaMean(μtime[t], ψtime[t]))
end
y = y[2:end];
Xmean = Xmean[2:end,:];
Xprec = Xprec[2:end,:];
μtime = μtime[2:end];
ψtime = ψtime[2:end];
αtime = μtime .* ψtime
βtime = (1 .- μtime) .* ψtime
X = [Xmean, Xprec]
p = size(X[1], 2)
q = size(X[2], 2)



# plot \mu and \psi time series
p1 = plot(μtime, xlabel = "time, "*L"t", title = L"\mu_t", lw = 2,  
    color = colors[1], legend = nothing)
p2 = plot(ψtime, xlabel = "time, "*L"t", title = L"\psi_t", lw = 2,  
    color = colors[3], legend = nothing)    
p3 = plot(αtime, xlabel = "time, "*L"t", title = L"\alpha_t", lw = 2,  
    color = colors[2], legend = nothing)
p4 = plot(βtime, xlabel = "time, "*L"t", title = L"\beta_t", lw = 2,  
    color = colors[4], legend = nothing)    
plot(p1, p2, p3, p4, layout = (4,1), size = (1000,1200), 
    xguidefontsize = 12, yguidefontsize = 14, titlefontsize=18, margin = 5mm)

minimum(αtime)
minimum(βtime)
extrema(y)

# ### Plot the parameter evolution path of βₜ
plt = []
for j = 1:p
    push!(plt, plot(β[:,j], label = "true", xlabel = "time, "*L"t", 
        ylabel = "", title = L"\beta_{%$(j-1)}", color = :black, lw = 2))
end
for j = 1:q
    push!(plt, plot(γ[:,j], label = "true", xlabel = "time, "*L"t", 
        ylabel = "", title = L"\gamma_{%$(j-1)}", color = :black, lw = 2))
end
plot(plt..., layout = (2,2), size = (1200, 1000), xguidefontsize = 12, 
    yguidefontsize = 14, titlefontsize=20, 
    legend = :bottomleft, margin = 5mm) 

# ### Plot the evolution of the Beta density over time and the time series
xgrid = 0.001:0.001:0.999
pdfvals = zeros(T, length(xgrid))
for t in 1:T
    pdfvals[t, :] = pdf.(BetaMean(μtime[t], ψtime[t]), xgrid)
end
pdfvals = pdfvals ./ maximum(pdfvals, dims = 2) # Normalize for better color scale
# plot a heatmap of the pdf evolution with logpdf scale for the colors
p1 = heatmap(1:T, xgrid, pdfvals', clims = (0,1), color  = :Blues,
    ylabel = "density", xlabel = "time, "*L"t", colorbar = false,
    title = "evolution of Beta density over time", colorbar_title = "PDF (normalized)", 
    size = (800,600))

p2 = plot(y, xlabel = "time, "*L"t", ylabel = L"y_t", lw = 1,  
    color = colors[3], title = "time series", legend = nothing)

plot(p1, p2, layout = (2,1), size = (800,800), 
    xguidefontsize = 12, yguidefontsize = 14, titlefontsize=18, margin = 5mm)


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
nPerGroup = 1
Y, Z, groupSizes = splitEqualGroups(y, X, nPerGroup)
Zmean = Z[1]
Zprec = Z[2]

# Instantiate model parameters (Σᵥ = I for all t), overwritten at each Gibbs iteration
param = ParamTvReg(LogVol2Covs(zeros(length(groupSizes),p+q)), Zmean, Zprec) 

# ### Set up the prior, model and algorithm settings
priorSettings = (
    ϕ₀ = 0.5, κ₀ = 0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀ = -10.0, σ₀ = 3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀ = 3.0, ψ₀ = 1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀ = zeros(p+q), Σ₀ = 1*I(p+q), # Prior for βₜ at time t=0
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
    nIter = 1000,               # Number of iterations in the Gibbs sampler
    nBurn = 5000,               # Number of burn-in iterations
    nMaxIter = 10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS = 500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod = eps(),       # Offset for log-volatility
    h_upper = Inf               # Upper bound for log-volatility
);

# ### PGAS 
θpost, Hpost, ϕpost, σ²ₙpost, μpost = GibbsTVGLM(Y, priorSettings, modelSettings, 
    algoSettings);

PGAS_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims = 3);
PlotPostParamEvolution!(plt, PGAS_quantiles, "PGAS($(algoSettings.nParticles))",    
    groupSizes; 
    dateVec = nothing, interval_style = :shaded, lw = 2, c = :gray)
plot(plt..., layout = (2,2), size = (1400, 1000), xlabel = "time", 
    bottommargin = 5mm, ylims = [-1.5,1.5], legend = :bottomleft)


# ### Laplace approximation
algoSettings = (; algoSettings..., stateSamplingMethod = :ffbs_laplace)

θpost, Hpost, ϕpost, σ²ₙpost, μpost, nFailure = GibbsTVGLM(Y, priorSettings, modelSettings, 
    algoSettings);

println("Laplace failed at $(100*nFailure[]/(algoSettings.nBurn+algoSettings.nIter))% of the simulated trajectories")

Laplace_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims = 3);
PlotPostParamEvolution!(plt, Laplace_quantiles, "Laplace", groupSizes; 
    dateVec = nothing, interval_style = :dash, lw = 2, c = colors[3])
plot(plt..., layout = (2,2), size = (1400, 1000), xlabel = "time", 
    bottommargin = 5mm, ylims = [-1.5,1.5], legend = :bottomleft)


# ### Iterated Posterior linearization filter
algoSettings = (; algoSettings..., stateSamplingMethod = :ffbs_slr)

θpost, Hpost, ϕpost, σ²ₙpost, μpost = GibbsTVGLM(Y, priorSettings, modelSettings, 
    algoSettings);

IPLF_quantiles = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims = 3);
PlotPostParamEvolution!(plt, IPLF_quantiles, "IPLF($(algoSettings.nMaxIter))", 
    groupSizes; 
    dateVec = nothing, interval_style = :solid, lw = 2, c = colors[1])
plot(plt..., layout = (2,2), size = (1400, 1000), xlabel = "time", 
    bottommargin = 5mm, legend = :bottomleft)
