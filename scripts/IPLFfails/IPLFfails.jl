
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using PDMats, LogExpFunctions
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim
using Utils: mvcolors as colors 

# Scatter plot the state against the observations so see how linear it is
nSim = 1000
invlink(x) = exp(x)
p = 1
x = [1]
β = [1.0]
τ = 1.0
μ = β
Σ = τ^2*I(p)
stateDraws = rand(MvNormal(μ,Σ), nSim)
obsDraws = zeros(Int, nSim)
for i in 1:nSim
    obsDraws[i] = rand(Poisson(invlink(x ⋅ stateDraws[:,i])))
end

plt = []
for j in 1:p
    correlation = round(cor(stateDraws[j,:], obsDraws), digits = 3)
    push!(plt, scatter(stateDraws[j,:], obsDraws, color = colors[j], label = "",
        title = L"\mathrm{Corr}(\beta_{%$(j-1)}, y) = %$(correlation)", xlabel = "State $j",
        ylabel = "Observations", markersize = 5, markerstrokecolor = :black, markerstrokewidth = 0))
end                 
plot(plt..., layout = (2,2), size = (1000, 800))

# Exponentially distributed observations
nSim = 1000
invlink(x) = exp(x)
p = 1
x = [1]
β = [1.0]
τ = 1.0
μ = β
Σ = τ^2*I(p)
stateDraws = rand(MvNormal(μ,Σ), nSim)
obsDrawsExp = zeros(nSim)
for i in 1:nSim
    obsDrawsExp[i] = rand(Exponential(invlink(x ⋅ stateDraws[:,i])))
end 
pltExp = []
for j in 1:p
    correlation = round(cor(stateDraws[j,:], obsDrawsExp), digits = 3)
    push!(pltExp, scatter(stateDraws[j,:], obsDrawsExp, color = colors[j], label = "",
        title = L"\mathrm{Corr}(\beta_{%$(j-1)}, y) = %$(correlation)", xlabel = "State $j",
        ylabel = "Observations", markersize = 5, markerstrokecolor = :black, markerstrokewidth = 0))
end
plot(pltExp..., layout = (2,2), size = (1000, 800))

# Beta distributed observations
nSim = 1000
invlinkmean(x) = LogExpFunctions.logistic(x) # Inverse link function for Beta regression
invlinkprec(x) = exp(x) # Inverse link function for Beta regressionp = 1
x = [1]
β = [1.0]
τ = 1.0
μ = β
Σ = τ^2*I(p)
stateDraws = rand(MvNormal(μ,Σ), nSim)

obsDrawsBeta = zeros(nSim)
for i in 1:nSim
    mean = invlinkmean(x ⋅ stateDraws[:,i])
    precision = invlinkprec(x ⋅ stateDraws[:,i])
    obsDrawsBeta[i] = rand(Beta(mean*precision, (1-mean)*precision))
end
pltBeta = []
for j in 1:nStates
    correlation = round(cor(stateDraws[j,:], obsDrawsBeta), digits = 3)
    push!(pltBeta, scatter(stateDraws[j,:], obsDrawsBeta, color = colors[j], label = "",
        title = L"\mathrm{Corr}(\beta_{%$(j-1)}, y) = %$(correlation)", xlabel = "State $j",
        ylabel = "Observations", markersize = 5, markerstrokecolor = :black, markerstrokewidth = 0))
end
plot(pltBeta..., layout = (2,2), size = (1000, 800))        