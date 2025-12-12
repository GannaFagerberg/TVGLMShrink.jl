using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim
using Utils: mvcolors as colors

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=10, ytickfontsize=10, xguidefontsize=8, yguidefontsize=10,
    titlefontsize = 14, markerstrokecolor = :auto)

Random.seed!(123);

# ### Simulate data from the Poisson regression model with fixed parameter paths
T = 500;
p = 3; # Number of parameters, including intercept
X = [ones(T+1) randn(T+1, p-1)]; # Design matrix
y = zeros(Int, T+1)
β = zeros(T+1,p) # Store the regression parameters
β[1,:] = [0.0, 0.0, 0.5]
invlink(x) = exp(x) # Inverse link function for Poisson regression
for t in 2:(T+1)
    β[t,1] = sin(2π*t/T)
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
    y[t] = rand(Poisson(invlink(X[t,:] ⋅ β[t,:])))
end
y = y[2:end];
X = X[2:end,:];

# ### Plot the parameter evolution path of βₜ
plt = []
for j = 1:p
    push!(plt, plot(β[:,j], label = L"\beta_{%$(j-1)}", xlabel = "time, "*L"t", 
        ylabel = L"\beta_{%$(j-1)}", title = "Path of "*L"\beta_{%$(j-1)}", 
        color = colors[j], lw = 2))
end
plot(plt..., layout = (p,1), size = (1200, 800), xguidefontsize = 12, 
    ylim = (-1.2,1.2),
    yguidefontsize = 14, titlefontsize=16, legend = nothing, margin = 5mm) 

p1 = plot(X[:,2], y, seriestype = :scatter, markersize = 2,    
    xlabel = L"x_1", ylabel = "y", title = "Simulated data from Poisson regression model", 
    color = colors[2], legend = nothing)

p2 = plot(X[:,3], y, seriestype = :scatter, markersize = 2,    
    xlabel = L"x_2", ylabel = "y", title = "Simulated data from Poisson regression model", 
    color = colors[2], legend = nothing)

p3 = plot(y, xlabel = "time, "*L"t", ylabel = L"y_t", lw = 1,  
    color = colors[1], legend = nothing)
p3 = scatter!(y, markersize = 3, color = colors[1])

