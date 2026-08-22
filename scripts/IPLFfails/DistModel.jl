
using LinearAlgebra
using Random

Random.seed!(123)

## Simulate data from the different regression model with fixed parameter paths
T = 500;
β₀ = [2, 0]
p = size(β₀, 1)[1]

## Generate covariate
X = ones(T + 1)
X = hcat(X, simulateAR(T + 1, [0.7], 1.0))

## fixed parameter paths
function simulate_μ(X, T, p, link, β₀)
    β = zeros(T + 1, p)
    β[1, :] = β₀
    λtime = zeros(T + 1)

    for t in 2:(T+1)
        β[t, 1] = sin(2π * t / T)
        if t < T / 3
            β[t, 2] = 0
        else
            if t < ((2 / 3) * T)
                β[t, 2] = -1
            else
                β[t, 2] = 1
            end
        end

        λtime[t] = linkinv.(link[1], dot(β[t, :], X[t, :]))
        end

    return β[2:end, :], λtime[2:end]
end
 
## Poisson and Exponential regression model
link = (LogLinLink(),)
β, λtime = simulate_μ(X, T, p, link, β₀);
y = zeros(T)
for i in 1:T
    y[i] = rand.(Poisson.(λtime[i]))
end
PoisData = [y X[2:end,:] β λtime]


### Exponential regression
y = zeros(T)
for i in 1:T
    y[i] = rand.(Exponential.(λtime[i]))
end
ExpData = [y X[2:end,:] β λtime]



## NegativeBinomial
NegBinomMean(μ, r) = NegativeBinomial(r, r ./ (r .+ μ))
## fixed parameter paths
function simulate_neg(X, T, p, link, β₀)
    β = zeros(T + 1, p)
    β[1, :] = β₀
    λtime = zeros(T + 1)

    for t in 2:(T+1)
        β[t, 1] = sin(2π * t / T)
        if t < T / 3
            β[t, 2] = 1.5
        else
            if t < ((2 / 3) * T)
                β[t, 2] = 1
            else
                β[t, 2] = 2
            end
        end

        λtime[t] = linkinv.(link[1], dot(β[t, :], X[t, :]))
        end

    return β[2:end, :], λtime[2:end]
end
link = (LogLinLink(),)
β1, λtime = simulate_neg(X, T, p, link, β₀);
γ1 = [t < T / 2 ? -1.5 : 0.0 for t in 1:T]
ψtime = exp.(γ1)

y = [rand(NegBinomMean(λtime[t],ψtime[t])) for t in 1:T]
NegbinomData = [y X[2:end,:] β1 γ1 λtime ψtime]

## Beta regression
γ = [t < T / 2 ? 1 : 3 for t in 1:T]
ψtime = exp.(γ)
BetaMean(μ, ψ) = Beta(1.0e-15 + μ * ψ, 1.0e-15 + (1 - μ) * ψ)
invlink_logit = (LogitLink(),)
_,μtime = simulate_μ(X, T, p, invlink_logit, β₀)
y = [rand(BetaMean(μtime[t],ψtime[t])) for t in 1:T]
y = clamp.(y, 1e-16, 1-1e-16)
BetaData = [y X[2:end,:] β γ μtime ψtime ]


function plot_param_path_poisreg(β)
    p = size(β, 2)
    plt = []
    for j = 1:p
        push!(plt, plot(β[:, j], label="true", xlabel="time, " * L"t",
            ylabel="", title=L"\beta_{%$(j-1)}", color=:black, lw=3))
    end
    plt = plot(plt..., layout=(2, 1), size=(1200, 1000), xguidefontsize=12,
        yguidefontsize=14, titlefontsize=20,
        legend=:bottomleft, margin=5mm)
    return plt
end

function plot_param_path_betareg(β, γ)
    p = size(β, 2)
    q = size(γ, 2)
    plt = []
    for j = 1:p
        push!(plt, plot(β[:, j], label="true", xlabel="time, " * L"t",
            ylabel="", title=L"\beta_{%$(j-1)}", color=:black, lw=3))
    end
    for j = 1:q
        push!(plt, plot(γ[:, j], label="true", xlabel="time, " * L"t",
            ylabel="", title=L"\gamma_{%$(j-1)}", color=:black, lw=3))
    end
    plt = plot(plt..., layout=(3, 1), size=(1200, 1000), xguidefontsize=12,
        yguidefontsize=14, titlefontsize=20,
        legend=:bottomleft, margin=5mm)
    return plt
end
