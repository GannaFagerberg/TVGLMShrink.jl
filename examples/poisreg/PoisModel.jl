# Poisson regression model

# ### Set up the Poisson regression model
mutable struct ParamPoisReg{T,S<:AbstractMatrix{T}}
    Σᵥ::Vector{PDMat{T,S}}
    Z::Vector{Matrix{T}}
end


observation(param, state, t) = product_distribution(Poisson.(invlink_dgp.(param.Z[t] * state)))
condMean(param, state, t) = invlink_dgp.(param.Z[t] * state)
condCov(param, state, t) = diagm(invlink_dgp.(param.Z[t] * state))


## Prior for initial value of the state

function prior_t0(priorparam_λ, FisherInfo, κ₀, p)

    f(x) = priorparam_λ - invlink_dgp(x)
    m = find_zero(f, 0.0) 
    
    μ₀ = [m; zeros(p - 1)]
    Finfo = FisherInfo([], μ₀, 0) # Fisher information for one observation
    Σ₀ = (1 / κ₀) * inv((1 / T) * Finfo)
    return μ₀, Hermitian(Σ₀)

end

## Scaling 
function FisherInfo(θ, μ, t, X)
    S = X' * Diagonal(invlink_dgp.(X * μ)) * X
    return S
end

# Function that computes the Fisher info (not Scaling matrix) for all obs 
FisherInfo(θ, μ, t) = FisherInfo(θ, μ, t, X[:, covSel[1]])



function simulateVAR(T, Φ, Σₑ, μ)
    p = length(Φ)
    d = size(μ, 1)
    X = zeros(2*T, d)
    X[1:p,:] = μ
    L = sqrt(Σₑ)
    for t in (p + 1):(2*T)
        xₜ = copy(μ)
        for i in 1:p
            xₜ += Φ[i] * (X[t - i,:] - μ)
        end
        xₜ += L * randn(d)
        X[t,:] = xₜ
    end
    return X[(T + 1):end, :]
end


function simulateDSP(T, p, μ, φ, α, β, X, invlink, FisherInfo; initval = zeros(p)')
    y = zeros(T)
    λtime = zeros(T)
    θ = [initval; zeros(T,p)]
    h = [μ'; zeros(T,p)]
    η_t = zeros(p)

    for t in 2:(T+1)

        S = inv(sqrt(FisherInfo([],θ[t-1,:],t-1,X)/T))
        κ = rand(Beta(β, α), p)
        η_t = log.(1 ./ κ .- 1)

        h[t,:] = μ + φ .* (h[t-1,:] - μ) + η_t

        Σ_t = S * Diagonal(exp.(h[t,:])) * S
        Σ_t = Hermitian(Σ_t)
    
        ν_t = rand(MvNormal(zeros(p), Σ_t))

        θ[t,:] = θ[t-1,:] + ν_t
    end

    for i in 1:T
        λtime[i] = invlink.(dot(θ[i+1,:], X[i,:]))
        y[i] = rand.(Poisson.(λtime[i]))
    end
    return θ[2:end,:], y, λtime
end


function simulate_poisson_reg_data(T, p, covSel, invlink, φ, σₑ, mₑ)
    X = ones(T + 1)
    X = hcat(X, simulateVAR(T+1, [φ], σₑ, mₑ))
    X = X[2:end,:]
    β, y, λtime = simulateDSP(T, p, [-15, -15,-15], 0.5, 1/2, 1/2, X, invlink,FisherInfo)

    return y, X, β, λtime
end



# Plot the parameter evolution path of βₜ
function plot_param_path_poisreg(β)
    p = size(β, 2)
    plt = []
    for j = 1:p
        push!(plt, plot(β[:, j], label="true", xlabel="time, " * L"t",
            ylabel="", title=L"\beta_{%$(j-1)}", color=:black, lw=2))
    end
    plt = plot(plt..., layout=(3, 1), size=(1200, 1000), xguidefontsize=12,
        yguidefontsize=14, titlefontsize=20,
        legend=:bottomleft, margin=5mm)
    return plt
end

# plot \lambda time series
function plot_poisparam_evolution(λtime)

    p1 = plot(λtime, xlabel="time, " * L"t", title=L"\lambda_t", lw=2,
        color=colors[1], legend=nothing)
    plt = plot(p1, layout=(1, 1), size=(1200, 800),
        xguidefontsize=12, yguidefontsize=14, titlefontsize=18, margin=5mm)
    return plt
end

# Plot the evolution of the Poisson density over time and the time series
function plot_poisdensity_evolution(λtime, y)

    T = length(y)
    xgrid = 0:2:maximum(y) # grid of x values for plotting the Poisson density
    pdfvals = zeros(T, length(xgrid))
    for t in 1:T
        pdfvals[t, :] = pdf.(Poisson(λtime[t]), xgrid)
    end
    pdfvals = pdfvals ./ maximum(pdfvals, dims=2) # Normalize for better color scale
    # plot a heatmap of the pdf evolution with logpdf scale for the colors
    p1 = heatmap(1:T, xgrid, pdfvals', clims=(0, 1), color=:Blues,
        ylabel="density", xlabel="time, " * L"t", colorbar=false,
        title="evolution of Poisson density over time", colorbar_title="PDF (normalized)",
        size=(800, 600))

    p2 = plot(y, xlabel="time, " * L"t", ylabel=L"y_t", lw=1,
        color=colors[3], title="time series", legend=nothing)

    plt = plot(p1, p2, layout=(2, 1), size=(1200, 800),
        xguidefontsize=12, yguidefontsize=14, titlefontsize=18, margin=5mm)
    return plt

end
