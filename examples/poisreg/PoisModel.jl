# Poisson regression model
observation(param, state, t) = product_distribution(Poisson.(invlink.(param.Z[1][t] * state)))
condMean(param, state, t) = invlink.(param.Z[1][t] * state)
condCov(param, state, t) = diagm(invlink.(param.Z[1][t] * state))
function FisherInfoPois(param, μ, t)
    F = XDiagX(param.X[1], invlink.(param.X[1] * μ)) # does X'Diagonal()*X fast 
    return F
end

function simulateDSP(T, p, μ, φ, α, β, X, invlink, FisherInfo; initval=zeros(p)')
    y = zeros(T)
    λtime = zeros(T)
    θ = [initval; zeros(T, p)]
    h = [μ'; zeros(T, p)]
    η_t = zeros(p)

    for t in 2:(T+1)

        S = inv(sqrt(FisherInfo([], θ[t-1, :], t - 1, X) / T))
        κ = rand(Beta(β, α), p)
        η_t = log.(1 ./ κ .- 1)

        h[t, :] = μ + φ .* (h[t-1, :] - μ) + η_t

        Σ_t = S * Diagonal(exp.(h[t, :])) * S
        Σ_t = Hermitian(Σ_t)

        ν_t = rand(MvNormal(zeros(p), Σ_t))

        θ[t, :] = θ[t-1, :] + ν_t
    end

    for i in 1:T
        λtime[i] = invlink.(dot(θ[i+1, :], X[i, :]))
        y[i] = rand.(Poisson.(λtime[i]))
    end
    return θ[2:end, :], y, λtime
end


function simulate_poisson_reg_data(T, p, covSel, invlink, φ, σₑ, mₑ)
    X = ones(T + 1)
    X = hcat(X, simulateVAR(T + 1, [φ], σₑ, mₑ))
    X = X[2:end, :]
    β, y, λtime = simulateDSP(T, p, [-15, -15, -15], 0.5, 1 / 2, 1 / 2, X, invlink, FisherInfo)

    return y, X, β, λtime
end

function simulate_poisson_reg_data_fixed(T, p, covSel, invlink, φ, σₑ, mₑ, β₀)
    X = ones(T + 1)
    X = hcat(X, simulateVAR(T + 1, [φ], σₑ, mₑ))
    β = zeros(T + 1, p)
    β[1, :] = β₀
    λtime = zeros(T + 1)
    y = zeros(T + 1)

    for t in 2:(T+1)
        β[t, 1] = sin(2π * t / T)
        if t < T / 3
            β[t, 2] = 0
        else
            if t < ((2 / 3) * T)
                β[t, 2] = -0.5
            else
                β[t, 2] = 1.0
            end
        end
        β[t, 3] = -0.05

        λtime[t] = invlink.(dot(β[t, :], X[t, :]))
        y[t] = rand.(Poisson.(λtime[t]))
    end

    return y[2:end], X[2:end, :], β[2:end, :], λtime[2:end]
end

function simulate_poisson_reg_data_fixed_together(T, p, covSel, invlink, φ, σₑ, mₑ, β₀)
    X = ones(T + 1)
    X = hcat(X, simulateVAR(T + 1, [φ], σₑ, mₑ))
    #X = moving_average_covariates(X, 20)
    β = zeros(T + 1, p)
    β[1, :] = β₀
    λtime = zeros(T + 1)
    y = zeros(T + 1)

    for t in 2:(T+1)
        β[t, 1] = β₀[1] + sin(2π * t / T)
        if t < T / 3
            β[t, 2] = β₀[2]
            β[t, 3] = β₀[3]
        else
            if t < ((2 / 3) * T)
                β[t, 2] = β₀[2] - 1
                β[t, 3] = β₀[3] + 0.0
            else
                β[t, 2] = β₀[2] + 1
                β[t, 3] = β₀[3] - 0.3
            end
        end

        λtime[t] = invlink.(dot(β[t, :], X[t, :]))
        y[t] = rand.(Poisson.(λtime[t]))
    end

    return y[2:end], X[2:end, :], β[2:end, :], λtime[2:end]
end



function simulate_poisson_reg_data_fixed_innov(T, p, covSel, invlink, FisherInfo,
    φ, σₑ, mₑ; initval=zeros(p))

    X = ones(T + 1)
    X = hcat(X, simulateVAR(T + 1, [φ], σₑ, mₑ))
    X[:, 2:end] .= X[:, 2:end] .- mean(X, dims=1)[2:end]' # Center the covariates

    μ₀ = [2, 0, 0]
    Σ₀ = Hermitian(inv(FisherInfo([], μ₀, 0, X) / T))

    β = zeros(T + 1, p)
    β[1, :] = rand(MvNormal(μ₀, Σ₀))
    λtime = zeros(T + 1)
    y = zeros(T + 1)

    S = sqrt(inv(FisherInfo([], μ₀, 0, X) / T))

    println(S)

    for t in 2:(T+1)

        ν = zeros(3)
        ν[1] = 0.01 * sin(2π * t / T)
        ν[2] = t == round(Int, T / 3) ? 1 : (t == round(Int, (2 * T / 3)) ? -1.0 : 0.0)
        ν[3] = t == round(Int, (2 * T / 3)) ? -2 : 0.0
        S = inv(sqrt(FisherInfo([], β[t-1, :], t - 1, X) / T))
        β[t, :] = β[t-1, :] + S * ν

        λtime[t] = invlink.(dot(β[t, :], X[t, :]))
        y[t] = rand.(Poisson.(λtime[t]))
    end

    return y[2:end], X[2:end, :], β[2:end, :], λtime[2:end]
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
    xgrid = 0:1:maximum(y) # grid of x values for plotting the Poisson density
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
