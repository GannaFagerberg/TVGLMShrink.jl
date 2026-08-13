# Poisson regression model
observation(param, state, t) = product_distribution(Poisson.(linkinv.(param.link[1], param.Z[1][t] * state)))
condMean(param, state, t) = linkinv.(param.link[1], param.Z[1][t] * state)
condCov(param, state, t) = diagm(linkinv.(param.link[1], param.Z[1][t] * state))

function simulate_poisson_reg_data_fixed(T, p, covSel, link, φ, σₑ, mₑ, β₀)
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

        λtime[t] = linkinv.(link[1], dot(β[t, :], X[t, :]))
        y[t] = rand.(Poisson.(λtime[t]))
    end

    return y[2:end], X[2:end, :], β[2:end, :], λtime[2:end]
end

# Poisson regression model
observation(param, state, t) = product_distribution(Poisson.(linkinv.(param.link[1], param.Z[1][t] * state)))
condMean(param, state, t) = linkinv.(param.link[1], param.Z[1][t] * state)
condCov(param, state, t) = diagm(linkinv.(param.link[1], param.Z[1][t] * state))
