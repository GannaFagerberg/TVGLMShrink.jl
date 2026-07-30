# Poisson regression model
observation(param, state, t) = product_distribution(Poisson.(linkinv.(param.link[1], param.Z[1][t] * state)))
condMean(param, state, t) = linkinv.(param.link[1], param.Z[1][t] * state)
condCov(param, state, t) = diagm(linkinv.(param.link[1], param.Z[1][t] * state))
function FisherInfoPois(param, β, t)
    dμ = mueta.(param.link[1], param.X[1] * β)
    F = XDiagX(param.X[1], dμ .^ 2 ./ (linkinv.(param.link[1], param.X[1]*β)))
    return F
end

function FisherInfoLocalPois(param, β, t)
    dμ = mueta.(param.link[1], param.Z[1][t] * β)
    F = XDiagX(param.Z[1][t], dμ .^ 2 ./ (linkinv.(param.link[1], param.Z[1][t]*β)))
    return F
end

