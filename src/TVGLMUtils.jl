using LinearAlgebra

"""
    simulateAR(T, ϕ, σₑ, μ=0)

Simulate an AR(p) process of length T with parameters ϕ, noise std σₑ and mean μ.
- T is the length of the time series
- ϕ is a vector of length p with the AR coefficients
- σₑ is the standard deviation of the noise
- μ is the mean of the process
"""
function simulateAR(T, ϕ, σₑ, μ=0)
    p = length(ϕ)
    x = zeros(2 * T)
    x[1:p] .= μ
    for t in (p+1):(2*T)
        x[t] = μ + ϕ ⋅ (reverse(x[t-p:t-1]) .- μ) + σₑ * randn()
    end
    return x[(T+1):end]
end

"""
    simulateVAR(T, Φ, Σₑ, μ)
    
Simulate a VAR(p) process of length T with VAR parameters Φ, noise covariance Σₑ and mean μ.
- T is the length of the time series
- Φ is a vector of length p with the VAR coefficients (each element is a matrix)
- Σₑ is the covariance matrix of the noise
- μ is the mean vector of the process
"""
function simulateVAR(T, Φ, Σₑ, μ)
    p = length(Φ)
    d = size(μ, 1)
    X = zeros(2 * T, d)
    X[1:p, :] = μ
    L = sqrt(Σₑ)
    for t in (p+1):(2*T)
        xₜ = copy(μ)
        for i in 1:p
            xₜ += Φ[i] * (X[t-i, :] - μ)
        end
        xₜ += L * randn(d)
        X[t, :] = xₜ
    end
    return X[(T+1):end, :]
end

""" 
    interpParam2Obs(θ, groupSizes=nothing, dateVec=nothing; sample_t0=false, 
        output_t0=false, interpMethod=:constant) 

Interpolation from parameter time scale to observation time scale. 

- θ is Tgroup x p x nQuant, where Tgroup includes t=0 if sample_t0=true.
- sample_t0 = true if parameter at t=0 is sampled.
- output_t0 = true if the interpolated parameter at t=0 should be included in the output.
- groupSizes is a vector of length Tgroup - sample_t0, with the number of obs in each group.
"""
function interpParam2Obs(θ, groupSizes=nothing, dateVec=nothing; sample_t0=false,
    output_t0=false, interpMethod=:constant)

    Tgroup, p, nQuant = size(θ)
    if isnothing(groupSizes)
        groupSizes = ones(Int, Tgroup - sample_t0)
    end

    if sample_t0
        timeVec = 0:sum(groupSizes)
        groupSizes = [1; groupSizes]
    else
        timeVec = 1:sum(groupSizes)
    end

    # Set up date vectors
    if !isnothing(dateVec) && sample_t0
        timeUnit = dateVec[2] - dateVec[1]
        dateVec = [dateVec[1] - timeUnit; dateVec]
    end
    if isnothing(dateVec)
        dateVec = timeVec
    end
    if length(dateVec) != length(timeVec)
        error("Length of dateVec must match the time dimension of the data")
    end

    # Duplicate the group-level parameters to observation-level
    Tobs = sum(groupSizes)
    θobs = Array{eltype(θ)}(undef, Tobs, p, nQuant)
    idx = 1
    for t_group in 1:Tgroup
        for i in 1:groupSizes[t_group]
            for q in 1:nQuant
                if interpMethod == :constant
                    θobs[idx, :, q] .= θ[t_group, :, q]
                elseif interpMethod == :linear
                    if t_group == 1 || t_group == Tgroup
                        θobs[idx, :, q] .= θ[t_group, :, q]
                    else
                        weight = (i - 1) / groupSizes[t_group]
                        θobs[idx, :, q] .= (1 - weight) * θ[t_group, :, q] +
                                           weight * θ[t_group+1, :, q]
                    end
                else
                    error("Unsupported interpolation method: $interpMethod")
                end
            end
            idx += 1
        end
    end
    if sample_t0 && !output_t0
        θobs = θobs[2:end, :, :]
        dateVec = dateVec[2:end]
    end

    return θobs, dateVec
end

function scalingLabel(scaling)
    return scaling == :full ? "F" : (scaling == :diagonal ? "D" : "N")
end

""" 
    XDiagX(X, d)

Efficiently compute X' * Diagonal(d) * X without forming the diagonal matrix.
- X is n x p
- d is a vector of length n
"""
function XDiagX(X, d)
    n, p = size(X)
    XDX = zeros(eltype(X), p, p)
    @inbounds for j in 1:p
        for i in 1:j
            value = zero(eltype(XDX))
            for k in 1:n
                value += d[k] * X[k, i] * X[k, j]
            end
            XDX[i, j] = value
            XDX[j, i] = value
        end
    end
    return XDX
end

""" 
    DiagXDiagX(X, d)

Efficiently compute the diagonal of X' * Diagonal(d) * X without forming the diagonal matrix.
- X is n x p
- d is a vector of length n
"""
function DiagXDiagX(X, d)
    n, p = size(X)
    S = promote_type(eltype(X), eltype(d))
    d_out = zeros(S, p)
    @inbounds for j in 1:p
        value = zero(S)
        for k in 1:n
            value += d[k] * X[k, j]^2
        end
        d_out[j] = value
    end
    return d_out
end

""" 
    XDiagZ(X, d)

Efficiently compute X' * Diagonal(d) * Z without forming the diagonal matrix.
- X is n x p
- d is a vector of length n
- Z is n x q
"""
function XDiagZ(X, d, Z)
    n, p = size(X)
    _, q = size(Z)
    S = promote_type(eltype(X), eltype(d), eltype(Z))
    XDZ = zeros(S, p, q)
    @inbounds for j in 1:p
        for l in 1:q
            value = zero(S)
            for k in 1:n
                value += d[k] * X[k, j] * Z[k, l]
            end
            XDZ[j, l] = value
        end
    end
    return XDZ
end

"""
    fisher_scaling_diagonal_first(X, d)

Compute the full Fisher information scaling matrix S^(-1/2) where S = X' * Diagonal(d) * X.
- X is n x p
- d is a vector of length n
"""
function fisher_scaling_diagonal_first_global(X, d)
    d_S = DiagXDiagX(X, d)                # diag(S = X'DX)
    return 1 ./ sqrt.(d_S)                # diag(S)^(-1/2)
end

"""
    fisher_scaling_diagonal_first_local(xₜ, d)
Compute the local Fisher information scaling matrix for a single observation xₜ.
- xₜ is a vector of length p
- d is a scalar weight for the t:th observation
"""
function fisher_scaling_diagonal_first_local(xₜ, d)
    return @. 1 / sqrt(d * (xₜ^2))
end


"""
    densityScores(θpost, θtrue)

Compute density scores (CRPS, energy score, variogram score) for posterior samples `θpost` against true parameters `θtrue`.
- `θpost` is T x p x nIter matrix of posterior samples of the parameters at each time point.
- `θtrue` is T x p matrix of the true parameter values at each time point.
Returns:
- `CRPS`: T x p matrix of CRPS scores for each parameter at each time point.
- `energyScore`: vector of length T with the energy score for the joint distribution of parameters at each time point.
- `variogramScore`: vector of length T with the variogram score for the joint distribution of parameters at each time point.
"""
function densityScores(θpost, θtrue)

    T, p, nIter = size(θpost)
    CRPS = zeros(T, p)
    energyScore = zeros(T)
    variogramScore = zeros(T)
    for t in 1:T
        for j in 1:p
            CRPS[t, j] = crps(θpost[t, j, :], θtrue[t, j])
        end
        energyScore[t] = energy_score(θpost[t, :, :]', θtrue[t, :])
        variogramScore[t] = variogram_score(θpost[t, :, :]', θtrue[t, :])
    end
    return CRPS, energyScore, variogramScore
end