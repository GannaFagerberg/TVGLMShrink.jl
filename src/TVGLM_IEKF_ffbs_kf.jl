### Without scaling
function FFBS_IEKF_transformed!(Draws,U,Y,A,B,condMoments::Function,condMeanJacobian::Function,param,
                                Σₙ,μ₀,Σ₀,maxIter;tol=1e-2,filter_output=false,sample_t0=true,nFailure=Ref(0))

    T = length(Y)
    n = length(μ₀)
    q = size(U, 2)

    staticA = ndims(A) != 3

    # Same convention as in the IPLF code.
    Bmat = B isa Number ? fill(float(B), n, q) : B

    # ----------------------------------------------------------
    # Filtering storage
    # ----------------------------------------------------------

    μ_filter = zeros(T, n)
    Σ_filter = zeros(n, n, T)

    μ_pred = zeros(T, n)
    Σ_pred = zeros(n, n, T)

    μ = deepcopy(μ₀)
    Σ = deepcopy(Σ₀)

    # ----------------------------------------------------------
    # Forward filtering
    # ----------------------------------------------------------

    for t in 1:T

        filter_result = try

            At = staticA ? A : (@view A[:, :, t])

            # Allow static covariance, vector of covariance objects,
            # or n×n×T covariance array.
            Σₙ_raw = if ndims(Σₙ) == 3
                @view Σₙ[:, :, t]
            elseif Σₙ isa AbstractVector
                Σₙ[t]
            else
                Σₙ
            end

            Σₙt = Hermitian(
                Matrix(Σₙ_raw) + eps(Float64) * I
            )

            # Control input always represented as vector.
            u = @view U[t, :]

            kalmanfilter_update_transformed_IEKF(μ,Σ,u,Y[t],At,Bmat,condMoments,condMeanJacobian,
                                                param,Σₙt,t,maxIter;tol=tol)

        catch err
            nFailure[] += 1
            @error("Sufficient-statistics IEKF failed at time $t",exception=(err, catch_backtrace()))
            return nothing
        end

        μ, Σ, μ̄, Σ̄ = filter_result
        μ_filter[t, :] .= μ
        Σ_filter[:, :, t] .= Σ
        μ_pred[t, :] .= μ̄
        Σ_pred[:, :, t] .= Σ̄
    end

    # ----------------------------------------------------------
    # Backward sampling
    # ----------------------------------------------------------
    SMCsamplers.BackwardSampling!(Draws,μ_filter,Σ_filter,μ_pred,Σ_pred,A,μ₀,Σ₀;sample_t0=sample_t0)
    if filter_output
        return μ_filter, Σ_filter
    end
    return nothing
end

### Scaled vesrion
function FFBS_IEKF_transformed_scaled!(Draws,U,Y,A,B,condMoments::Function,condMeanJacobian::Function,param,Σₙ,
                                        μ₀,Σ₀,maxIter,ScaleMat, Svec;tol=1e-2,filter_output=false,sample_t0=true,nFailure=Ref(0))

    T = length(Y)
    n = length(μ₀)
    q = size(U, 2)

    staticA = ndims(A) != 3

    # Same convention as in the IPLF code.
    Bmat = B isa Number ? fill(float(B), n, q) : B

    # ----------------------------------------------------------
    # Filtering storage
    # ----------------------------------------------------------

    μ_filter = zeros(T, n)
    Σ_filter = zeros(n, n, T)

    μ_pred = zeros(T, n)
    Σ_pred = zeros(n, n, T)

    μ = deepcopy(μ₀)
    Σ = deepcopy(Σ₀)

    # ----------------------------------------------------------
    # Forward filtering
    # ----------------------------------------------------------

    for t in 1:T

        filter_result = try

            S = ScaleMat(param, μ, t)
            Svec[:, :, t] .= S

            At = staticA ? A : (@view A[:, :, t])

            # Allow static covariance, vector of covariance objects,

            Σₙ_raw = if ndims(Σₙ) == 3
                    S*(@view Σₙ[:, :, t])*S
            elseif Σₙ isa AbstractVector
                S*Σₙ[t]*S
            else
                S*Σₙ*S
            end

            Σₙt = Hermitian(
                Matrix(Σₙ_raw) + eps(Float64) * I
            )

            # Control input always represented as vector.
            u = @view U[t, :]

            kalmanfilter_update_transformed_IEKF(μ,Σ,u,Y[t],At,Bmat,condMoments,condMeanJacobian,param,
                                                    Σₙt,t,maxIter;tol=tol)

        catch err
            nFailure[] += 1
            @error("Sufficient-statistics IEKF failed at time $t",exception=(err, catch_backtrace()))
            return nothing
        end

        μ, Σ, μ̄, Σ̄ = filter_result

        μ_filter[t, :] .= μ
        Σ_filter[:, :, t] .= Σ
        μ_pred[t, :] .= μ̄
        Σ_pred[:, :, t] .= Σ̄
    end

    # ----------------------------------------------------------
    # Backward sampling
    # ----------------------------------------------------------
    SMCsamplers.BackwardSampling!(Draws,μ_filter,Σ_filter,μ_pred,Σ_pred,A,μ₀,Σ₀;sample_t0=sample_t0)
    if filter_output
        return μ_filter, Σ_filter
    end

    return nothing
end

### Bounded implementation
function kalmanfilter_update_transformed_IEKF(
    mu::AbstractVector,
    Omega::AbstractMatrix,
    u::AbstractVector,
    z::AbstractVector,
    A::AbstractMatrix,
    B::AbstractMatrix,
    condMoments,
    condJacobian,
    param,
    Sigma_n::AbstractMatrix,
    t,
    maxIter::Integer;
    tol::Real = 1e-2,
    covariance_floor::Real = 1e-8,
    return_diagnostics::Bool = false,
)

    maxIter >= 1 ||throw(ArgumentError("maxIter must be at least one."))

    # ==========================================================
    # Prior
    # ==========================================================

    mu_prior = A * mu + B * u
    Omega_prior = A * Omega * A' + Sigma_n

    # ==========================================================
    # IEKF iterations
    # ==========================================================

    mu_iter = copy(mu_prior)
    mean_distance = Inf
    iteration_used = 0

    # quantities from the last IEKF iteration
    H_k = nothing
    R_k = nothing
    S_k = nothing
    K_k = nothing

    constrained   = true
    precision_idx = length(mu_iter)
    delta_gamma   = 0.5 # Maximum precision-state change per iteration
    sd_gamma_max  = 0.5      # maximum posterior SD of precision state
    var_gamma_max = sd_gamma_max^2

    if maxIter == 1
        prior_sd_gamma = sqrt(Omega_prior[precision_idx, precision_idx])
        delta_gamma_eff = 1.0
    else
        delta_gamma_eff = delta_gamma
    end

    ###
    for iteration in 1:maxIter

        iteration_used = iteration
        h_k, R_k = condMoments(mu_iter, param, t)
        H_k = condJacobian(mu_iter, param, t)

        # Omega_prior remains fixed
        S_k = H_k * Omega_prior * H_k' + R_k
        K_k = Omega_prior * H_k' / S_k

        # Ordinary IEKF proposal
        mu_new = mu_prior + K_k * (z - h_k - H_k * (mu_prior - mu_iter))

        # ------------------------------------------------------
        # Restrict the proposed precision-state increment
        # ------------------------------------------------------
        d = mu_new - mu_iter
        b = clamp(d[precision_idx], -delta_gamma_eff, delta_gamma_eff) # clamp only precision
     
        if constrained 
            step_limited = b != d[precision_idx]

                if step_limited

                    # Required column of C = Omega_prior - K_k * H_k * Omega_prior
                    P_col = @view Omega_prior[:, precision_idx]
                    C_col = P_col - K_k * (H_k * P_col)
                    Crr   = C_col[precision_idx]
                    if !isfinite(Crr) || Crr <= 0
                        error("IEKF precision-state variance must be finite and positive.")
                    end
                    
                    # Adjust the full increment using its covariance with precision
                    adjustment = (b - d[precision_idx]) / Crr
                    d .+= adjustment .* C_col
                    d[precision_idx] = b
                    mu_new = mu_iter + d   
            end
        end

        # ------------------------------------------------------
        # Convergence
        # ------------------------------------------------------
        mean_distance = norm(mu_new - mu_iter)
        mu_iter = mu_new

        if !step_limited && mean_distance < tol
            break
        end
    end

    # ==========================================================
    # Covariance update
    #
    # Use H_k, R_k and K_k from the LAST IEKF iteration.
    # Do NOT recompute them after the iteration has stopped.
    # ==========================================================

    # Joseph
    #I_KH = I - K_k * H_k
    #Omega_updated = I_KH * Omega_prior * I_KH' + K_k * R_k * K_k'

    Omega_updated =(I - K_k * H_k) *Omega_prior
    Omega_updated = Matrix(Symmetric((Omega_updated + Omega_updated')/2))

   if constrained
        Omega_updated = constrain_precision_variance(Omega_updated,precision_idx,sd_gamma_max)
   end

    # ==========================================================
    # Return
    # ==========================================================

    if return_diagnostics

        return (
            mu = mu_iter,
            Omega = Omega_updated,
            mu_prior = mu_prior,
            Omega_prior = Omega_prior,

            diagnostics = (
                iteration = iteration_used,
                mean_distance = mean_distance,
                marginal_observation = copy(h_k),
                observation_covariance = copy(R_k),
                linearization = copy(H_k),
                innovation_covariance = copy(S_k),
                gain = copy(K_k),
            ),
        )
    end

    return (
        mu_iter,
        Omega_updated,
        mu_prior,
        Omega_prior,
    )
end

### The one I use wihtour bounds
function kalmanfilter_update_transformed_IEKF_ref(
    mu::AbstractVector,
    Omega::AbstractMatrix,
    u::AbstractVector,
    z::AbstractVector,
    A::AbstractMatrix,
    B::AbstractMatrix,
    condMoments,
    condJacobian,
    param,
    Sigma_n::AbstractMatrix,
    t,
    maxIter::Integer;
    tol::Real = 1e-2,
    covariance_floor::Real = 1e-8,
    return_diagnostics::Bool = false,
)

    maxIter >= 1 ||
        throw(ArgumentError("maxIter must be at least one."))

    # ==========================================================
    # Prior
    # ==========================================================

    mu_prior = A * mu + B * u
    Omega_prior = A * Omega * A' + Sigma_n

    # ==========================================================
    # IEKF iterations
    # ==========================================================

    mu_iter = copy(mu_prior)

    mean_distance = Inf
    iteration_used = 0

    # quantities from the last IEKF iteration
    H_k = nothing
    R_k = nothing
    S_k = nothing
    K_k = nothing

    for iteration in 1:maxIter

        iteration_used = iteration

        # ------------------------------------------------------
        # Conditional moments at current linearization point
        # ------------------------------------------------------

        h_k, R_k =condMoments(mu_iter,param,t)

        # ------------------------------------------------------
        # Jacobian at current linearization point
        # ------------------------------------------------------

        H_k =condJacobian(mu_iter,param,t)

        # ------------------------------------------------------
        # Kalman gain
        #
        # Omega_prior remains FIXED during IEKF iterations
        # ------------------------------------------------------

        S_k =H_k * Omega_prior * H_k' +R_k
        K_k =Omega_prior * H_k' / S_k

        # ------------------------------------------------------
        # IEKF mean update
        # ------------------------------------------------------

        mu_new = mu_prior +K_k * (z -h_k -H_k * (mu_prior - mu_iter))

        # OBS! Project precision state back to its admissible domain
    
        #idx = [3]
        #mu_new[idx] .= max.(mu_new[idx], -3)

        # ------------------------------------------------------
        # Convergence
        # ------------------------------------------------------
        mean_distance = norm(mu_new - mu_iter)
        mu_iter       = mu_new

        mean_distance < tol &&
            break
    end

    # ==========================================================
    # Covariance update
    #
    # Use H_k, R_k and K_k from the LAST IEKF iteration.
    # Do NOT recompute them after the iteration has stopped.
    # ==========================================================

    Omega_updated =(I - K_k * H_k) *Omega_prior
    # Joseph
    #I_KH = I - K_k * H_k
    #Omega_updated = I_KH * Omega_prior * I_KH' + K_k * R_k * K_k'
    Omega_updated = Matrix(Symmetric((Omega_updated + Omega_updated')/2))

    # ==========================================================
    # Return
    # ==========================================================

    if return_diagnostics

        return (
            mu = mu_iter,
            Omega = Omega_updated,
            mu_prior = mu_prior,
            Omega_prior = Omega_prior,

            diagnostics = (
                iteration = iteration_used,
                mean_distance = mean_distance,
                marginal_observation = copy(h_k),
                observation_covariance = copy(R_k),
                linearization = copy(H_k),
                innovation_covariance = copy(S_k),
                gain = copy(K_k),
            ),
        )
    end

    return (
        mu_iter,
        Omega_updated,
        mu_prior,
        Omega_prior,
    )
end

###################
### Beta moments
###################

function BetaSuffStatsShapes(state, param, t;
    mean_boundary=1e-12,
    min_concentration=1e-10,
    shape_floor=1e-6
)

    idxμ = param.Zidx[1]
    idxκ = param.Zidx[2]

    Zμ = param.Z[1][t]
    Zκ = param.Z[2][t]

    ημ = Zμ * state[idxμ]
    ηκ = Zκ * state[idxκ]

    μ_raw = linkinv.(Ref(param.link[1]), ημ)
    κ_raw = linkinv.(Ref(param.link[2]), ηκ)

    μ = clamp.(μ_raw, mean_boundary, 1.0 - mean_boundary)
    κ = max.(κ_raw, min_concentration)

    α_raw = μ .* κ
    β_raw = (1.0 .- μ) .* κ

    α = max.(α_raw, shape_floor)
    β = max.(β_raw, shape_floor)

    κ_eff = α .+ β

    return (
        Zμ=Zμ,
        Zκ=Zκ,
        ημ=ημ,
        ηκ=ηκ,
        μ=μ,
        κ=κ,
        α=α,
        β=β,
        κ_eff=κ_eff,
        α_raw=α_raw,
        β_raw=β_raw
    )
end



function BetaSuffStatsCondMoments(state, param, t)

    s = BetaSuffStatsShapes(state, param, t)

    α = s.α
    β = s.β
    κ = s.κ_eff

    g = length(α)

    h = zeros(2g)
    R = zeros(2g, 2g)

    for i in 1:g

        j = g + i

        ψ1α = trigamma(α[i])
        ψ1β = trigamma(β[i])
        ψ1κ = trigamma(κ[i])

        # same ordering as BetaSuffStatsGrouped()
        h[i] =digamma(α[i]) - digamma(κ[i])
        h[j] =digamma(β[i]) - digamma(κ[i])

        R[i, i] = ψ1α - ψ1κ
        R[j, j] = ψ1β - ψ1κ

        R[i, j] = -ψ1κ
        R[j, i] = -ψ1κ
    end

    return h, R
end

function BetaSuffStatsJacobian(state, param, t)

    s = BetaSuffStatsShapes(state, param, t)

    Zμ = s.Zμ
    Zκ = s.Zκ

    ημ = s.ημ
    ηκ = s.ηκ

    μ = s.μ
    κ = s.κ

    α = s.α
    β = s.β
    κeff = s.κ_eff

    α_raw = s.α_raw
    β_raw = s.β_raw

    idxμ = param.Zidx[1]
    idxκ = param.Zidx[2]

    n = length(state)
    g = length(μ)

    H = zeros(2g, n)

    dμ = mueta.(Ref(param.link[1]), ημ)
    dκ = mueta.(Ref(param.link[2]), ηκ)

    for i in 1:g

        j = g + i

        # ------------------------------------------------------
        # Derivatives before shape floors
        # ------------------------------------------------------

        dα_dημ =κ[i] * dμ[i]
        dα_dηκ =μ[i] * dκ[i]
        dβ_dημ =-κ[i] * dμ[i]
        dβ_dηκ =(1.0 - μ[i]) * dκ[i]

        # ------------------------------------------------------
        # Shape floors:
        #
        # α = max(α_raw, shape_floor)
        # β = max(β_raw, shape_floor)
        #
        # derivative is zero when floor is active
        # ------------------------------------------------------

        #if α_raw[i] <= α[i] && α_raw[i] != α[i]
            #dα_dημ = 0.0
            #dα_dηκ = 0.0
        #end

        #if β_raw[i] <= β[i] && β_raw[i] != β[i]
            #dβ_dημ = 0.0
            #dβ_dηκ = 0.0
        #end

        # κeff = α + β
        dκeff_dημ =dα_dημ + dβ_dημ
        dκeff_dηκ =dα_dηκ + dβ_dηκ

        ψ1α = trigamma(α[i])
        ψ1β = trigamma(β[i])
        ψ1κ = trigamma(κeff[i])

        # ------------------------------------------------------
        # h1 = ψ(α) - ψ(κeff)
        # ------------------------------------------------------

        dh1_dημ =ψ1α * dα_dημ -ψ1κ * dκeff_dημ
        dh1_dηκ =ψ1α * dα_dηκ -ψ1κ * dκeff_dηκ

        # ------------------------------------------------------
        # h2 = ψ(β) - ψ(κeff)
        # ------------------------------------------------------

        dh2_dημ =ψ1β * dβ_dημ -ψ1κ * dκeff_dημ
        dh2_dηκ =ψ1β * dβ_dηκ -ψ1κ * dκeff_dηκ

        # ------------------------------------------------------
        # Chain rule through regression design
        # ------------------------------------------------------

        H[i, idxμ] .=dh1_dημ .* Zμ[i, :]
        H[i, idxκ] .=dh1_dηκ .* Zκ[i, :]
        H[j, idxμ] .=dh2_dημ .* Zμ[i, :]
        H[j, idxκ] .=dh2_dηκ .* Zκ[i, :]
    end

    return H
end


##############################
# Beta single suff statistics
##############################

function BetaSingleSuffStatsCondMoments(
    state,
    param,
    t,
    stat::Symbol
)

    stat in (:logy, :log1my) ||
        throw(ArgumentError("stat must be :logy or :log1my"))

    s = BetaSuffStatsShapes(state, param, t)

    α = s.α
    β = s.β
    κeff = s.κ_eff

    g = length(α)

    h = zeros(g)
    R = zeros(g, g)

    for i in 1:g

        ψ1κ = trigamma(κeff[i])

        if stat === :logy

            h[i] =
                digamma(α[i]) -
                digamma(κeff[i])

            R[i, i] =
                trigamma(α[i]) -
                ψ1κ

        else  # :log1my

            h[i] =
                digamma(β[i]) -
                digamma(κeff[i])

            R[i, i] =
                trigamma(β[i]) -
                ψ1κ
        end
    end

    return h, R
end

function BetaSingleSuffStatsJacobian(
    state,
    param,
    t,
    stat::Symbol
)

    stat in (:logy, :log1my) ||
        throw(ArgumentError("stat must be :logy or :log1my"))

    s = BetaSuffStatsShapes(state, param, t)

    Zμ = s.Zμ
    Zκ = s.Zκ

    ημ = s.ημ
    ηκ = s.ηκ

    μ = s.μ
    κ = s.κ

    α = s.α
    β = s.β
    κeff = s.κ_eff

    idxμ = param.Zidx[1]
    idxκ = param.Zidx[2]

    n = length(state)
    g = length(μ)

    H = zeros(g, n)

    # Derivatives of underlying links
    dμ = mueta.(Ref(param.link[1]), ημ)
    dκ = mueta.(Ref(param.link[2]), ηκ)

    for i in 1:g

        # ------------------------------------------------------
        # Underlying Beta model:
        #
        # α = μκ
        # β = (1-μ)κ
        #
        # Numerical floors are NOT differentiated.
        # ------------------------------------------------------

        dα_dημ =  κ[i] * dμ[i]
        dα_dηκ =  μ[i] * dκ[i]

        dβ_dημ = -κ[i] * dμ[i]
        dβ_dηκ = (1.0 - μ[i]) * dκ[i]

        # κeff = α + β
        dκeff_dημ =
            dα_dημ + dβ_dημ

        dκeff_dηκ =
            dα_dηκ + dβ_dηκ

        ψ1κ = trigamma(κeff[i])

        if stat === :logy

            # h = ψ(α) - ψ(κeff)

            ψ1α = trigamma(α[i])

            dh_dημ =
                ψ1α * dα_dημ -
                ψ1κ * dκeff_dημ

            dh_dηκ =
                ψ1α * dα_dηκ -
                ψ1κ * dκeff_dηκ

        else  # :log1my

            # h = ψ(β) - ψ(κeff)

            ψ1β = trigamma(β[i])

            dh_dημ =
                ψ1β * dβ_dημ -
                ψ1κ * dκeff_dημ

            dh_dηκ =
                ψ1β * dβ_dηκ -
                ψ1κ * dκeff_dηκ
        end

        # Chain rule to state vector
        H[i, idxμ] .=
            dh_dημ .* Zμ[i, :]

        H[i, idxκ] .=
            dh_dηκ .* Zκ[i, :]
    end

    return H
end

BetaLogYCondMoments(state, param, t) =
    BetaSingleSuffStatsCondMoments(
        state, param, t, :logy
    )

BetaLogYJacobian(state, param, t) =
    BetaSingleSuffStatsJacobian(
        state, param, t, :logy
    )


BetaLog1mYCondMoments(state, param, t) =
    BetaSingleSuffStatsCondMoments(
        state, param, t, :log1my
    )

BetaLog1mYJacobian(state, param, t) =
    BetaSingleSuffStatsJacobian(
        state, param, t, :log1my
    )

##############################
# Constrain precsion variance
##############################
### If we decide to constrain the spread in the variance
function constrain_precision_variance(
    Omega_updated::AbstractMatrix,
    precision_idx::Int,
    sd_gamma_max::Real
)

    Omega = Matrix(Omega_updated)

    var_gamma_max = sd_gamma_max^2
    Orr = Omega[precision_idx, precision_idx]

    if !isfinite(Orr) || Orr <= 0
        error("IEKF precision-state variance must be finite and positive.")
    end

    if Orr > var_gamma_max

        # Required rescaling of the precision-state standard deviation
        s = sqrt(var_gamma_max / Orr)

        # Scale corresponding row and column
        Omega[precision_idx, :] .*= s
        Omega[:, precision_idx] .*= s

        # Restore exact symmetry numerically
        Omega = Matrix(Symmetric((Omega + Omega') / 2))
    end

    return Omega
end
