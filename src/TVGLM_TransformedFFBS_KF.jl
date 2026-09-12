mutable struct IPLFWorkspace{T}
    spread::Matrix{T}
    sigma_points::Matrix{T}
    conditional_means::Matrix{T}
    conditional_covariance::Matrix{T}
    z_mean::Vector{T}
    mean_conditional_covariance::Matrix{T}
    centered_states::Matrix{T}
    centered_means::Matrix{T}
    P_xz::Matrix{T}
    P_z::Matrix{T}
end

function IPLFWorkspace(
    ::Type{T},
    n::Int,
    m::Int
) where {T}

    nSigma = 2n + 1

    IPLFWorkspace(
        zeros(T, n, n),          # spread
        zeros(T, n, nSigma),     # sigma_points
        zeros(T, m, nSigma),     # conditional_means
        zeros(T, m, m),          # conditional_covariance   NEW
        zeros(T, m),             # z_mean
        zeros(T, m, m),          # mean_conditional_covariance
        zeros(T, n, nSigma),     # centered_states
        zeros(T, m, nSigma),     # centered_means
        zeros(T, n, m),          # P_xz
        zeros(T, m, m),          # P_z
    )
end


function FFBS_SLR_transformed!(
    Draws,
    U,
    Y,
    A,
    B,
    condMoments::Function,
    param,
    Σₙ,
    μ₀,
    Σ₀,
    maxIter,
    ws;
    α=1,
    β=0,
    κ=0,
    filter_output=false,
    sample_t0=true,
    nFailure=Ref(0)
)

    T = length(Y)
    n = length(μ₀)
    q = size(U, 2)

    staticA = ndims(A) != 3

    # The new update expects B to be a matrix and u to be a vector.
    # In GibbsTVGLM, B is currently the scalar 0.0.
    Bmat = B isa Number ? fill(float(B), n, q) : B

    # ----------------------------------------------------------
    # Unscented-transform weights
    # ----------------------------------------------------------

    λ = α^2 * (n + κ) - n

    ωₘ = [
        λ / (n + λ)
        ones(2 * n) ./ (2 * (n + λ))
    ]

    ωₛ = [
        λ / (n + λ) + (1 - α^2 + β)
        ωₘ[2:end]
    ]

    γ = sqrt(n + λ)

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

            # Support a static matrix, a vector of PDMat objects
            Σₙ_raw = if ndims(Σₙ) == 3
                @view Σₙ[:, :, t]
            elseif Σₙ isa AbstractVector
                Σₙ[t]
            else
                Σₙ
            end

            Σₙt = Hermitian(Matrix(Σₙ_raw) + eps(Float64) * I)
            u = @view U[t, :] ## keep the control input as a vector.
            kalmanfilter_update_transformed_IPLF(μ,Σ,u,Y[t],At,Bmat,condMoments,param,Σₙt,t,maxIter,γ,ωₘ,ωₛ,ws)

        catch err
            nFailure[] += 1
            @error("Sufficient-statistics IPLF failed at time $t",exception=(err, catch_backtrace()))
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

    Draws_old = copy(Draws)

        try
            SMCsamplers.BackwardSampling!(
                Draws,
                μ_filter,
                Σ_filter,
                μ_pred,
                Σ_pred,
                A,
                μ₀,
                Σ₀;
                sample_t0 = sample_t0
            )

        catch err
            if err isa LinearAlgebra.PosDefException ||
            err isa LinearAlgebra.SingularException

                Draws .= Draws_old
                nFailure[] += 1

                return nothing
            else
                rethrow(err)
            end
    end

    if filter_output
        return μ_filter, Σ_filter
    end

    return nothing
end




function FFBS_SLR_transformed_ref!(
    Draws,
    U,
    Y,
    A,
    B,
    condMoments::Function,
    param,
    Σₙ,
    μ₀,
    Σ₀,
    maxIter,
    ws;
    α=1,
    β=0,
    κ=0,
    filter_output=false,
    sample_t0=true,
    nFailure=Ref(0)
)

    T = length(Y)
    n = length(μ₀)
    q = size(U, 2)

    staticA = ndims(A) != 3

    # The new update expects B to be a matrix and u to be a vector.
    # In GibbsTVGLM, B is currently the scalar 0.0.
    Bmat = B isa Number ? fill(float(B), n, q) : B

    # ----------------------------------------------------------
    # Unscented-transform weights
    # ----------------------------------------------------------

    λ = α^2 * (n + κ) - n

    ωₘ = [
        λ / (n + λ)
        ones(2 * n) ./ (2 * (n + λ))
    ]

    ωₛ = [
        λ / (n + λ) + (1 - α^2 + β)
        ωₘ[2:end]
    ]

    γ = sqrt(n + λ)

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

        # Support a static matrix, a vector of PDMat objects,
        # or an n×n×T covariance array.
        Σₙ_raw = if ndims(Σₙ) == 3
            @view Σₙ[:, :, t]
        elseif Σₙ isa AbstractVector
            Σₙ[t]
        else
            Σₙ
        end

        Σₙt = Hermitian(Matrix(Σₙ_raw) + eps(Float64) * I)

        # Always keep the control input as a vector.
        u = @view U[t, :]

           kalmanfilter_update_transformed_IPLF(
                μ,
                Σ,
                u,
                Y[t],          # transformed sufficient-statistic observation
                At,
                Bmat,
                condMoments,
                param,
                Σₙt,
                t,
                maxIter,
                γ,
                ωₘ,
                ωₛ,
                ws
            )

        catch err

            nFailure[] += 1

            @error(
                "Sufficient-statistics IPLF failed at time $t",
                exception=(err, catch_backtrace())
            )

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

    SMCsamplers.BackwardSampling!(Draws, μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀; sample_t0=sample_t0)

    if filter_output
        return μ_filter, Σ_filter
    end

    return nothing
end


# Convert a scalar or grouped value into a vector
function _beta_vector(value, name::AbstractString)

    if value isa Real
        return [float(value)]
    elseif value isa AbstractArray
        isempty(value) && throw(ArgumentError("$name cannot be empty."))
        return vec(float.(collect(value)))
    else
        throw(ArgumentError(
            "$name must be a scalar or an array; got $(typeof(value))."
        ))
    end
end



### More allocation free
# Take a group of beta observations
# Transforms very observation into the two sufficient statistics of the Beta exponential family
# Keeps the two sufficient statistics for every observation in the group
# Stores all the log(y) first and then log(1-y)
# Lopps through every observation



function kalmanfilter_update_transformed_IPLF(
    mu::AbstractVector,
    Omega::AbstractMatrix,
    u::AbstractVector,
    z::AbstractVector,
    A::AbstractMatrix,
    B::AbstractMatrix,
    condMoments,
    param,
    Sigma_n::AbstractMatrix,
    t,
    maxIter::Integer,
    gamma,
    w_mean::AbstractVector,
    w_cov::AbstractVector,
    ws;
    tol::Real = 1e-3,
    covariance_floor::Real = 1e-8,
    return_diagnostics::Bool = false,
)

    maxIter >= 1 || throw(ArgumentError("maxIter must be at least one."))

    # ----------------------------------------------------------
    # Prior propagation
    # ----------------------------------------------------------
    mu_prior = A * mu .+ B * u
    Omega_prior = A * Omega * A' + Sigma_n

    mu_iter = copy(mu_prior)
    Omega_iter = copy(Omega_prior)

    # z is already an AbstractVector; no need to collect/copy it
    z_vec = z
    observation_dimension = length(z_vec)
    last_diagnostics = nothing

    # ----------------------------------------------------------
    # IPLF iterations
    # ----------------------------------------------------------
    constrained   = true
    weighted_mean = true
    Joseph        = false

    precision_idx = 3
    #delta_gamma   = 1.0
    delta_gamma   = 0.5
    #precision_link = param.link[2]
    sd_gamma_max = 1.0      # maximum posterior SD of precision state
    var_gamma_max = sd_gamma_max^2

    # Only relevant for the hard positive link
    #if  precision_link isa PositiveHardLink
       # precision_link = param.link[2]
       # hard_constraint = precision_link isa PositiveHardLink 
       # precision_floor = hard_constraint ? precision_link.floor : -Inf
       # boundary_eps    = 1e-10
    #end

    n_state = length(mu_prior)
    I_n = Matrix{eltype(Omega_prior)}(I, n_state, n_state)

    for iteration in 1:maxIter

        L = cholesky(Hermitian(Omega_iter)).L

        # Sigma points
        spread = L * gamma
        sigma_points = hcat(mu_iter,mu_iter .+ spread,mu_iter .- spread)
        number_of_points = size(sigma_points, 2)

        length(w_mean) == number_of_points ||
            throw(DimensionMismatch(
                "length(w_mean)=$(length(w_mean)), but there are " *
                "$number_of_points sigma points.",
            ))

        length(w_cov) == number_of_points ||
            throw(DimensionMismatch(
                "length(w_cov)=$(length(w_cov)), but there are " *
                "$number_of_points sigma points.",
            ))

        # ======================================================
        # Conditional moments
        #
        # STEP 2:
        #   - Store all conditional means because they are needed
        #     later for Cov[x,z] and Var(E[z|x]).
        #
        #   - Do NOT store all conditional covariance matrices.
        #     Accumulate
        #
        #       E[Var(z|x)] ≈ Σ_j w_mean[j] R_j
        #
        #     immediately.
        # ======================================================

        # One column per sigma point
        conditional_means     = ws.conditional_means
        covariance_j          = ws.conditional_covariance
        z_mean                = ws.z_mean
        mean_conditional_covariance = ws.mean_conditional_covariance
        if weighted_mean

            fill!(z_mean, 0)
            fill!(mean_conditional_covariance, 0)

            for j in 1:number_of_points

                mean_j = @view conditional_means[:, j]
                condMoments(mean_j,covariance_j,param,@view(sigma_points[:, j]),t,)

                # Reference μₖ⁺
                z_mean .+= w_mean[j] .* mean_j

                # Reference E[R(x)] term
                mean_conditional_covariance .+=w_cov[j] .* covariance_j
            end

        else

            fill!(mean_conditional_covariance, 0)

        for j in 1:number_of_points
            mean_j = @view conditional_means[:, j]
            condMoments(mean_j,covariance_j,param,@view(sigma_points[:, j]),t)
            mean_conditional_covariance .+=w_cov[j] .* covariance_j
        end

            # h(mu_iter)
            z_mean .= @view conditional_means[:, 1]
            
        end

        # ======================================================
        # Cross covariance Cov[x,z]
        # ======================================================
        centered_states = ws.centered_states
        centered_means  = ws.centered_means
        centered_states .= sigma_points .- mu_iter
        centered_means  .= conditional_means .- z_mean

        P_xz = ws.P_xz
        fill!(P_xz, 0)
        @inbounds for j in 1:number_of_points
            wj = w_cov[j]
            for b in axes(centered_means, 1)
                mb = centered_means[b, j]
                for a in axes(centered_states, 1)
                    P_xz[a, b] += wj * centered_states[a, j] * mb
                end
            end
        end

        # ======================================================
        # Total observation covariance
        # Var[z] =E[Var(z|x)] + Var(E[z|x])
        # ======================================================

        P_z = ws.P_z
        P_z .= mean_conditional_covariance ## E[Var(z|x)]

        # + Var(E[z|x])
        for j in 1:number_of_points
            wj = w_cov[j]
            @inbounds for b in axes(centered_means, 1)
                db = centered_means[b, j]
                for a in axes(centered_means, 1)
                    P_z[a, b] += wj * centered_means[a, j] * db
                end
            end
        end

        #P_z = _make_spd(P_z; relative_floor = covariance_floor)

        # ======================================================
        # Statistical linear regression
        #
        # z_t ≈ H_k*x_t + b_k + e_k
        # ======================================================

        H_k = P_xz' / Omega_iter
        b_k = z_mean - H_k * mu_iter
 
        # ----------------------------------------------------------
        #   Numerically stable SLR residual covariance:
        #   R_k = E[Var(z|x)]+ E[(m(x)-z_mean-H_k(x-mu_iter))+(m(x)-z_mean-H_k(x-mu_iter))']
        # ----------------------------------------------------------

        R_k = copy(mean_conditional_covariance)
        for j in 1:number_of_points
            residual_j =@view(centered_means[:, j]) - H_k * @view(centered_states[:, j])
            R_k .+= w_cov[j] .* (residual_j * residual_j')
        end

        # ======================================================
        # Kalman update
        # ======================================================
        
        z_prior_mean = H_k * mu_prior + b_k
        innovation_covariance = H_k * Omega_prior * H_k' + R_k
        gain = Omega_prior * H_k' / innovation_covariance

        # Ordinary IPLF proposal
        mu_updated = mu_prior + gain * (z_vec - z_prior_mean)

        # ======================================================
        # Restrict precision-state increment
        # ======================================================
        
    
        if constrained 

            # ========================================================
            # 0. UNCONSTRAINED POSTERIOR COVARIANCE
            # ========================================================

                C_k = Omega_prior - gain * (H_k * Omega_prior)

                # Numerical symmetry
                C_k = (C_k + C_k') / 2

                if !all(isfinite, C_k)
                    error("IPLF posterior covariance contains non-finite values.")
                end

                # --------------------------------------------------------
                # Numerical PSD safeguard
                # --------------------------------------------------------
                # This is NOT the precision-variance cap.
                # It only repairs small numerical loss of positive definiteness.

                F = eigen(Symmetric(C_k))
                λmin = minimum(F.values)

                if λmin < covariance_floor

                    # Optional: distinguish a genuine breakdown from a tiny
                    # floating-point violation.
                    scale = max(opnorm(C_k, Inf), 1.0)
                    neg_tol = 1e-8 * scale

                    if λmin < -neg_tol
                        error(
                            "IPLF posterior covariance is genuinely indefinite: " *
                            "lambda_min = $λmin"
                        )
                    end

                    vals = max.(F.values, covariance_floor)

                    C_k = F.vectors * Diagonal(vals) * F.vectors'
                    C_k = (C_k + C_k') / 2
                end


                # ========================================================
                # 1. CONSTRAIN POSTERIOR MEAN / STEP
                # ========================================================

                if constrained

                    d = mu_updated - mu_iter

                    # Limit change in the precision state
                    b = clamp(
                        d[precision_idx],
                        -delta_gamma,
                        delta_gamma
                    )

                    step_limited = b != d[precision_idx]

                    if step_limited

                        # IMPORTANT:
                        # use the already computed/stabilized posterior covariance
                        C_col = @view C_k[:, precision_idx]
                        Crr   = C_k[precision_idx, precision_idx]

                        if !isfinite(Crr) || Crr <= 0
                            error(
                                "IPLF precision-state variance must be finite and positive."
                            )
                        end

                        # Conditional/projection correction
                        adjustment =
                            (b - d[precision_idx]) / Crr

                        d .+= adjustment .* C_col

                        # Impose requested precision increment exactly
                        d[precision_idx] = b

                        mu_updated = mu_iter + d
                    end
                end


                # ========================================================
                # 2. CONSTRAIN POSTERIOR COVARIANCE
                # ========================================================

                Crr = C_k[precision_idx, precision_idx]

                if !isfinite(Crr) || Crr <= 0
                    error(
                        "IPLF precision-state variance must be finite and positive."
                    )
                end

                if Crr > var_gamma_max

                    s = sqrt(var_gamma_max / Crr)

                    # C* = D C D
                    # Scale precision row and column together.
                    C_k[precision_idx, :] .*= s
                    C_k[:, precision_idx] .*= s

                    # Good to retain this
                    C_k .= (C_k + C_k') / 2
                end


        
        end # end constraint

        if Joseph
            #I_KH = I_n - gain * H_k
            #Omega_updated =I_KH * Omega_prior * I_KH' +gain * R_k * gain'
        else
             #Omega_updated = Omega_prior -gain * innovation_covariance * gain'
             Omega_updated = C_k
        end

        # ======================================================
        # IPLF convergence
        # ======================================================
        distance = gaussian_kld(mu_iter,Omega_iter,mu_updated,Omega_updated;relative_floor = covariance_floor)
        mu_iter = mu_updated
        Omega_iter = Omega_updated

        distance < tol && break
    end

    # ----------------------------------------------------------
    # Return
    # ----------------------------------------------------------

    if return_diagnostics

        return (
            mu = mu_iter,
            Omega = Omega_iter,
            mu_prior = mu_prior,
            Omega_prior = Omega_prior,
            diagnostics = last_diagnostics,
        )
    end

    return mu_iter, Omega_iter, mu_prior, Omega_prior
end



function kalmanfilter_update_transformed_IPLF_use_now(
    mu::AbstractVector,
    Omega::AbstractMatrix,
    u::AbstractVector,
    z::AbstractVector,
    A::AbstractMatrix,
    B::AbstractMatrix,
    condMoments,
    param,
    Sigma_n::AbstractMatrix,
    t,
    maxIter::Integer,
    gamma,
    w_mean::AbstractVector,
    w_cov::AbstractVector,
    ws;
    tol::Real = 1e-3,
    covariance_floor::Real = 1e-8,
    return_diagnostics::Bool = false,
)

    maxIter >= 1 || throw(ArgumentError("maxIter must be at least one."))

    # ----------------------------------------------------------
    # Prior propagation
    # ----------------------------------------------------------
    mu_prior = A * mu .+ B * u
    Omega_prior = A * Omega * A' + Sigma_n

    mu_iter = copy(mu_prior)
    Omega_iter = copy(Omega_prior)

    # z is already an AbstractVector; no need to collect/copy it
    z_vec = z
    observation_dimension = length(z_vec)
    last_diagnostics = nothing

    # ----------------------------------------------------------
    # IPLF iterations
    # ----------------------------------------------------------

    precision_idx = 3
    #delta_gamma   = 1.0
    delta_gamma   = 0.5
    #precision_link = param.link[2]
    sd_gamma_max = 1.0      # maximum posterior SD of precision state
    var_gamma_max = sd_gamma_max^2

    # Only relevant for the hard positive link
    #if  precision_link isa PositiveHardLink
       # precision_link = param.link[2]
       # hard_constraint = precision_link isa PositiveHardLink 
       # precision_floor = hard_constraint ? precision_link.floor : -Inf
       # boundary_eps    = 1e-10
    #end

    n_state = length(mu_prior)
    I_n = Matrix{eltype(Omega_prior)}(I, n_state, n_state)

    for iteration in 1:maxIter

        L = cholesky(Hermitian(Omega_iter)).L

        # Sigma points
        spread = L * gamma
        sigma_points = hcat(mu_iter,mu_iter .+ spread,mu_iter .- spread)
        number_of_points = size(sigma_points, 2)

        length(w_mean) == number_of_points ||
            throw(DimensionMismatch(
                "length(w_mean)=$(length(w_mean)), but there are " *
                "$number_of_points sigma points.",
            ))

        length(w_cov) == number_of_points ||
            throw(DimensionMismatch(
                "length(w_cov)=$(length(w_cov)), but there are " *
                "$number_of_points sigma points.",
            ))

        # ======================================================
        # Conditional moments
        #
        # STEP 2:
        #   - Store all conditional means because they are needed
        #     later for Cov[x,z] and Var(E[z|x]).
        #
        #   - Do NOT store all conditional covariance matrices.
        #     Accumulate
        #
        #       E[Var(z|x)] ≈ Σ_j w_mean[j] R_j
        #
        #     immediately.
        # ======================================================

        # One column per sigma point
        conditional_means     = ws.conditional_means
        covariance_j          = ws.conditional_covariance
        z_mean                = ws.z_mean
        mean_conditional_covariance = ws.mean_conditional_covariance

        weighted_mean = true

        if weighted_mean

        fill!(z_mean, 0)
        fill!(mean_conditional_covariance, 0)

            for j in 1:number_of_points

                mean_j = @view conditional_means[:, j]
                condMoments(mean_j,covariance_j,param,@view(sigma_points[:, j]),t,)

                # Reference μₖ⁺
                z_mean .+= w_mean[j] .* mean_j

                # Reference E[R(x)] term
                mean_conditional_covariance .+=w_cov[j] .* covariance_j
            end

        else

            fill!(mean_conditional_covariance, 0)

        for j in 1:number_of_points
            mean_j = @view conditional_means[:, j]
            condMoments(mean_j,covariance_j,param,@view(sigma_points[:, j]),t)
            mean_conditional_covariance .+=w_cov[j] .* covariance_j
        end

            # h(mu_iter)
            z_mean .= @view conditional_means[:, 1]
            
        end

        # ======================================================
        # Cross covariance Cov[x,z]
        # ======================================================
        centered_states = ws.centered_states
        centered_means  = ws.centered_means
        centered_states .= sigma_points .- mu_iter
        centered_means  .= conditional_means .- z_mean

        P_xz = ws.P_xz
        fill!(P_xz, 0)
        @inbounds for j in 1:number_of_points
            wj = w_cov[j]
            for b in axes(centered_means, 1)
                mb = centered_means[b, j]
                for a in axes(centered_states, 1)
                    P_xz[a, b] += wj * centered_states[a, j] * mb
                end
            end
        end

        # ======================================================
        # Total observation covariance
        # Var[z] =E[Var(z|x)] + Var(E[z|x])
        # ======================================================

        P_z = ws.P_z
        P_z .= mean_conditional_covariance ## E[Var(z|x)]

        # + Var(E[z|x])
        for j in 1:number_of_points
            wj = w_cov[j]
            @inbounds for b in axes(centered_means, 1)
                db = centered_means[b, j]
                for a in axes(centered_means, 1)
                    P_z[a, b] += wj * centered_means[a, j] * db
                end
            end
        end

        #P_z = _make_spd(P_z; relative_floor = covariance_floor)

        # ======================================================
        # Statistical linear regression
        #
        # z_t ≈ H_k*x_t + b_k + e_k
        # ======================================================

        H_k = P_xz' / Omega_iter
        b_k = z_mean - H_k * mu_iter
 
        # ----------------------------------------------------------
        #   Numerically stable SLR residual covariance:
        #   R_k = E[Var(z|x)]+ E[(m(x)-z_mean-H_k(x-mu_iter))+(m(x)-z_mean-H_k(x-mu_iter))']
        # ----------------------------------------------------------

        R_k = copy(mean_conditional_covariance)
        for j in 1:number_of_points
            residual_j =@view(centered_means[:, j]) - H_k * @view(centered_states[:, j])
            R_k .+= w_cov[j] .* (residual_j * residual_j')
        end

        # ======================================================
        # Kalman update
        # ======================================================
        
        z_prior_mean = H_k * mu_prior + b_k
        innovation_covariance = H_k * Omega_prior * H_k' + R_k
        gain = Omega_prior * H_k' / innovation_covariance

        # Ordinary IPLF proposal
        mu_updated = mu_prior + gain * (z_vec - z_prior_mean)

        # ======================================================
        # Restrict precision-state increment
        # ======================================================
        
        constrained = true

        if constrained 

            d = mu_updated - mu_iter
            b = clamp(d[precision_idx],-delta_gamma, delta_gamma)
           
            # Additional lower-bound constraint: ONLY PositiveHardLink
            
           #if precision_link isa PositiveHardLink
              #  lower_bound = precision_link.floor + boundary_eps
               # b = max(b, lower_bound - mu_iter[precision_idx])
            #end

            step_limited = b != d[precision_idx]

            if step_limited
                # Required column of the posterior covariance
                P_col = @view Omega_prior[:, precision_idx]

                # precision_idx-th column of the unconstrained posterior covariance
                C_col = P_col - gain * (H_k * P_col)
                Crr   = C_col[precision_idx]
                if !isfinite(Crr) || Crr <= 0
                    error("IPLF precision-state variance must be finite and positive.")
                end

                # Conditional/projection adjustment of the full state increment
                adjustment = (b - d[precision_idx]) / Crr
                d .+= adjustment .* C_col

                # Enforce constrained precision increment exactly
                d[precision_idx] = b
                mu_updated = mu_iter + d
            end

        # ========================================================
        # 2. CONSTRAIN POSTERIOR COVARIANCE
        # ========================================================

        # Unconstrained IPLF posterior covariance at iteration k
        C_k = Omega_prior - gain * (H_k * Omega_prior)
        C_k = (C_k + C_k') / 2

        Crr = C_k[precision_idx, precision_idx]

            if !isfinite(Crr) || Crr <= 0
                error(
                    "IPLF precision-state variance must be finite and positive."
                )
            end

            if Crr > var_gamma_max

                s = sqrt(var_gamma_max / Crr)

                # Scale row and column together so that
                # C[precision_idx, precision_idx] becomes var_gamma_max
                C_k[precision_idx, :] .*= s
                C_k[:, precision_idx] .*= s

                # Numerical symmetrization
                C_k .= (C_k + C_k') / 2
            end
        
        end # end constraint

        Joseph = false
        if Joseph
            #I_KH = I_n - gain * H_k
            #Omega_updated =I_KH * Omega_prior * I_KH' +gain * R_k * gain'
        else
             #Omega_updated = Omega_prior -gain * innovation_covariance * gain'
             Omega_updated = C_k
        end

        # ======================================================
        # IPLF convergence
        # ======================================================
        distance = gaussian_kld(mu_iter,Omega_iter,mu_updated,Omega_updated;relative_floor = covariance_floor)
        mu_iter = mu_updated
        Omega_iter = Omega_updated

        distance < tol && break
    end

    # ----------------------------------------------------------
    # Return
    # ----------------------------------------------------------

    if return_diagnostics

        return (
            mu = mu_iter,
            Omega = Omega_iter,
            mu_prior = mu_prior,
            Omega_prior = Omega_prior,
            diagnostics = last_diagnostics,
        )
    end

    return mu_iter, Omega_iter, mu_prior, Omega_prior
end



function kalmanfilter_update_transformed_IPLF_ref(
    mu::AbstractVector,
    Omega::AbstractMatrix,
    u::AbstractVector,
    z::AbstractVector,
    A::AbstractMatrix,
    B::AbstractMatrix,
    condMoments,
    param,
    Sigma_n::AbstractMatrix,
    t,
    maxIter::Integer,
    gamma,
    w_mean::AbstractVector,
    w_cov::AbstractVector,
    ws;
    tol::Real = 1e-3,
    covariance_floor::Real = 1e-8,
    return_diagnostics::Bool = false,
)

    maxIter >= 1 || throw(ArgumentError("maxIter must be at least one."))

    # ----------------------------------------------------------
    # Prior propagation
    # ----------------------------------------------------------

    mu_prior = A * mu .+ B * u
    Omega_prior = _make_spd(A * Omega * A' + Sigma_n;relative_floor = covariance_floor,)
    #Omega_prior = A * Omega * A' + Sigma_n

    mu_iter = copy(mu_prior)
    Omega_iter = copy(Omega_prior)

    # z is already an AbstractVector; no need to collect/copy it
    z_vec = z
    observation_dimension = length(z_vec)
    last_diagnostics = nothing

    # ----------------------------------------------------------
    # IPLF iterations
    # ----------------------------------------------------------

    for iteration in 1:maxIter

        Omega_iter = _make_spd(Omega_iter;relative_floor = covariance_floor,)
        #L = cholesky(Symmetric(Omega_iter)).L
        F_iter = cholesky(Symmetric(Omega_iter))
        L = F_iter.L

        # Sigma points
        spread = L * gamma
        sigma_points = hcat(mu_iter,mu_iter .+ spread,mu_iter .- spread)
        number_of_points = size(sigma_points, 2)

        length(w_mean) == number_of_points ||
            throw(DimensionMismatch(
                "length(w_mean)=$(length(w_mean)), but there are " *
                "$number_of_points sigma points.",
            ))

        length(w_cov) == number_of_points ||
            throw(DimensionMismatch(
                "length(w_cov)=$(length(w_cov)), but there are " *
                "$number_of_points sigma points.",
            ))

        # ======================================================
        # Conditional moments
        #
        # STEP 2:
        #   - Store all conditional means because they are needed
        #     later for Cov[x,z] and Var(E[z|x]).
        #
        #   - Do NOT store all conditional covariance matrices.
        #     Accumulate
        #
        #       E[Var(z|x)] ≈ Σ_j w_mean[j] R_j
        #
        #     immediately.
        # ======================================================

        # One column per sigma point
        conditional_means     = ws.conditional_means
        covariance_j          = ws.conditional_covariance
        z_mean                = ws.z_mean
        mean_conditional_covariance = ws.mean_conditional_covariance

        fill!(mean_conditional_covariance, 0)
        for j in 1:number_of_points
            mean_j = @view conditional_means[:, j]
            condMoments(mean_j,covariance_j,param,@view(sigma_points[:, j]),t)
            mean_conditional_covariance .+=w_mean[j] .* covariance_j
        end

        # h(mu_iter)
        z_mean .= @view conditional_means[:, 1]
        
        # ======================================================
        # Cross covariance Cov[x,z]
        # ======================================================

        centered_states = ws.centered_states
        centered_means  = ws.centered_means
        centered_states .= sigma_points .- mu_iter
        centered_means  .= conditional_means .- z_mean

        P_xz = ws.P_xz
        fill!(P_xz, 0)
        @inbounds for j in 1:number_of_points
            wj = w_cov[j]
            for b in axes(centered_means, 1)
                mb = centered_means[b, j]
                for a in axes(centered_states, 1)
                    P_xz[a, b] += wj * centered_states[a, j] * mb
                end
            end
        end

        # ======================================================
        # Total observation covariance
        #
        # Var[z] =
        #   E[Var(z|x)] + Var(E[z|x])
        # ======================================================

        P_z = ws.P_z
        P_z .= mean_conditional_covariance ## E[Var(z|x)]

        # + Var(E[z|x])
        for j in 1:number_of_points
            wj = w_cov[j]
            @inbounds for b in axes(centered_means, 1)
                db = centered_means[b, j]
                for a in axes(centered_means, 1)
                    P_z[a, b] += wj * centered_means[a, j] * db
                end
            end
        end

        P_z = _make_spd(P_z; relative_floor = covariance_floor)

        # ======================================================
        # Statistical linear regression
        #
        # z_t ≈ H_k*x_t + b_k + e_k
        # ======================================================

        #H_k = P_xz' / Omega_iter
        H_k = (F_iter\P_xz)'
        b_k = z_mean - H_k * mu_iter
        #R_k = _make_spd(P_z - H_k * Omega_iter * H_k';relative_floor = covariance_floor,)
        #R_k = P_z - H_k * Omega_iter * H_k'

        # ----------------------------------------------------------
        # Numerically stable SLR residual covariance
        #
        # R_k =
        #   E[Var(z|x)]
        #   + E[(m(x)-z_mean-H_k(x-mu_iter))
        #       (m(x)-z_mean-H_k(x-mu_iter))']
        # ----------------------------------------------------------

        R_k = copy(mean_conditional_covariance)
        for j in 1:number_of_points
           residual_j =@view(centered_means[:, j]) -H_k * @view(centered_states[:, j])
           R_k .+= w_cov[j] .* (residual_j * residual_j')
        end

        # ======================================================
        # Kalman update
        #
        # Linearization changes between IPLF iterations,
        # but every iteration updates from the SAME prior.
        # ======================================================

        z_prior_mean = H_k * mu_prior + b_k
        innovation_covariance = _make_spd(H_k * Omega_prior * H_k' + R_k;relative_floor = covariance_floor,)
        #innovation_covariance = H_k * Omega_prior * H_k' + R_k

        #F_innov =cholesky(Symmetric(innovation_covariance))
        #gain = (F_innov  \ (H_k * Omega_prior'))'
        
        gain = Omega_prior * H_k' / innovation_covariance
        mu_updated =mu_prior + gain * (z_vec - z_prior_mean)

        # OBS! Project precision state back to its admissible domain
        #idx = [3]
        #mu_updated[idx] .= max.(mu_updated[idx], -3)

        ### Wihtout Joseph
        Omega_updated = _make_spd(Omega_prior - gain * innovation_covariance * gain';relative_floor = covariance_floor,)
        Omega_updated = Omega_prior -gain * innovation_covariance * gain'

        # With Joseph
        #n_state = length(mu_prior)
        #I_n = Matrix{eltype(Omega_prior)}(I, n_state, n_state)
        #I_KH = I_n - gain * H_k
        #Omega_updated =I_KH * Omega_prior * I_KH' +gain * R_k * gain'
        #Omega_updated = _make_spd(Omega_updated;relative_floor = covariance_floor)

        # ======================================================
        # IPLF convergence
        # ======================================================

        distance = gaussian_kld(mu_iter,Omega_iter,mu_updated,Omega_updated;relative_floor = covariance_floor)
        #last_diagnostics = (iteration = iteration,distance = distance,predicted_observation = z_prior_mean,marginal_observation = z_mean,
        #observation_covariance = P_z,linearization = H_k,offset = b_k,residual_covariance = R_k,innovation_covariance = innovation_covariance,gain = gain,)

        mu_iter = mu_updated
        Omega_iter = Omega_updated

        distance < tol && break
    end

    # ----------------------------------------------------------
    # Return
    # ----------------------------------------------------------

    if return_diagnostics

        return (
            mu = mu_iter,
            Omega = Omega_iter,
            mu_prior = mu_prior,
            Omega_prior = Omega_prior,
            diagnostics = last_diagnostics,
        )
    end

    return mu_iter, Omega_iter, mu_prior, Omega_prior
end



function kalmanfilter_update_transformed_IPLF_weighted(
    mu::AbstractVector,
    Omega::AbstractMatrix,
    u::AbstractVector,
    z::AbstractVector,
    A::AbstractMatrix,
    B::AbstractMatrix,
    condMoments,
    param,
    Sigma_n::AbstractMatrix,
    t,
    maxIter::Integer,
    gamma,
    w_mean::AbstractVector,
    w_cov::AbstractVector,
    ws;
    tol::Real = 1e-3,
    covariance_floor::Real = 1e-8,
    return_diagnostics::Bool = false,
)

    maxIter >= 1 || throw(ArgumentError("maxIter must be at least one."))

    # ----------------------------------------------------------
    # Prior propagation
    # ----------------------------------------------------------

    mu_prior = A * mu .+ B * u

    Omega_prior = _make_spd(A * Omega * A' + Sigma_n;relative_floor = covariance_floor,)
    #Omega_prior = A * Omega * A' + Sigma_n

    mu_iter = copy(mu_prior)
    Omega_iter = copy(Omega_prior)

    # z is already an AbstractVector; no need to collect/copy it
    z_vec = z
    observation_dimension = length(z_vec)
    last_diagnostics = nothing

    # ----------------------------------------------------------
    # IPLF iterations
    # ----------------------------------------------------------

    for iteration in 1:maxIter

        Omega_iter = _make_spd(Omega_iter;relative_floor = covariance_floor,)
        #L = cholesky(Symmetric(Omega_iter)).L
        F_iter = cholesky(Symmetric(Omega_iter))
        L = F_iter.L

        # Sigma points
        spread = L * gamma

        sigma_points = hcat(mu_iter,mu_iter .+ spread,mu_iter .- spread)
        number_of_points = size(sigma_points, 2)

        length(w_mean) == number_of_points ||
            throw(DimensionMismatch(
                "length(w_mean)=$(length(w_mean)), but there are " *
                "$number_of_points sigma points.",
            ))

        length(w_cov) == number_of_points ||
            throw(DimensionMismatch(
                "length(w_cov)=$(length(w_cov)), but there are " *
                "$number_of_points sigma points.",
            ))

        # ======================================================
        # Conditional moments
        #
        # STEP 2:
        #   - Store all conditional means because they are needed
        #     later for Cov[x,z] and Var(E[z|x]).
        #
        #   - Do NOT store all conditional covariance matrices.
        #     Accumulate
        #
        #       E[Var(z|x)] ≈ Σ_j w_mean[j] R_j
        #
        #     immediately.
        # ======================================================

        # One column per sigma point
        conditional_means     = ws.conditional_means
        covariance_j          = ws.conditional_covariance
        z_mean                = ws.z_mean
        mean_conditional_covariance = ws.mean_conditional_covariance

        fill!(z_mean, 0)
        fill!(mean_conditional_covariance, 0)

        for j in 1:number_of_points

            # Write conditional mean directly into workspace column j
            mean_j = @view conditional_means[:, j]
            condMoments(mean_j,covariance_j,param,@view(sigma_points[:, j]),t)

            length(mean_j) == observation_dimension ||
                throw(DimensionMismatch(
                    "Conditional mean at sigma point $j has length " *
                    "$(length(mean_j)), but observation dimension is " *
                    "$observation_dimension.",
                ))

            size(covariance_j) ==
                (observation_dimension, observation_dimension) ||
                throw(DimensionMismatch(
                    "Conditional covariance at sigma point $j must be " *
                    "$observation_dimension-by-$observation_dimension.",
                ))

            # E[z]
            z_mean .+= w_mean[j] .* mean_j

            # E[Var(z|x)]
            mean_conditional_covariance .+= w_mean[j] .* covariance_j
        end
        # ======================================================
        # Cross covariance Cov[x,z]
        # ======================================================

        centered_states = ws.centered_states
        centered_means  = ws.centered_means
        centered_states .= sigma_points .- mu_iter
        centered_means  .= conditional_means .- z_mean
        P_xz = ws.P_xz
        
        fill!(P_xz, 0)
        @inbounds for j in 1:number_of_points
            wj = w_cov[j]

            for b in axes(centered_means, 1)
                mb = centered_means[b, j]

                for a in axes(centered_states, 1)
                    P_xz[a, b] +=
                        wj * centered_states[a, j] * mb
                end
            end
        end
        # ======================================================
        # Total observation covariance
        #
        # Var[z] =
        #   E[Var(z|x)] + Var(E[z|x])
        # ======================================================

        P_z = ws.P_z
        P_z .= mean_conditional_covariance ## E[Var(z|x)]

        # + Var(E[z|x])
        for j in 1:number_of_points
            wj = w_cov[j]

            @inbounds for b in axes(centered_means, 1)
                db = centered_means[b, j]

                for a in axes(centered_means, 1)
                    P_z[a, b] += wj * centered_means[a, j] * db
                end
            end
        end

        P_z = _make_spd(P_z; relative_floor = covariance_floor)

        # ======================================================
        # Statistical linear regression
        #
        # z_t ≈ H_k*x_t + b_k + e_k
        # ======================================================

        #H_k = P_xz' / Omega_iter
        H_k = (F_iter \ P_xz)'
        b_k = z_mean - H_k * mu_iter
        R_k = _make_spd(P_z - H_k * Omega_iter * H_k';relative_floor = covariance_floor,)
        #R_k = P_z - H_k * Omega_iter * H_k'

        # ----------------------------------------------------------
        # Numerically stable SLR residual covariance
        #
        # R_k =
        #   E[Var(z|x)]
        #   + E[(m(x)-z_mean-H_k(x-mu_iter))
        #       (m(x)-z_mean-H_k(x-mu_iter))']
        # ----------------------------------------------------------

        #R_k = copy(mean_conditional_covariance)

        #for j in 1:number_of_points

           #residual_j =
              # @view(centered_means[:, j]) -
              # H_k * @view(centered_states[:, j])
              # R_k .+= w_cov[j] .* (residual_j * residual_j')
        #end

        # ======================================================
        # Kalman update
        #
        # Linearization changes between IPLF iterations,
        # but every iteration updates from the SAME prior.
        # ======================================================

        z_prior_mean =H_k * mu_prior + b_k
        innovation_covariance = _make_spd(H_k * Omega_prior * H_k' + R_k;relative_floor = covariance_floor,)
        #innovation_covariance = H_k * Omega_prior * H_k' + R_k

        #F_innov =cholesky(Symmetric(innovation_covariance))
        #gain = (F_innov  \ (H_k * Omega_prior'))'
        gain = Omega_prior * H_k' / innovation_covariance
        mu_updated =mu_prior + gain * (z_vec - z_prior_mean)

        # OBS! Project precision state back to its admissible domain
        #precision_idx = param.Zidx[2]
        #precision_floor = -3.0
        #mu_updated[precision_idx] .= max.(mu_updated[precision_idx],precision_floor)

        idx = [3]
        mu_updated[idx] .= max.(mu_updated[idx], -3)

        ### Wihtout Joseph
        Omega_updated = _make_spd(Omega_prior - gain * innovation_covariance * gain';relative_floor = covariance_floor,)
        #Omega_updated = Omega_prior -gain * innovation_covariance * gain'

        # With Joseph
        #n_state = length(mu_prior)
        #I_n = Matrix{eltype(Omega_prior)}(I, n_state, n_state)
        #I_KH = I_n - gain * H_k
        
        #Omega_updated =I_KH * Omega_prior * I_KH' +gain * R_k * gain'
        #Omega_updated = _make_spd(Omega_updated;relative_floor = covariance_floor,)

        # ======================================================
        # IPLF convergence
        # ======================================================

        distance = gaussian_kld(
            mu_iter,
            Omega_iter,
            mu_updated,
            Omega_updated;
            relative_floor = covariance_floor,
        )

        last_diagnostics = (
            iteration = iteration,
            distance = distance,
            predicted_observation = z_prior_mean,
            marginal_observation = z_mean,
            observation_covariance = P_z,
            linearization = H_k,
            offset = b_k,
            residual_covariance = R_k,
            innovation_covariance = innovation_covariance,
            gain = gain,
        )

        mu_iter = mu_updated
        Omega_iter = Omega_updated

        distance < tol && break
    end

    # ----------------------------------------------------------
    # Return
    # ----------------------------------------------------------

    if return_diagnostics

        return (
            mu = mu_iter,
            Omega = Omega_iter,
            mu_prior = mu_prior,
            Omega_prior = Omega_prior,
            diagnostics = last_diagnostics,
        )
    end

    return mu_iter, Omega_iter, mu_prior, Omega_prior
end



###
@inline function _symmetrize_test(P::AbstractMatrix)
    return Matrix(Symmetric((P + P') / 2))
end

function _make_spd(
    P::AbstractMatrix;
    relative_floor::Real = 1e-10
)

    # Symmetrize
    S = Matrix(Symmetric((P + P') / 2))

    # Eigen decomposition
    decomposition = eigen(Symmetric(S))

    λ = decomposition.values

    scale = max(
        maximum(abs, λ),
        one(eltype(S))
    )

    floor_value =
        relative_floor * scale

    # If already safely positive definite, leave unchanged
    if minimum(λ) >= floor_value
        return S
    end

    # Otherwise floor small/negative eigenvalues
    λ_corrected =
        max.(λ, floor_value)

    return Matrix(
        Symmetric(
            decomposition.vectors *
            Diagonal(λ_corrected) *
            decomposition.vectors'
        )
    )
end


function _make_spd_ref(
    P::AbstractMatrix;
    relative_floor::Real=1e-10
)

    S = _symmetrize_test(P)

    decomposition = eigen(Symmetric(S))

    scale = max(
        maximum(abs, decomposition.values),
        one(eltype(S))
    )

    floor_value = relative_floor * scale

    eigenvalues = max.(
        decomposition.values,
        floor_value
    )

    return _symmetrize_test(
        decomposition.vectors *
        Diagonal(eigenvalues) *
        decomposition.vectors'
    )
end


function gaussian_kld(
    m0::AbstractVector,
    P0::AbstractMatrix,
    m1::AbstractVector,
    P1::AbstractMatrix;
    relative_floor::Real=1e-10
)

    P0_spd = _make_spd(P0; relative_floor=relative_floor)
    P1_spd = _make_spd(P1; relative_floor=relative_floor)

    factor0 = cholesky(Symmetric(P0_spd))
    factor1 = cholesky(Symmetric(P1_spd))

    difference = m1 - m0
    dimension = length(m0)

    # tr(P1^{-1} P0)
    trace_term = tr(factor1 \ P0_spd)

    # (m1-m0)' P1^{-1} (m1-m0)
    quadratic_term =
        dot(difference, factor1 \ difference)

    # log(det(P1)) - log(det(P0))
    logdet_P0 = 2 * sum(log, diag(factor0.L))
    logdet_P1 = 2 * sum(log, diag(factor1.L))

    logdet_term = logdet_P1 - logdet_P0

    kld = 0.5 * (
        trace_term +
        quadratic_term -
        dimension +
        logdet_term
    )

    return max(kld, zero(eltype(P0_spd)))
end

##### Laplace constrained
function FFBS_laplace_constrained!(Draws, U, Y, A, B, Σₙ, μ₀, Σ₀, observation, θ;
    filter_output=false, sample_t0=true, μ_init=nothing, max_iter=100,
    nFailure=Ref(0),nLaplaceFailure = Ref(0))

    T = length(Y)   # Number of time steps
    n = length(μ₀)  # Dimension of the state vector  
    #r = size(Y,2)   # Dimension of the observed data vector
    q = size(U, 2)   # Dimension of the control vector
    staticA = (ndims(A) == 3) ? false : true
    staticΣₙ = (ndims(Σₙ) == 3 || eltype(Σₙ) <: PDMat) ? false : true

    # Run Kalman filter and collect matrices
    μ_filter = zeros(T, n)      # Storage of μₜₜ
    Σ_filter = zeros(n, n, T)   # Storage of Σₜₜ
    μ_pred = zeros(T, n)        # Storage of μₜ,ₜ₋₁
    Σ_pred = zeros(n, n, T)     # Storage of Σₜ,ₜ₋₁

    μ = deepcopy(μ₀)
    Σ = deepcopy(Σ₀)
    #nLaplaceFailure = Ref(0)

    for t in 1:T
        At = staticA ? A : @view A[:, :, t]
        Σₙt = staticΣₙ ? Σₙ : Σₙ[t]
        u = (q == 1) ? U[t] : U[t, :]
        #y = (r == 1) ? Y[t] : Y[t,:]
        filter_result = try
            laplace_kalmanfilter_update_constrained(μ, Σ, u, Y[t], At, B, observation, θ, Σₙt, t,μ_init, max_iter; nFailure = nLaplaceFailure)
        catch
            nFailure[] += 1
            return nothing
        end
        μ, Σ, μ̄, Σ̄ = filter_result
        μ_filter[t, :] .= μ
        Σ_filter[:, :, t] .= Σ
        μ_pred[t, :] .= μ̄
        Σ_pred[:, :, t] .= Σ̄
    end

    try
        SMCsamplers.BackwardSampling!(Draws, μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀;
            sample_t0=sample_t0)
    catch
        nFailure[] += 1
        return nothing
    end

    if filter_output
        return μ_filter, Σ_filter
    end
    return nothing
end


function laplace_kalmanfilter_update_constrained(μ, Ω, u, y, A, B, observation, param, Σₙ, t,μ_init=nothing, max_iter=100; nFailure = nothing)

    #nLaplaceFailure = Ref(0)

    # Prior propagation step - moving state forward without new measurement
    μ̄ = A * μ .+ B * u
    Ω̄ = Hermitian(A * Ω * A' + Σₙ)

    if isnothing(μ_init)
        μ_init = μ̄
    end

    # Measurement update - updating the N(μ̄, Ω̄) prior with the new data point
    μ, Ω = try
        filt_logpost(x) = logpdf(observation(param, x, t), y) + logpdf(MvNormal(μ̄[:], Ω̄), x)
        laplace_approximation_constrained(filt_logpost, μ_init, 1.0, max_iter; nFailure=nFailure)  # Initial guess 
    catch
        error("Laplace approximation failed at time $t")
    end

    return μ, Ω, μ̄, Ω̄
end


function laplace_approximation_constrained(
    logposterior,
    initial_guess,
    cov_scale = 1.0,
    max_iter = 10;
    nFailure = nothing
)

    constrained = false

    if constrained

       p = length(initial_guess)

        lower = fill(-Inf, p)
        upper = fill( Inf, p)

        lower[3] = -3.0
        upper[3] = 15.0

        initial_feasible = copy(initial_guess)

        margin = 1e-3

        for j in eachindex(initial_feasible)

            if isfinite(lower[j])
                initial_feasible[j] =
                    max(
                        initial_feasible[j],
                        lower[j] + margin
                    )
            end

            if isfinite(upper[j])
                initial_feasible[j] =
                    min(
                        initial_feasible[j],
                        upper[j] - margin
                    )
            end
        end

        options = Optim.Options(
            iterations = max_iter,
            outer_iterations = 20,
            f_abstol = 1e-6,
            show_warnings = true
        )

        optres = Optim.optimize(
            x -> -logposterior(x),
            lower,
            upper,
            initial_feasible,
            Optim.Fminbox(Optim.LBFGS()),
            options;
            autodiff = :forward
        )

        θ_mode = Optim.minimizer(optres)

        if !Optim.converged(optres)

            if nFailure !== nothing
                nFailure[] += 1
            end
            #println(
                #"WARNING: constrained mode optimization did not converge"
            #)
        end

    else ### not constrained

        optres = Optim.optimize(x -> -logposterior(x), initial_guess, 
                 method=Optim.NewtonTrustRegion(); autodiff=:forward, f_abstol=1e-6, iterations=max_iter)
                 #method=Optim.NewtonTrustRegion();autodiff=:forward, f_abstol=1e-6, iterations=max_iter)

        θ_mode = Optim.minimizer(optres)
        #if Optim.iterations(optres) > 10
           #println("nIter to mode is larger than 10: $(Optim.iterations(optres))")
        #end
        
        if !Optim.converged(optres)

            if nFailure !== nothing
                nFailure[] += 1
            end
     
        end
    end


    # ========================================================
    # Laplace covariance at final mode
    # ========================================================

    #H_mode = ForwardDiff.hessian(logposterior,θ_mode)
    #Q_mode =_make_spd(-H_mode;relative_floor = 1e-8)

    Q_mode =-ForwardDiff.hessian(logposterior,θ_mode)
    Q_mode = Symmetric(Q_mode)
    
    if !isposdef(Q_mode)
        error("Raw negative Hessian is not positive definite at Laplace mode")
    end

    Σ = inv(Matrix(Q_mode))
    #Σ *= cov_scale^2

    return θ_mode, Σ
end



function FFBS_laplace_scaled_constrained!(Draws, U, Y, A, B, Σₙ, μ₀, Σ₀, observation, θ, ScaleMat, Svec;
    filter_output=false, sample_t0=true, μ_init=nothing, max_iter=100,
    nFailure=Ref(0))
    T = length(Y)   # Number of time steps

    n = length(μ₀)  # Dimension of the state vector  
    #r = size(Y,2)   # Dimension of the observed data vector
    q = size(U, 2)   # Dimension of the control vector
    staticA = (ndims(A) == 3) ? false : true
    staticΣₙ = (ndims(Σₙ) == 3 || eltype(Σₙ) <: PDMat) ? false : true

    # Run Kalman filter and collect matrices
    μ_filter = zeros(T, n)      # Storage of μₜₜ
    Σ_filter = zeros(n, n, T)   # Storage of Σₜₜ
    μ_pred = zeros(T, n)        # Storage of μₜ,ₜ₋₁
    Σ_pred = zeros(n, n, T)     # Storage of Σₜ,ₜ₋₁

    μ = deepcopy(μ₀)
    Σ = deepcopy(Σ₀)
    for t in 1:T
        filter_result = try
            S = ScaleMat(θ, μ, t)
            Svec[:, :, t] .= S

            At = staticA ? A : @view A[:, :, t]
            Σₙt = staticΣₙ ? Hermitian(S * Σₙ * S) : Hermitian(S * Σₙ[t] * S) + eps() * I
            u = (q == 1) ? U[t] : U[t, :]
            #y = (r == 1) ? Y[t] : Y[t,:]

            laplace_kalmanfilter_update(μ, Σ, u, Y[t], At, B, observation, θ, Σₙt, t,
                μ_init, max_iter)
        catch
            nFailure[] += 1
            return nothing
        end
        μ, Σ, μ̄, Σ̄ = filter_result
        μ_filter[t, :] .= μ
        Σ_filter[:, :, t] .= Σ
        μ_pred[t, :] .= μ̄
        Σ_pred[:, :, t] .= Σ̄
    end

    BackwardSampling!(Draws, μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀;
        sample_t0=sample_t0)

    if filter_output
        return μ_filter, Σ_filter
    end
    return nothing
end


function laplace_approximation_constrained_ref(logposterior, initial_guess, cov_scale=1.0, max_iter=100)

    if t==1
        println("USING REF function")
    end

    # Find mode (MAP estimate)
    handbaked = false
    if handbaked
        function find_mode(x0)
            x = copy(x0)
            for _ in 1:max_iter
                g = ForwardDiff.gradient(logposterior, x)
                H = ForwardDiff.hessian(logposterior, x)
                #g = ForwardDiff.derivative(logposterior, x)
                #H = ForwardDiff.derivative(x -> ForwardDiff.derivative(logposterior, x), x)
                Δx = -H \ g  # Newton-Raphson step
                x += Δx
                if norm(Δx) < 1e-6
                    return x
                end
            end
            error("Mode finding did not converge")
        end
        θ_mode = find_mode(initial_guess)
    else

        optres = Optim.optimize(x -> -logposterior(x), initial_guess, method=Optim.NewtonTrustRegion();autodiff=:forward, f_abstol=1e-6, iterations=max_iter)

        θ_mode = Optim.minimizer(optres)
        if Optim.iterations(optres) > 10
            println("nIter to mode is larger than 10: $(Optim.iterations(optres))")
        end
    end

    #idx = [3]
    #θ_mode[idx] .= max.(θ_mode[idx], -3)

    # Compute Hessian at mode
    Σ = -inv(ForwardDiff.hessian(logposterior, θ_mode))  # Covariance matrix
    #Σ = -inv(ForwardDiff.derivative(θ_mode -> ForwardDiff.derivative(logposterior, #θ_mode), θ_mode))  # Covariance matrix

    # Adjust covariance if needed (sometimes too narrow/wide)
    Σ *= cov_scale^2

    #idx = [1,2,3]
    #θ_mode[idx] .= max.(θ_mode[idx], -5)

    # Return results
    return θ_mode, Σ

end
