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



function FFBS_SLR_transformed_scaling!(
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
    ScaleMat, Svec, 
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
        ones(2 * n) / (2 * (n + λ))
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

        S = ScaleMat(param, μ, t)
        Svec[:, :, t] .= S

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

        Σₙt = Hermitian(
            Matrix(S * Σₙ_raw * S) + eps(Float64) * I
        )

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
        ones(2 * n) / (2 * (n + λ))
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

    maxIter >= 1 ||
        throw(ArgumentError("maxIter must be at least one."))

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

        sigma_points = hcat(
            mu_iter,
            mu_iter .+ spread,
            mu_iter .- spread
        )

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
        covariance_j       = ws.conditional_covariance
        z_mean                = ws.z_mean
        mean_conditional_covariance = ws.mean_conditional_covariance

        fill!(z_mean, 0)
        fill!(mean_conditional_covariance, 0)

        for j in 1:number_of_points

            # Write conditional mean directly into workspace column j
            mean_j = @view conditional_means[:, j]

            condMoments(
                mean_j,
                covariance_j,
                param,
                @view(sigma_points[:, j]),
                t
            )

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

        #if t == 1
            #@show P_xz

            #for k in axes(P_xz, 1)
               # println(
                    #"state $k: ",
                   # norm(@view P_xz[k, :])
               # )
            #end
        #end
        # ======================================================
        # Total observation covariance
        #
        # Var[z] =
        #   E[Var(z|x)] + Var(E[z|x])
        # ======================================================

        P_z = ws.P_z

        # E[Var(z|x)]
        P_z .= mean_conditional_covariance

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

        F_innov =cholesky(Symmetric(innovation_covariance))
        gain = (F_innov  \ (H_k * Omega_prior'))'

        #valsS = eigvals(Symmetric(innovation_covariance))
        # if minimum(valsS) < 1e-10 * maximum(valsS)
           # @show t
           # @show minimum(valsS)
           # @show maximum(valsS)
           # @show cond(innovation_covariance)
            #@show eigvals(Symmetric(R_k))
           # @show mu
        #end
        #gain = Omega_prior * H_k' / innovation_covariance

        mu_updated =mu_prior + gain * (z_vec - z_prior_mean)

        # OBS! Project precision state back to its admissible domain
        #precision_idx = param.Zidx[2]
        #precision_floor = -3.0
        #mu_updated[precision_idx] .= max.(mu_updated[precision_idx],precision_floor)

        idx = [1, 2]
        parameter_floor = 0.1^3
        #hit_floor = any(mu_updated[idx] .< parameter_floor)
        #if hit_floor
            #@show t iteration mu_updated[idx]
        #end

        mu_updated[idx] .= max.(mu_updated[idx], parameter_floor)

        ### Wihtout Joseph
        #Omega_updated = _make_spd(Omega_prior - gain * innovation_covariance * gain';relative_floor = covariance_floor,)
        #Omega_updated = Omega_prior -gain * innovation_covariance * gain'

        # With Joseph
        n_state = length(mu_prior)
        I_n = Matrix{eltype(Omega_prior)}(I, n_state, n_state)
        I_KH = I_n - gain * H_k
        
        Omega_updated =
            I_KH * Omega_prior * I_KH' +
            gain * R_k * gain'

        Omega_updated = _make_spd(
            Omega_updated;
            relative_floor = covariance_floor,
        )

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

        #damping = 0.3
        #mu_next =
            #(1 - damping) .* mu_iter +
            #damping .* mu_updated

        #Omega_next =
            #(1 - damping) .* Omega_iter +
           #damping .* Omega_updated

        #Omega_next = _make_spd(
            #Omega_next;
           #relative_floor = covariance_floor,
        #)

        #mu_iter = mu_next
        #Omega_iter = Omega_next

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

function gaussian_kld_ref(
    m0::AbstractVector,
    P0::AbstractMatrix,
    m1::AbstractVector,
    P1::AbstractMatrix;
    relative_floor::Real=1e-10
)

    P0_spd = _make_spd(P0;relative_floor=relative_floor)
    #P0_spd = P0

    P1_spd = _make_spd(P1;relative_floor=relative_floor)
    #P1_spd  = P1

    factor1 = cholesky(Symmetric(P1_spd))

    difference = m1 - m0
    dimension = length(m0)

    trace_term = tr(factor1 \ P0_spd)

    quadratic_term =
        dot(difference, factor1 \ difference)

    logdet_term =
        logdet(Symmetric(P1_spd)) -
        logdet(Symmetric(P0_spd))

    return max(
        0.5 * (
            trace_term +
            quadratic_term -
            dimension +
            logdet_term
        ),
        zero(eltype(P0_spd))
    )
end


