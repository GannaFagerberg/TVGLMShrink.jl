function FFBS_SLR_test!(
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
    maxIter;
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
            Matrix(Σₙ_raw) + eps(Float64) * I
        )

        # Always keep the control input as a vector.
        u = @view U[t, :]

        filter_result = try

           kalmanfilter_update_IPLF_test(
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
                ωₛ
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

    SMCsamplers.BackwardSampling!(
        Draws,
        μ_filter,
        Σ_filter,
        μ_pred,
        Σ_pred,
        A,
        μ₀,
        Σ₀;
        sample_t0=sample_t0
    )

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


# Extract individual conditional variances
function _beta_variance_vector(value, group_size::Int)

    if value isa Real

        group_size == 1 || throw(DimensionMismatch(
            "A scalar variance was returned for group size $group_size."
        ))

        return [float(value)]

    elseif value isa AbstractVector

        length(value) == group_size || throw(DimensionMismatch(
            "Variance vector has length $(length(value)); " *
            "expected $group_size."
        ))

        return vec(float.(value))

    elseif value isa AbstractMatrix

        size(value) == (group_size, group_size) ||
            throw(DimensionMismatch(
                "Conditional covariance has size $(size(value)); " *
                "expected ($group_size, $group_size)."
            ))

        return float.(diag(value))

    else
        throw(ArgumentError(
            "Unsupported conditional covariance type $(typeof(value))."
        ))
    end
end

### More allocation free
# Take a group of beta observations
# Transforms very observation into the two sufficient statistics of the Beta exponential family
# Keeps the two sufficient statistics for every observation in the group
# Stores all the log(y) first and then log(1-y)
# Lopps through every observation

function beta_sufficient_observation_grouped(
    y;
    clip::Bool=false,
    boundary::Real=1e-12
)

    y_group = _beta_vector(y, "y")
    g = length(y_group)

    z = Vector{Float64}(undef, 2 * g)

    @inbounds for i in 1:g

        yi = y_group[i]

        isfinite(yi) ||
            throw(DomainError(
                yi,
                "Beta observations must be finite."
            ))

        if !(0 < yi < 1)
            if clip
                yi = clamp(yi, boundary, 1.0 - boundary)
            else
                throw(DomainError(
                    yi,
                    "Beta observations must lie in (0,1)."
                ))
            end
        end

        z[i]     = log(yi)
        z[g + i] = log1p(-yi)
    end

    return z
end

function beta_sufficient_observation_grouped_ref(
    y;
    clip::Bool=false,
    boundary::Real=1e-12
)

    y_group = _beta_vector(y, "y")

    if clip
        0 < boundary < 0.5 ||
            throw(ArgumentError(
                "boundary must lie in (0, 0.5)."
            ))
    end

    y_used = similar(y_group)

    for i in eachindex(y_group)

        yi = y_group[i]

        isfinite(yi) ||
            throw(DomainError(
                yi,
                "Beta observations must be finite."
            ))

        if 0 < yi < 1
            y_used[i] = yi
        elseif clip
            y_used[i] = clamp(
                yi,
                boundary,
                1.0 - boundary
            )
        else
            throw(DomainError(
                yi,
                "Beta observations must lie strictly in (0,1)."
            ))
        end
    end

    # Ordering:
    # [log(y₁), ..., log(y_g),
    #  log(1-y₁), ..., log(1-y_g)]
    return vcat(
        log.(y_used),
        log1p.(-y_used)
    )
end


## Takes existing Beta model functions for the raw observation
## Constructs new functions giving the conditional moments of the transformed sufficient-statistic observation
##

function make_beta_sufficient_statistics_adapters_grouped(
    raw_condMean,
    raw_condCov;
    variance_denominator_offset::Real = 1.0,
    mean_boundary::Real = 1e-12,
    min_concentration::Real = 1e-10
)

    function beta_shapes_from_original_model(
        param,
        state,
        t
    )

        # Conditional mean of original Beta observations
        μ_raw = raw_condMean(param, state, t)

        # Conditional covariance/variance of original Beta observations
        covariance_raw = raw_condCov(param, state, t)

        μ = _beta_vector(
            μ_raw,
            "raw_condMean output"
        )

        group_size = length(μ)

        variance_y = _beta_variance_vector(
            covariance_raw,
            group_size
        )

        # Numerical protection
        μ = clamp.(
            μ,
            mean_boundary,
            1.0 - mean_boundary
        )

        all(isfinite, variance_y) &&
            all(variance_y .> 0) ||
            throw(DomainError(
                variance_y,
                "Conditional Beta variances must be finite and positive."
            ))

        # Var(Y|x) = μ(1-μ)/(κ+1)
        κ =
            μ .* (1.0 .- μ) ./ variance_y .-
            variance_denominator_offset

        all(isfinite, κ) &&
            all(κ .> 0) ||
            throw(DomainError(
                κ,
                "The conditional moments imply non-positive concentration."
            ))

        κ = max.(κ, min_concentration)

        alpha_shape =
            max.(μ .* κ, min_concentration)

        beta_shape =
            max.((1.0 .- μ) .* κ, min_concentration)

        # Account for numerical flooring
        κ_effective = alpha_shape .+ beta_shape

        return κ_effective, alpha_shape, beta_shape
    end


    # ==========================================================
    # BOTH sufficient-statistic moments in one call
    # ==========================================================

    function condMoments_beta_sufficient_grouped(
        param,
        state,
        t
    )

        # IMPORTANT:
        # Beta shapes are constructed only ONCE here.
        κ, alpha_shape, beta_shape =
            beta_shapes_from_original_model(
                param,
                state,
                t
            )

        # ------------------------------------------------------
        # Conditional mean
        # ------------------------------------------------------

        mean_log_y =
            digamma.(alpha_shape) .-
            digamma.(κ)

        mean_log_one_minus_y =
            digamma.(beta_shape) .-
            digamma.(κ)

        mean_z = vcat(
            mean_log_y,
            mean_log_one_minus_y
        )

        # ------------------------------------------------------
        # Conditional covariance
        # ------------------------------------------------------

        trigamma_κ = trigamma.(κ)

        variance_log_y =
            trigamma.(alpha_shape) .-
            trigamma_κ

        variance_log_one_minus_y =
            trigamma.(beta_shape) .-
            trigamma_κ

        covariance_logs =
            -trigamma_κ

        g = length(κ)

        covariance_type = promote_type(
            eltype(variance_log_y),
            eltype(variance_log_one_minus_y),
            Float64
        )

        R = zeros(
            covariance_type,
            2 * g,
            2 * g
        )

        @inbounds for i in 1:g

            j = g + i

            R[i, i] =
                variance_log_y[i]

            R[j, j] =
                variance_log_one_minus_y[i]

            R[i, j] =
                covariance_logs[i]

            R[j, i] =
                covariance_logs[i]
        end

        return mean_z, R
    end

    return condMoments_beta_sufficient_grouped
end


function kalmanfilter_update_IPLF_test(
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
    w_cov::AbstractVector;
    tol::Real = 1e-3,
    covariance_floor::Real = 1e-10,
    return_diagnostics::Bool = false,
)

    maxIter >= 1 ||
        throw(ArgumentError("maxIter must be at least one."))

    # ----------------------------------------------------------
    # Prior propagation
    # ----------------------------------------------------------

    mu_prior = A * mu .+ B * u

    #Omega_prior = _make_spd(A * Omega * A' + Sigma_n;relative_floor = covariance_floor,)
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

    for iteration in 1:maxIter

        #Omega_iter = _make_spd(Omega_iter;relative_floor = covariance_floor,)
       
        L = cholesky(Symmetric(Omega_iter)).L

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

        # Evaluate first sigma point so that we can infer element types
        mean_j, covariance_j =
            condMoments(
                param,
                @view(sigma_points[:, 1]),
                t
            )

        length(mean_j) == observation_dimension ||
            throw(DimensionMismatch(
                "Conditional mean has length $(length(mean_j)), " *
                "but observation dimension is $observation_dimension.",
            ))

        size(covariance_j) ==
            (observation_dimension, observation_dimension) ||
            throw(DimensionMismatch(
                "Conditional covariance must be " *
                "$observation_dimension-by-$observation_dimension.",
            ))

        mean_type = promote_type(
            eltype(mean_j),
            eltype(z_vec),
            eltype(Omega_iter)
        )

        covariance_type = promote_type(
            eltype(covariance_j),
            eltype(Omega_iter)
        )

        # One column per sigma point
        conditional_means = Matrix{mean_type}(
            undef,
            observation_dimension,
            number_of_points
        )

        # Marginal observation mean E[z]
        z_mean = zeros(
            mean_type,
            observation_dimension
        )

        # Accumulates E[Var(z|x)]
        mean_conditional_covariance = zeros(
            covariance_type,
            observation_dimension,
            observation_dimension
        )

        # ------------------------------------------------------
        # First sigma point
        # ------------------------------------------------------

        conditional_means[:, 1] .= mean_j

        z_mean .+=
            w_mean[1] .* mean_j

        mean_conditional_covariance .+=
            w_mean[1] .* covariance_j

        # ------------------------------------------------------
        # Remaining sigma points
        # ------------------------------------------------------

        for j in 2:number_of_points

            mean_j, covariance_j =
                condMoments(
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

            # Store mean because it is needed later
            conditional_means[:, j] .= mean_j

            # E[z]
            z_mean .+=
                w_mean[j] .* mean_j

            # E[Var(z|x)]
            #
            # covariance_j is NOT stored after this iteration.
            mean_conditional_covariance .+=
                w_mean[j] .* covariance_j
        end

        # ======================================================
        # Cross covariance Cov[x,z]
        # ======================================================

        centered_states =
            sigma_points .- mu_iter

        # conditional_means is now an observation_dimension ×
        # number_of_points matrix.
        centered_means =
            conditional_means .- z_mean

        # Cov[x,z] = Cov[x,E[z|x]]
        P_xz =
            centered_states *
            Diagonal(w_cov) *
            centered_means'

        # ======================================================
        # Total observation covariance
        #
        # Var[z]
        #   = E[Var(z|x)]
        #   + Var(E[z|x])
        # ======================================================

        # First term was already accumulated above
        P_z = copy(mean_conditional_covariance)

        # Add Var(E[z|x])
        for j in 1:number_of_points

            difference =
                @view centered_means[:, j]

            P_z .+=
                w_cov[j] .* (difference * difference')
        end

        #P_z = _make_spd(P_z; relative_floor = covariance_floor)

        # ======================================================
        # Statistical linear regression
        #
        # z_t ≈ H_k*x_t + b_k + e_k
        # ======================================================

        H_k = P_xz' / Omega_iter

        b_k =
            z_mean - H_k * mu_iter

        #R_k = _make_spd(P_z - H_k * Omega_iter * H_k';relative_floor = covariance_floor,)
        R_k = P_z - H_k * Omega_iter * H_k'

        # ======================================================
        # Kalman update
        #
        # Linearization changes between IPLF iterations,
        # but every iteration updates from the SAME prior.
        # ======================================================

        z_prior_mean =
            H_k * mu_prior + b_k

        #innovation_covariance = _make_spd(H_k * Omega_prior * H_k' + R_k;relative_floor = covariance_floor,)
        innovation_covariance = H_k * Omega_prior * H_k' + R_k

        gain =
            Omega_prior * H_k' / innovation_covariance

        mu_updated =
            mu_prior +
            gain * (z_vec - z_prior_mean)

        #Omega_updated = _make_spd(Omega_prior -gain * innovation_covariance * gain';relative_floor = covariance_floor,)
        Omega_updated = Omega_prior -gain * innovation_covariance * gain'

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

@inline function _symmetrize_test(P::AbstractMatrix)
    return Matrix(Symmetric((P + P') / 2))
end

function _make_spd(
    P::AbstractMatrix;
    relative_floor::Real = 1e-10
)

    # Symmetrize first
    S = Matrix(Symmetric((P + P') / 2))

    # Check whether already positive definite
    F = cholesky(Symmetric(S); check = false)

    if issuccess(F)
        return S
    end

    println("Matrix is not positive definite. Applying eigenvalue correction.")

    # Eigenvalue correction
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

    return Matrix(
        Symmetric(
            decomposition.vectors *
            Diagonal(eigenvalues) *
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

    #P0_spd = _make_spd(P0;relative_floor=relative_floor)
    P0_spd = P0

    #P1_spd = _make_spd(P1;relative_floor=relative_floor)
    P1_spd  = P1

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


