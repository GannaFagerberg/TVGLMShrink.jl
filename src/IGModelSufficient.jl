# ============================================================
# Inverse Gaussian sufficient-statistic observation helpers
#
# Parameterisation:
#
#   Y | μ, κ ~ InverseGaussian(μ, κ)
#
# with
#
#   E[Y]   = μ
#   Var[Y] = μ^3 / κ
#
# Latent-state predictors:
#
#   μ = g_μ^{-1}(η_μ)
#   κ = g_κ^{-1}(η_κ)
#
# Observation transformation:
#
#   z(y) =
#   [
#       y,
#       1/y
#   ]
#
# For a group of observations:
#
#   z =
#   [
#       Y_1, ..., Y_g,
#       1/Y_1, ..., 1/Y_g
#   ]
#
# Hence dimension = 2g.
# ============================================================


# ------------------------------------------------------------
# Helper: convert scalar/vector observation to vector
# ------------------------------------------------------------

function _inverse_gaussian_vector(y, name="y")

    if y isa Real
        return [float(y)]

    elseif y isa AbstractVector
        return vec(float.(y))

    else
        throw(ArgumentError(
            "$name must be a scalar or vector, got $(typeof(y))."
        ))
    end
end


# ============================================================
# TRANSFORM RAW OBSERVATIONS
# ============================================================

function inverse_gaussian_sufficient_observation_grouped(
    y;
    clip::Bool = false,
    boundary::Real = 1e-12
)

    y_group = _inverse_gaussian_vector(y, "y")
    g = length(y_group)

    z = Vector{Float64}(undef, 2g)

    @inbounds for i in 1:g

        yi = y_group[i]

        isfinite(yi) ||
            throw(DomainError(
                yi,
                "Inverse Gaussian observations must be finite."
            ))

        if yi <= 0

            if clip
                yi = max(yi, boundary)

            else
                throw(DomainError(
                    yi,
                    "Inverse Gaussian observations must be positive."
                ))
            end
        end

        # First sufficient statistic
        z[i] = yi

        # Second sufficient statistic
        z[g + i] = inv(yi)
    end

    return z
end


# ============================================================
# CONDITIONAL MOMENTS OF GROUPED SUFFICIENT STATISTICS
# ============================================================

function make_inverse_gaussian_sufficient_statistics_adapters_grouped(
    raw_condMean,
    raw_condCov;
    mean_floor::Real = 1e-12,
    shape_floor::Real = 1e-12
)

    # raw_condMean/raw_condCov are kept in the signature so that
    # this follows exactly the same interface as the Beta code.
    #
    # Here we can obtain μ and κ directly from the state.

    function condMoments_inverse_gaussian_sufficient_grouped!(
        mean_z,
        R,
        param,
        state,
        t
    )

        # ------------------------------------------------------
        # Linear predictors
        # ------------------------------------------------------

        ημ =
            param.Z[1][t] *
            state[param.Zidx[1]]

        ηκ =
            param.Z[2][t] *
            state[param.Zidx[2]]


        # ------------------------------------------------------
        # Inverse link
        # ------------------------------------------------------

        μ =
            linkinv.(
                Ref(param.link[1]),
                ημ
            )

        κ =
            linkinv.(
                Ref(param.link[2]),
                ηκ
            )

        #if minimum(μ) < 1e-4 || minimum(κ) < 1e-4

            #println("Extreme IG sigma point at t = $t")

            #@show minimum(ημ)
            #@show maximum(ημ)
           # @show minimum(ηκ)
            #@show maximum(ηκ)

           # @show minimum(μ)
           # @show maximum(μ)
           # @show minimum(κ)
           # @show maximum(κ)

        #end


        # ------------------------------------------------------
        # Numerical safeguards
        # ------------------------------------------------------

        μ = max.(μ, mean_floor)
        κ = max.(κ, shape_floor)

        all(isfinite, μ) ||
            throw(DomainError(
                μ,
                "Non-finite Inverse Gaussian mean."
            ))

        all(isfinite, κ) ||
            throw(DomainError(
                κ,
                "Non-finite Inverse Gaussian shape parameter."
            ))



        # Number of observations in this group
        g = length(μ)

        length(κ) == g ||
        throw(DimensionMismatch(
            "μ has length $g but κ has length $(length(κ))."
        ))

        length(mean_z) == 2g ||
            throw(DimensionMismatch(
                "mean_z has length $(length(mean_z)); expected $(2g)."
            ))

        size(R) == (2g, 2g) ||
            throw(DimensionMismatch(
                "R has size $(size(R)); expected ($(2g), $(2g))."
            ))


        # Conditional independence across observations
        fill!(R, zero(eltype(R)))


        # ======================================================
        # Moments
        # ======================================================

        @inbounds for i in 1:g

            j = g + i

            μi = μ[i]
            κi = κ[i]


            # --------------------------------------------------
            # Conditional means
            #
            # E[Y]     = μ
            #
            # E[1/Y]   = 1/μ + 1/κ
            # --------------------------------------------------

            mean_z[i] =
                μi

            mean_z[j] =
                inv(μi) + inv(κi)


            # --------------------------------------------------
            # Conditional covariance
            #
            # Var(Y) =
            #     μ^3 / κ
            #
            # Cov(Y, 1/Y) =
            #     -μ / κ
            #
            # Var(1/Y) =
            #     1/(μκ) + 2/κ^2
            # --------------------------------------------------

            variance_y =
                μi^3 / κi

            variance_inv_y =
                inv(μi * κi) +
                2.0 / κi^2

            covariance_y_inv_y =
                -μi / κi


            R[i, i] =
                variance_y

            R[j, j] =
                variance_inv_y

            R[i, j] =
                covariance_y_inv_y

            R[j, i] =
                covariance_y_inv_y
        end

        return nothing
    end

    return condMoments_inverse_gaussian_sufficient_grouped!
end


# ============================================================
# TRANSFORM TYPE
# ============================================================

struct InverseGaussianSuffStats{F1,F2,F3} <: AbstractObsTransform
    transform_obs::F1
    make_cond_moments::F2
    obs_dim::F3
end


# ============================================================
# GROUPED TRANSFORM
# ============================================================

function InverseGaussianSuffStatsGrouped(;
    mean_floor::Real = 1e-12,
    shape_floor::Real = 1e-12
)

    return InverseGaussianSuffStats(

        # Transform raw observations
        y -> inverse_gaussian_sufficient_observation_grouped(y),

        # Construct conditional-moment function
        (condMean, condCov) ->
            make_inverse_gaussian_sufficient_statistics_adapters_grouped(
                condMean,
                condCov;
                mean_floor = mean_floor,
                shape_floor = shape_floor
            ),

        # Dimension of transformed observation
        nPerGroup -> 2 * nPerGroup
    )
end


# ============================================================
# PREPARE TRANSFORM FOR IPLF
# ============================================================

function prepare_observation_transform(
    transform::InverseGaussianSuffStats,
    Y,
    condMean,
    condCov,
    nPerGroup
)

    Y_transformed = [
        transform.transform_obs(Y[t])
        for t in eachindex(Y)
    ]

    condMoments =
        transform.make_cond_moments(
            condMean,
            condCov
        )

    nObs =
        transform.obs_dim(nPerGroup)

    return (
        Y = Y_transformed,
        condMoments = condMoments,
        nObs = nObs
    )
end

##################################
# FISHER
##################################
### Inverse Gaussian Fisher information

function fisher_inversegaussian_blocks(
    Xm,
    Xp,
    βm,
    βp,
    linkm,
    linkp
)

    # Linear predictors
    ηm = Xm * βm
    ηp = Xp * βp

    # Distribution parameters before numerical safeguards
    μ_raw = linkinv.(linkm, ηm)
    κ_raw = linkinv.(linkp, ηp)

    # Derivatives wrt linear predictors
    dμ = mueta.(linkm, ηm)
    dκ = mueta.(linkp, ηp)

    # Check validity
    all(isfinite, μ_raw) ||
        throw(DomainError(
            μ_raw,
            "Non-finite Inverse Gaussian mean."
        ))

    all(isfinite, κ_raw) ||
        throw(DomainError(
            κ_raw,
            "Non-finite Inverse Gaussian shape parameter."
        ))

    # Numerical safeguards
    μ_floor = 1e-12
    κ_floor = 1e-12

    μ = max.(μ_raw, μ_floor)
    κ = max.(κ_raw, κ_floor)

    # Fisher weights
    wm =
        (κ ./ μ.^3) .* dμ.^2

    wp =
        (1.0 ./ (2.0 .* κ.^2)) .* dκ.^2

    # State-space Fisher blocks
    Hmm = XDiagX(Xm, wm)
    Hpp = XDiagX(Xp, wp)

    # Mean and shape are Fisher orthogonal
    Hmp = zeros(
        promote_type(eltype(Hmm), eltype(Hpp)),
        size(Xm, 2),
        size(Xp, 2)
    )

    return [
        Hmm   Hmp
        Hmp'  Hpp
    ]
end


# ============================================================
# Fisher information using the full design matrices
# ============================================================

function FisherInfoInverseGaussian(param, μ, t)

    pm = size(param.X[1], 2)

    return fisher_inversegaussian_blocks(
        param.X[1],
        param.X[2],
        μ[1:pm],
        μ[(pm + 1):end],
        param.link[1],
        param.link[2]
    )
end


# ============================================================
# Local Fisher information at time/group t
# ============================================================

function FisherInfoInverseGaussian_local(param, μ, t)

    pm = size(param.Z[1][t], 2)

    return fisher_inversegaussian_blocks(
        param.Z[1][t],
        param.Z[2][t],
        μ[1:pm],
        μ[(pm + 1):end],
        param.link[1],
        param.link[2]
    )
end