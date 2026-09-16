# ============================================================
# Von Mises sufficient-statistic observation helpers
#
# Parameterisation:
#
#   Y | μ, κ ~ VonMises(μ, κ)
#
# with density
#
#   p(y | μ, κ)
#       = exp{κ cos(y - μ)} / (2π I₀(κ))
#
# Latent-state predictors:
#
#   μ = g_μ^{-1}(η_μ)
#   κ = g_κ^{-1}(η_κ)
#
# Typically:
#
#   g_μ^{-1}(η_μ) = η_μ
#   g_κ^{-1}(η_κ) = exp(η_κ)
#
# Observation transformation:
#
#   z(y) =
#   [
#       cos(y),
#       sin(y)
#   ]
#
# For a group of observations:
#
#   z =
#   [
#       cos(Y₁), ..., cos(Y_g),
#       sin(Y₁), ..., sin(Y_g)
#   ]
#
# Hence dimension = 2g.
# ============================================================


# ------------------------------------------------------------
# Helper: convert scalar/vector observation to vector
# ------------------------------------------------------------

function _von_mises_vector(y, name="y")

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
# BESSEL RATIOS
# ============================================================

# A_j(κ) = I_j(κ) / I_0(κ)
#
# besselix is used because the common exponential scaling
# cancels in the ratio.

_von_mises_A1(κ) =
    besselix(1, κ) / besselix(0, κ)

_von_mises_A2(κ) =
    besselix(2, κ) / besselix(0, κ)


# ============================================================
# TRANSFORM RAW OBSERVATIONS
# ============================================================

function von_mises_sufficient_observation_grouped(y)

    y_group = _von_mises_vector(y, "y")
    g = length(y_group)

    z = Vector{Float64}(undef, 2g)

    @inbounds for i in 1:g

        yi = y_group[i]

        isfinite(yi) ||
            throw(DomainError(
                yi,
                "Von Mises observations must be finite."
            ))

        # First sufficient statistic
        z[i] = cos(yi)

        # Second sufficient statistic
        z[g + i] = sin(yi)
    end

    return z
end


# ============================================================
# CONDITIONAL MOMENTS OF GROUPED SUFFICIENT STATISTICS
# ============================================================

function make_von_mises_sufficient_statistics_adapters_grouped(
    raw_condMean,
    raw_condCov;
    concentration_floor::Real = 1e-10
)

    # raw_condMean/raw_condCov are retained only so that
    # this has the same interface as the other observation
    # transformations.
    #
    # μ and κ are obtained directly from the latent state.

    function condMoments_von_mises_sufficient_grouped!(
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
        # Inverse links
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


        # ------------------------------------------------------
        # Numerical safeguard
        #
        # μ is an angle and must NOT be made positive.
        # Only κ requires κ > 0.
        # ------------------------------------------------------

        κ = max.(κ, concentration_floor)

        all(isfinite, μ) ||
            throw(DomainError(
                μ,
                "Non-finite Von Mises mean direction."
            ))

        all(isfinite, κ) ||
            throw(DomainError(
                κ,
                "Non-finite Von Mises concentration."
            ))


        # Number of observations in the group
        g = length(μ)

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

            A1 = _von_mises_A1(κi)
            A2 = _von_mises_A2(κi)


            # --------------------------------------------------
            # Conditional means
            #
            # E[cos(Y)] = A₁(κ) cos(μ)
            #
            # E[sin(Y)] = A₁(κ) sin(μ)
            # --------------------------------------------------

            mean_cos =
                A1 * cos(μi)

            mean_sin =
                A1 * sin(μi)

            mean_z[i] =
                mean_cos

            mean_z[j] =
                mean_sin


            # --------------------------------------------------
            # Second moments
            #
            # E[cos²(Y)]
            #   = 1/2 [1 + A₂(κ) cos(2μ)]
            #
            # E[sin²(Y)]
            #   = 1/2 [1 - A₂(κ) cos(2μ)]
            #
            # E[cos(Y) sin(Y)]
            #   = 1/2 A₂(κ) sin(2μ)
            # --------------------------------------------------

            E_cos2 =
                0.5 * (
                    1.0 +
                    A2 * cos(2.0 * μi)
                )

            E_sin2 =
                0.5 * (
                    1.0 -
                    A2 * cos(2.0 * μi)
                )

            E_cos_sin =
                0.5 *
                A2 *
                sin(2.0 * μi)


            # --------------------------------------------------
            # Conditional covariance
            # --------------------------------------------------

            variance_cos =
                E_cos2 -
                mean_cos^2

            variance_sin =
                E_sin2 -
                mean_sin^2

            covariance_cos_sin =
                E_cos_sin -
                mean_cos * mean_sin


            R[i, i] =
                variance_cos

            R[j, j] =
                variance_sin

            R[i, j] =
                covariance_cos_sin

            R[j, i] =
                covariance_cos_sin
        end

        return nothing
    end

    return condMoments_von_mises_sufficient_grouped!
end


# ============================================================
# TRANSFORM TYPE
# ============================================================

struct VonMisesSuffStats{F1,F2,F3} <: AbstractObsTransform
    transform_obs::F1
    make_cond_moments::F2
    obs_dim::F3
end


# ============================================================
# GROUPED TRANSFORM
# ============================================================

function VonMisesSuffStatsGrouped(;
    concentration_floor::Real = 1e-10
)

    return VonMisesSuffStats(

        # Transform raw observations
        y ->
            von_mises_sufficient_observation_grouped(y),

        # Construct conditional-moment function
        (condMean, condCov) ->
            make_von_mises_sufficient_statistics_adapters_grouped(
                condMean,
                condCov;
                concentration_floor = concentration_floor
            ),

        # Dimension of transformed observation
        nPerGroup -> 2 * nPerGroup
    )
end


# ============================================================
# PREPARE TRANSFORM FOR IPLF
# ============================================================

function prepare_observation_transform(
    transform::VonMisesSuffStats,
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

#################################
### Fisher##
################################

##################################
# FISHER
##################################

### Von Mises Fisher information
function fisher_vonmises_blocks(
    Xm,
    Xp,
    βm,
    βp,
    linkm::Link,
    linkp::Link
)

    # ----------------------------------------------------------
    # Linear predictors
    # ----------------------------------------------------------

    ηm = Xm * βm
    ηp = Xp * βp


    # ----------------------------------------------------------
    # Mean direction and concentration
    # ----------------------------------------------------------

    μ = linkinv.(Ref(linkm), ηm)
    κ = linkinv.(Ref(linkp), ηp)


    # ----------------------------------------------------------
    # Link derivatives
    #
    # With IdentityLink:
    #     dμ/dηm = 1
    #
    # With LogLinLink:
    #     dκ/dηp = κ
    # ----------------------------------------------------------

    dμ = mueta.(Ref(linkm), ηm)
    dκ = mueta.(Ref(linkp), ηp)


    # ----------------------------------------------------------
    # Numerical safeguards
    #
    # No restriction is required for μ.
    # Only κ must be positive.
    # ----------------------------------------------------------

    κ = max.(κ, 1e-10)


    # ----------------------------------------------------------
    # Bessel ratios
    #
    # A₁(κ) = I₁(κ) / I₀(κ)
    # A₂(κ) = I₂(κ) / I₀(κ)
    # ----------------------------------------------------------

    A1 =
        _von_mises_A1.(κ)

    A2 =
        _von_mises_A2.(κ)


    # ----------------------------------------------------------
    # Fisher information in (μ, κ):
    #
    # I_μμ =
    #     κ A₁(κ)
    #
    # I_κκ =
    #     1/2 [1 + A₂(κ)] - A₁(κ)^2
    #
    # I_μκ =
    #     0
    # ----------------------------------------------------------

    Iμμ =
        κ .* A1

    Iκκ =
        0.5 .* (1.0 .+ A2) .-
        A1.^2


    # Small numerical protection against roundoff
    Iκκ =
        max.(Iκκ, 1e-12)


    # ----------------------------------------------------------
    # Transform Fisher information from (μ, κ)
    # to the latent predictors through the link derivatives
    # ----------------------------------------------------------

    wm =
        Iμμ .* dμ.^2

    wp =
        Iκκ .* dκ.^2


    # ----------------------------------------------------------
    # Build state-space Fisher blocks
    # ----------------------------------------------------------

    Hmm =
        XDiagX(Xm, wm)

    Hpp =
        XDiagX(Xp, wp)


    # Mean direction and concentration are Fisher-orthogonal
    Hmp =
        zeros(
            eltype(Hmm),
            size(Xm, 2),
            size(Xp, 2)
        )


    return [
        Hmm  Hmp
        Hmp' Hpp
    ]
end