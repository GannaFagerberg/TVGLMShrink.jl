# ============================================================
# Negative-binomial transformed-observation helpers
# using a closed-form shifted-Gamma surrogate
#
# Parameterisation:
#
#     Y | mu, r ~ NB(mu, r)
#
#     E[Y]   = mu
#     Var[Y] = mu + mu^2/r
#
# Observation transformation, with shift = 1 by default:
#
#     z(y) = [log(y + shift), y]
#
# For a group of observations:
#
#     zbar =
#     [
#         mean(log(Y_i + shift)),
#         mean(Y_i)
#     ]
#
# The public names are intentionally unchanged so this block can
# replace the previous NB transformed-observation implementation.
# ============================================================

using SpecialFunctions: digamma, trigamma




function _nb_vector(value, name::AbstractString)

    if value isa Real

        return [float(value)]

    elseif value isa AbstractVector

        return vec(float.(value))

    else

        throw(ArgumentError(
            "$name must be a scalar or vector; got $(typeof(value))."
        ))
    end
end


function _nb_variance_vector(value, group_size::Int)

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


# ============================================================
# Transform observed NB data
#
# Function name kept unchanged for compatibility.
#
# Returns
#
#     [mean(log(Y_i + shift)), mean(Y_i)]
# ============================================================

function nb_factorial_observation_averaged(
    y;
    shift::Real = 1.0
)

    isfinite(shift) && shift > 0 ||
        throw(ArgumentError(
            "shift must be finite and strictly positive."
        ))

    y_group = _nb_vector(
        y,
        "y"
    )

    g = length(y_group)

    mean_log_shifted_y = 0.0
    mean_y = 0.0

    @inbounds for i in eachindex(y_group)

        yi = y_group[i]

        isfinite(yi) ||
            throw(DomainError(
                yi,
                "Negative-binomial observations must be finite."
            ))

        yi >= 0 ||
            throw(DomainError(
                yi,
                "Negative-binomial observations must be non-negative."
            ))

        isinteger(yi) ||
            throw(DomainError(
                yi,
                "Negative-binomial observations must be integer-valued."
            ))

        mean_log_shifted_y += log(yi + shift)
        mean_y += yi
    end

    return [
        mean_log_shifted_y / g,
        mean_y / g
    ]
end


# ============================================================
# Construct conditional moments of transformed observation
#
# Function name kept unchanged for compatibility.
# ============================================================

function make_nb_factorial_statistics_adapters_averaged(
    raw_condMean,
    raw_condCov;
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    min_overdispersion::Real = 1e-10,
    relative_floor::Real = 1e-10,
    shift::Real = 1.0
)

    isfinite(shift) && shift > 0 ||
        throw(ArgumentError(
            "shift must be finite and strictly positive."
        ))


    # ==========================================================
    # Recover common NB parameters from original model moments
    #
    # raw_condMean gives mu.
    #
    # raw_condCov gives
    #
    #     Var(Y) = mu + mu^2/r,
    #
    # hence
    #
    #     r = mu^2 / (Var(Y) - mu).
    #
    # The current implementation assumes common mu and r within
    # each group, as in the no-regression simulation.
    # ==========================================================

    function nb_parameters_from_original_model_averaged(
        param,
        state,
        t
    )

        mu_raw =
            raw_condMean(
                param,
                state,
                t
            )

        covariance_raw =
            raw_condCov(
                param,
                state,
                t
            )

        mu =
            _nb_vector(
                mu_raw,
                "raw_condMean output"
            )

        g = length(mu)

        variance_y =
            _nb_variance_vector(
                covariance_raw,
                g
            )


        # ------------------------------------------------------
        # No-regression/common-state case
        # ------------------------------------------------------

        mui =
            max(
                mu[1],
                min_mean
            )

        variance_i =
            variance_y[1]


        isfinite(mui) &&
            mui > 0 ||
            throw(DomainError(
                mui,
                "Conditional negative-binomial mean must be finite and positive."
            ))


        isfinite(variance_i) &&
            variance_i > 0 ||
            throw(DomainError(
                variance_i,
                "Conditional negative-binomial variance must be finite and positive."
            ))


        # ------------------------------------------------------
        # Recover NB size parameter r
        #
        # Var(Y) - mu = mu^2/r
        # ------------------------------------------------------

        overdispersion =
            variance_i - mui

        overdispersion =
            max(
                overdispersion,
                min_overdispersion
            )

        r =
            mui^2 / overdispersion


        isfinite(r) &&
            r > 0 ||
            throw(DomainError(
                r,
                "The conditional moments imply invalid NB dispersion."
            ))


        r =
            max(
                r,
                min_dispersion
            )


        return mui, r, g
    end


    # ==========================================================
    # Conditional moments from the shifted-Gamma surrogate
    #
    # Let
    #
    #     W = Y + shift,
    #     m = E[W] = mu + shift,
    #     v = Var(W) = Var(Y) = mu + mu^2/r.
    #
    # Match W to a Gamma(shape, rate) distribution:
    #
    #     shape = m^2/v,
    #     rate  = m/v.
    #
    # For z = [log(W), Y], the surrogate moments are
    #
    #     E[log(W)]       = digamma(shape) - log(rate),
    #     E[Y]            = mu,
    #     Var(log(W))     = trigamma(shape),
    #     Cov(log(W), Y)  = 1/rate = v/m,
    #     Var(Y)          = v.
    # ==========================================================

    function condMoments_nb_factorial_averaged!(
        mean_z,
        R,
        param,
        state,
        t
    )

        mu, r, g =
            nb_parameters_from_original_model_averaged(
                param,
                state,
                t
            )


        length(mean_z) == 2 ||
            throw(DimensionMismatch(
                "mean_z has length $(length(mean_z)), expected 2."
            ))


        size(R) == (2, 2) ||
            throw(DimensionMismatch(
                "R has size $(size(R)), expected (2, 2)."
            ))


        # ------------------------------------------------------
        # NB mean and variance
        # ------------------------------------------------------

        variance_y =
            mu +
            mu^2 / r


        # ------------------------------------------------------
        # Moment-matched shifted-Gamma parameters
        # ------------------------------------------------------

        shifted_mean =
            mu + shift

        gamma_shape =
            shifted_mean^2 / variance_y

        gamma_rate =
            shifted_mean / variance_y


        isfinite(gamma_shape) &&
            gamma_shape > 0 ||
            throw(DomainError(
                gamma_shape,
                "The shifted-Gamma surrogate has an invalid shape."
            ))


        isfinite(gamma_rate) &&
            gamma_rate > 0 ||
            throw(DomainError(
                gamma_rate,
                "The shifted-Gamma surrogate has an invalid rate."
            ))


        # ======================================================
        # Conditional mean of transformed observation
        # ======================================================

        mean_z[1] =
            digamma(gamma_shape) -
            log(gamma_rate)

        mean_z[2] =
            mu


        # ======================================================
        # Conditional covariance for one observation
        # ======================================================

        variance_log_shifted_y =
            trigamma(gamma_shape)

        covariance_log_shifted_y_y =
            1.0 / gamma_rate


        # ======================================================
        # Covariance of group average
        #
        # If observations are conditionally independent and
        # share the same mu and r:
        #
        #     Var(zbar) = Var(z)/g.
        # ======================================================

        R[1, 1] =
            variance_log_shifted_y / g

        R[1, 2] =
            covariance_log_shifted_y_y / g

        R[2, 1] =
            R[1, 2]

        R[2, 2] =
            variance_y / g


        # ------------------------------------------------------
        # Numerical safeguard
        # ------------------------------------------------------

        R .= _make_spd(
            R;
            relative_floor = relative_floor
        )


        return nothing
    end


    return condMoments_nb_factorial_averaged!
end

# ============================================================
# Observation-transform object
#
# Name kept unchanged so existing scripts continue to use
#
#     NBFactorialStatsAveraged()
# ============================================================

struct NBFactorialStats{F1,F2,F3} <: AbstractObsTransform

    transform_obs::F1
    make_cond_moments::F2
    obs_dim::F3

end


function NBFactorialStatsAveraged(;
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    min_overdispersion::Real = 1e-10,
    relative_floor::Real = 1e-10,
    shift::Real = 1.0
)

    return NBFactorialStats(

        # ------------------------------------------------------
        # Observation transformation
        #
        # y -> [mean(log(y + shift)), mean(y)]
        # ------------------------------------------------------

        y ->
            nb_factorial_observation_averaged(
                y;
                shift = shift
            ),


        # ------------------------------------------------------
        # Conditional shifted-Gamma surrogate moments
        # ------------------------------------------------------

        (condMean, condCov) ->
            make_nb_factorial_statistics_adapters_averaged(
                condMean,
                condCov;
                min_mean = min_mean,
                min_dispersion = min_dispersion,
                min_overdispersion = min_overdispersion,
                relative_floor = relative_floor,
                shift = shift
            ),


        # ------------------------------------------------------
        # Transformed observation dimension
        # ------------------------------------------------------

        nPerGroup -> 2
    )
end


# ============================================================
# Prepare transformed observations for SLR/IPLF
#
# Interface unchanged.
# ============================================================

function prepare_observation_transform(
    transform::NBFactorialStats,
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
        transform.obs_dim(
            nPerGroup
        )


    return (
        Y = Y_transformed,
        condMoments = condMoments,
        nObs = nObs
    )
end


############
### Fisher
############

function fisher_nb_size(
    μ,
    r;
    tol = 1e-12,
    maxiter::Int = 100_000
)

    μ = max(μ, 1e-12)
    r = max(r, 1e-8)

    p = r / (r + μ)
    q = μ / (r + μ)

    py = p^r
    cdf = py

    s = max(1.0 - cdf, 0.0) / r^2

    j = 0

    while (1.0 - cdf > tol) && (j < maxiter)

        j += 1

        py *= ((j - 1 + r) / j) * q
        cdf += py

        survival = max(1.0 - cdf, 0.0)

        s += survival / (r + j)^2
    end

    Irr =s -μ / (r * (μ + r))

    return max(Irr, eps(Float64))
end

function fisher_nb_blocks(
    Xm,
    Xr,
    βm,
    βr,
    linkm::Link,
    linkr::Link
)

    # ------------------------------------------------------
    # Linear predictors
    # ------------------------------------------------------

    ηm = Xm * βm
    ηr = Xr * βr


    # ------------------------------------------------------
    # Transform to μ and r
    # ------------------------------------------------------

    μ = GLM.linkinv.(linkm, ηm)
    r = linkinv.(linkr, ηr)

    μ = max.(μ, 1e-12)
    r = max.(r, 1e-8)


    # ------------------------------------------------------
    # Derivatives of inverse links
    # ------------------------------------------------------

    dμ = GLM.mueta.(linkm, ηm)
    dr = mueta.(linkr, ηr)

    # ------------------------------------------------------
    # Fisher information in (μ,r)
    #
    # I_μμ = r / [μ(μ+r)]
    #
    # I_μr = 0
    #
    # I_rr computed numerically
    # ------------------------------------------------------

    Iμμ =r ./ (μ .* (μ .+ r))

    Irr =fisher_nb_size.(μ, r)


    # ------------------------------------------------------
    # Transform Fisher information to linear-predictor scale
    # ------------------------------------------------------

    wm =Iμμ .* dμ.^2

    wr = Irr .* dr.^2

    # Mean and size are Fisher-orthogonal
    wmr =zeros(eltype(wm), length(wm))

    # ------------------------------------------------------
    # Regression/state Fisher blocks
    # ------------------------------------------------------

    Hmm =XDiagX(Xm, wm)

    Hrr =XDiagX(Xr, wr)

    Hmr =XDiagZ(Xm, wmr, Xr)


    return [
        Hmm    Hmr
        Hmr'   Hrr
    ]
end

# Function that computes the Fisher info
# for all observations

function FisherInfoNB(param, μ, t)

    pm = size(param.X[1], 2)

    return fisher_nb_blocks(
        param.X[1],
        param.X[2],
        μ[1:pm],
        μ[(pm + 1):end],
        param.link[1],
        param.link[2]
    )
end

# ============================================================
# GROUPED shifted-Gamma surrogate for NB regression
#
# z =
# [
#   log(Y_1 + shift), ..., log(Y_g + shift),
#   Y_1,               ..., Y_g
# ]
#
# Dimension = 2g
# ============================================================


function nb_factorial_observation_grouped(
    y;
    shift::Real = 1.0
)

    isfinite(shift) && shift > 0 ||
        throw(ArgumentError(
            "shift must be finite and strictly positive."
        ))

    y_group =
        _nb_vector(
            y,
            "y"
        )

    g = length(y_group)

    z =
        Vector{Float64}(
            undef,
            2 * g
        )

    @inbounds for i in 1:g

        yi = y_group[i]

        isfinite(yi) ||
            throw(DomainError(
                yi,
                "Negative-binomial observations must be finite."
            ))

        yi >= 0 ||
            throw(DomainError(
                yi,
                "Negative-binomial observations must be non-negative."
            ))

        isinteger(yi) ||
            throw(DomainError(
                yi,
                "Negative-binomial observations must be integer-valued."
            ))

        # First block: log(Y_i + shift)
        z[i] =
            log(yi + shift)

        # Second block: Y_i
        z[g + i] =
            yi
    end

    return z
end

function make_nb_factorial_statistics_adapters_grouped(
    raw_condMean,
    raw_condCov;
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    relative_floor::Real = 1e-10,
    shift::Real = 1.0
)

    isfinite(shift) && shift > 0 ||
        throw(ArgumentError(
            "shift must be finite and strictly positive."
        ))


    # ========================================================
    # Construct observation-specific NB parameters directly
    # from the regression state.
    #
    # This is preferable to recovering r from condCov in the
    # regression case.
    # ========================================================

    function nb_parameters_from_original_model_grouped(
        param,
        state,
        t
    )

        # ----------------------------------------------------
        # Linear predictors
        # ----------------------------------------------------

        ημ =
            param.Z[1][t] *
            state[param.Zidx[1]]

        ηr =
            param.Z[2][t] *
            state[param.Zidx[2]]


        # ----------------------------------------------------
        # Parameter scale
        # ----------------------------------------------------

        μ =
            linkinv.(
                Ref(param.link[1]),
                ημ
            )

        r =
            linkinv.(
                Ref(param.link[2]),
                ηr
            )


        μ =
            _nb_vector(
                μ,
                "Negative-binomial mean"
            )

        r =
            _nb_vector(
                r,
                "Negative-binomial size"
            )


        # ----------------------------------------------------
        # Regression may occur only in one parameter.
        #
        # E.g. μ_i varies with x_i but r is common.
        # ----------------------------------------------------

        g =
            max(
                length(μ),
                length(r)
            )


        if length(μ) == 1 && g > 1

            μ =
                fill(
                    μ[1],
                    g
                )

        elseif length(μ) != g

            throw(DimensionMismatch(
                "NB mean has length $(length(μ)); expected 1 or $g."
            ))
        end


        if length(r) == 1 && g > 1

            r =
                fill(
                    r[1],
                    g
                )

        elseif length(r) != g

            throw(DimensionMismatch(
                "NB size has length $(length(r)); expected 1 or $g."
            ))
        end


        # ----------------------------------------------------
        # Numerical safeguards
        # ----------------------------------------------------

        μ =
            max.(
                μ,
                min_mean
            )

        r =
            max.(
                r,
                min_dispersion
            )


        all(isfinite, μ) ||
            throw(DomainError(
                μ,
                "Non-finite negative-binomial mean."
            ))

        all(isfinite, r) ||
            throw(DomainError(
                r,
                "Non-finite negative-binomial size."
            ))


        return μ, r
    end


    # ========================================================
    # Conditional moments of the grouped transformed obs.
    # ========================================================

    function condMoments_nb_factorial_grouped!(
        mean_z,
        R,
        param,
        state,
        t
    )

        μ, r =
            nb_parameters_from_original_model_grouped(
                param,
                state,
                t
            )

        g =
            length(μ)


        length(mean_z) == 2g ||
            throw(DimensionMismatch(
                "mean_z has length $(length(mean_z)), " *
                "expected $(2g)."
            ))

        size(R) == (2g, 2g) ||
            throw(DimensionMismatch(
                "R has size $(size(R)), " *
                "expected ($(2g), $(2g))."
            ))


        # Different raw observations are conditionally
        # independent given the state.
        fill!(
            R,
            zero(eltype(R))
        )


        # ====================================================
        # One shifted-Gamma surrogate per observation
        # ====================================================

        @inbounds for i in 1:g

            μi =
                μ[i]

            ri =
                r[i]


            # ------------------------------------------------
            # Original NB moments
            #
            # Var(Y_i) = μ_i + μ_i²/r_i
            # ------------------------------------------------

            variance_y =
                μi +
                μi^2 / ri


            # ------------------------------------------------
            # Shifted variable
            #
            # W_i = Y_i + shift
            #
            # E[W_i]   = μ_i + shift
            # Var[W_i] = Var[Y_i]
            # ------------------------------------------------

            shifted_mean =
                μi +
                shift


            # ------------------------------------------------
            # Moment-matched Gamma surrogate
            #
            # W_i* ~ Gamma(shape_i, rate_i)
            #
            # shape_i = m_i²/v_i
            # rate_i  = m_i/v_i
            # ------------------------------------------------

            gamma_shape =
                shifted_mean^2 /
                variance_y

            gamma_rate =
                shifted_mean /
                variance_y


            isfinite(gamma_shape) &&
                gamma_shape > 0 ||
                throw(DomainError(
                    gamma_shape,
                    "Invalid shifted-Gamma shape."
                ))

            isfinite(gamma_rate) &&
                gamma_rate > 0 ||
                throw(DomainError(
                    gamma_rate,
                    "Invalid shifted-Gamma rate."
                ))


            # ------------------------------------------------
            # Surrogate conditional moments
            # ------------------------------------------------

            mean_log =
                digamma(gamma_shape) -
                log(gamma_rate)

            variance_log =
                trigamma(gamma_shape)

            covariance_log_y =
                1.0 /
                gamma_rate


            # ------------------------------------------------
            # Position in full 2g-dimensional observation
            # ------------------------------------------------

            j =
                g + i


            # ------------------------------------------------
            # Conditional mean
            #
            # [
            #   E log(Y_i + shift)
            #   E Y_i
            # ]
            # ------------------------------------------------

            mean_z[i] =
                mean_log

            mean_z[j] =
                μi


            # ------------------------------------------------
            # Conditional covariance block
            #
            # [
            #   Var(log(W_i))     Cov(log(W_i),Y_i)
            #   Cov(log(W_i),Y_i) Var(Y_i)
            # ]
            #
            # No division by g here because observations
            # are NOT averaged.
            # ------------------------------------------------

            Ri = [
                variance_log       covariance_log_y
                covariance_log_y   variance_y
            ]


            Ri =
                _make_spd(
                    Ri;
                    relative_floor =
                        relative_floor
                )


            R[i, i] =
                Ri[1, 1]

            R[i, j] =
                Ri[1, 2]

            R[j, i] =
                Ri[2, 1]

            R[j, j] =
                Ri[2, 2]
        end


        return nothing
    end


    return condMoments_nb_factorial_grouped!
end

function NBFactorialStatsGrouped(;
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    relative_floor::Real = 1e-10,
    shift::Real = 1.0
)

    return NBFactorialStats(

        # ----------------------------------------------------
        # Raw observation transformation
        # ----------------------------------------------------

        y ->
            nb_factorial_observation_grouped(
                y;
                shift = shift
            ),


        # ----------------------------------------------------
        # Shifted-Gamma surrogate conditional moments
        # ----------------------------------------------------

        (condMean, condCov) ->
            make_nb_factorial_statistics_adapters_grouped(
                condMean,
                condCov;
                min_mean =
                    min_mean,
                min_dispersion =
                    min_dispersion,
                relative_floor =
                    relative_floor,
                shift =
                    shift
            ),


        # ----------------------------------------------------
        # Two transformed quantities for each observation
        # ----------------------------------------------------

        nPerGroup ->
            2 * nPerGroup
    )
end

