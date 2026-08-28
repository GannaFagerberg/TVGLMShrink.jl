struct NBSuffStats{F1,F2,F3} <: AbstractObsTransform
    transform_obs::F1
    make_cond_moments::F2
    obs_dim::F3
end

# ============================================================
# Negative-binomial factorial-moment observation helpers
#
# Parameterisation:
#
# Y | μ, r ~ NB(μ, r)
#
# E[Y]   = μ
# Var[Y] = μ + μ²/r
#
# Observation transformation:
#
#     z(y) = [y, y(y-1)]
#
# This is a factorial-moment transformation, not a canonical
# sufficient-statistic representation.
# ============================================================


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

function nb_factorial_observation_averaged(
    y
)

    y_group = _nb_vector(
        y,
        "y"
    )

    g = length(y_group)

    mean_y = 0.0
    mean_factorial2 = 0.0

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

        mean_y += yi

        mean_factorial2 +=
            yi * (yi - 1.0)
    end

    return [
        mean_y / g,
        mean_factorial2 / g
    ]
end

function make_nb_factorial_statistics_adapters_averaged(
    raw_condMean,
    raw_condCov;
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    min_overdispersion::Real = 1e-10,
    relative_floor::Real = 1e-10
)

    # ==========================================================
    # Recover common NB parameters from raw observation moments
    # ==========================================================

    function nb_parameters_from_original_model_averaged(
        param,
        state,
        t
    )

        μ_raw =
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

        μ =
            _nb_vector(
                μ_raw,
                "raw_condMean output"
            )

        g = length(μ)

        variance_y =
            _nb_variance_vector(
                covariance_raw,
                g
            )

        # No-regression case:
        # common μ and r within the group
        μi =
            max(
                μ[1],
                min_mean
            )

        variance_i =
            variance_y[1]

        isfinite(μi) &&
            μi > 0 ||
            throw(DomainError(
                μi,
                "Conditional negative-binomial mean must be finite and positive."
            ))

        isfinite(variance_i) &&
            variance_i > 0 ||
            throw(DomainError(
                variance_i,
                "Conditional negative-binomial variance must be finite and positive."
            ))

        # NB requires Var(Y) > μ for finite r
        overdispersion =
            variance_i - μi

        overdispersion =
            max(
                overdispersion,
                min_overdispersion
            )

        # Var(Y) = μ + μ²/r
        r =
            μi^2 / overdispersion

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

        return μi, r, g
    end


    # ==========================================================
    # Conditional moments of averaged factorial statistics
    #
    # zbar =
    # [
    #   mean(Y_i),
    #   mean(Y_i(Y_i-1))
    # ]
    #
    # Dimension is always 2.
    # ==========================================================

    function condMoments_nb_factorial_averaged!(
        mean_z,
        R,
        param,
        state,
        t
    )

        μ, r, g =
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
        # Factorial moments
        # ------------------------------------------------------

        f1 =
            μ

        f2 =
            μ^2 *
            (1.0 + 1.0 / r)

        f3 =
            μ^3 *
            (1.0 + 1.0 / r) *
            (1.0 + 2.0 / r)

        f4 =
            μ^4 *
            (1.0 + 1.0 / r) *
            (1.0 + 2.0 / r) *
            (1.0 + 3.0 / r)


        # ------------------------------------------------------
        # Conditional mean
        # ------------------------------------------------------

        mean_z[1] =
            f1

        mean_z[2] =
            f2


        # ------------------------------------------------------
        # Conditional covariance for ONE observation
        #
        # Y² = (Y)_2 + Y
        #
        # Y (Y)_2 = (Y)_3 + 2(Y)_2
        #
        # (Y)_2² =
        #     (Y)_4 + 4(Y)_3 + 2(Y)_2
        # ------------------------------------------------------

        variance_y =
            f2 +
            f1 -
            f1^2

        covariance_y_factorial2 =
            f3 +
            2.0 * f2 -
            f1 * f2

        variance_factorial2 =
            f4 +
            4.0 * f3 +
            2.0 * f2 -
            f2^2


        # ------------------------------------------------------
        # Covariance of the AVERAGE
        # ------------------------------------------------------

        R[1, 1] =
            variance_y / g

        R[1, 2] =
            covariance_y_factorial2 / g

        R[2, 1] =
            R[1, 2]

        R[2, 2] =
            variance_factorial2 / g


        # Numerical safeguard
        R .= _make_spd(
            R;
            relative_floor = relative_floor
        )

        return nothing
    end

    return condMoments_nb_factorial_averaged!
end

struct NBFactorialStats{F1,F2,F3} <: AbstractObsTransform

    transform_obs::F1
    make_cond_moments::F2
    obs_dim::F3

end

function NBFactorialStatsAveraged(;
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    min_overdispersion::Real = 1e-10,
    relative_floor::Real = 1e-10
)

    return NBFactorialStats(

        # Observation transformation
        y ->
            nb_factorial_observation_averaged(
                y
            ),

        # Conditional moments
        (condMean, condCov) ->
            make_nb_factorial_statistics_adapters_averaged(
                condMean,
                condCov;
                min_mean = min_mean,
                min_dispersion = min_dispersion,
                min_overdispersion = min_overdispersion,
                relative_floor = relative_floor
            ),

        # Averaged transformation always has dimension 2
        nPerGroup -> 2
    )
end

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


# ==========================================================
# Fisher information for Negative Binomial
#
# Parameterization:
#
# Y ~ NB(μ, r)
#
# E[Y]   = μ
# Var[Y] = μ + μ²/r
#
# where r > 0 is the size / dispersion parameter.
# ==========================================================


# ----------------------------------------------------------
# Fisher information for r for one observation
# ----------------------------------------------------------
#
# I_rr =
#   sum_{j=0}^∞ P(Y > j) / (r+j)^2
#   - μ / (r(μ+r))
#
# The infinite sum is evaluated recursively.
# ----------------------------------------------------------

function fisher_nb_size(
    μ::Real,
    r::Real;
    tol::Real = 1e-12,
    maxiter::Int = 100_000
)

    μ = max(μ, 1e-12)
    r = max(r, 1e-8)

    # NB probability parameter
    p = r / (r + μ)
    q = μ / (r + μ)

    # P(Y = 0)
    py = p^r

    # CDF at zero
    cdf = py

    # j = 0 contribution:
    # P(Y > 0) / r²
    s = max(1.0 - cdf, 0.0) / r^2

    j = 0

    while (1.0 - cdf > tol) && (j < maxiter)

        j += 1

        # Recursive NB pmf:
        #
        # P(Y=j) =
        # P(Y=j-1) *
        # (j-1+r)/j *
        # μ/(μ+r)

        py *= ((j - 1 + r) / j) * q

        cdf += py

        survival = max(1.0 - cdf, 0.0)

        s += survival / (r + j)^2
    end

    Irr =
        s -
        μ / (r * (μ + r))

    # Protect against tiny negative values caused only
    # by numerical cancellation.
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

    Iμμ =
        r ./ (μ .* (μ .+ r))

    Irr =
        fisher_nb_size.(μ, r)


    # ------------------------------------------------------
    # Transform Fisher information to linear-predictor scale
    # ------------------------------------------------------

    wm =
        Iμμ .* dμ.^2

    wr =
        Irr .* dr.^2

    # Mean and size are Fisher-orthogonal
    wmr =
        zeros(eltype(wm), length(wm))


    # ------------------------------------------------------
    # Regression/state Fisher blocks
    # ------------------------------------------------------

    Hmm =
        XDiagX(Xm, wm)

    Hrr =
        XDiagX(Xr, wr)

    Hmr =
        XDiagZ(Xm, wmr, Xr)


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