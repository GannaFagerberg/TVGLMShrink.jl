


# ============================================================
# Negative-binomial transformed-observation helpers
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
#     z(y) = [y, log(1+y)]
#
# For a group:
#
#     z̄ =
#     [
#         mean(Y_i),
#         mean(log(1 + Y_i))
#     ]
#
# This is NOT a sufficient-statistic representation.
#
# The first component carries direct information about μ.
# The second component uses the whole count distribution
# while compressing large observations, and therefore
# provides additional information about r.
#
# PUBLIC/EXTERNAL NAMES ARE KEPT UNCHANGED so that the
# existing IPLF infrastructure requires no changes.
# ============================================================

struct NBSuffStats{F1,F2,F3} <: AbstractObsTransform
    transform_obs::F1
    make_cond_moments::F2
    obs_dim::F3
end

# ============================================================
# Helpers
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


# ============================================================
# Observed transformation
#
# NAME KEPT UNCHANGED:
#
#     nb_factorial_observation_averaged
#
# New transformation:
#
#     [
#         mean(Y),
#         mean(log(1+Y))
#     ]
# ============================================================

function nb_factorial_observation_averaged(
    y
)

    y_group =
        _nb_vector(
            y,
            "y"
        )

    g =
        length(y_group)

    mean_y =
        0.0

    mean_log1py =
        0.0


    @inbounds for i in eachindex(y_group)

        yi =
            y_group[i]


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


        mean_y +=
            yi

        mean_log1py +=
            log1p(yi)
    end


    return [
        mean_y / g,
        mean_log1py / g
    ]
end


# ============================================================
# Conditional moments involving log(1+Y)
#
# For Y ~ NB(μ,r), calculate numerically:
#
#     E[log(1+Y)]
#
#     E[log(1+Y)^2]
#
#     E[Y log(1+Y)]
#
# from the complete NB pmf.
#
# NB pmf recursion:
#
# P(Y=y) =
# P(Y=y-1) *
# ((y-1+r)/y) *
# μ/(μ+r)
#
# with
#
# P(Y=0) = (r/(r+μ))^r.
# ============================================================

function _nb_log1p_moments(
    μ::Real,
    r::Real;
    tol::Real = 1e-8,
    maxiter::Int = 10_000
)

    μ =
        max(
            float(μ),
            1e-12
        )

    r =
        max(
            float(r),
            1e-10
        )


    # --------------------------------------------------------
    # NB parameterisation
    # --------------------------------------------------------

    p =
        r / (r + μ)

    q =
        μ / (r + μ)


    # --------------------------------------------------------
    # P(Y = 0)
    #
    # log formulation is more stable than p^r
    # --------------------------------------------------------

    py =
        exp(
            r * log(p)
        )


    cumulative_probability =
        py


    # --------------------------------------------------------
    # Required moments
    #
    # At Y = 0:
    #
    # log(1+0) = 0,
    #
    # so the y=0 contribution to all three quantities is zero.
    # --------------------------------------------------------

    mean_log =
        0.0

    second_log =
        0.0

    mean_y_log =
        0.0


    y =
        0


    while (
        max(
            1.0 - cumulative_probability,
            0.0
        ) > tol
    ) && (
        y < maxiter
    )

        y +=
            1


        # ----------------------------------------------------
        # Recursive NB pmf
        # ----------------------------------------------------

        py *=
            ((y - 1 + r) / y) *
            q


        log_y =
            log1p(y)


        # ----------------------------------------------------
        # Accumulate moments
        # ----------------------------------------------------

        mean_log +=
            py *
            log_y


        second_log +=
            py *
            log_y^2


        mean_y_log +=
            py *
            y *
            log_y


        cumulative_probability +=
            py
    end


    return (
        mean_log,
        second_log,
        mean_y_log
    )
end


# ============================================================
# Construct transformed conditional moments
#
# NAME KEPT UNCHANGED:
#
#     make_nb_factorial_statistics_adapters_averaged
# ============================================================

function make_nb_factorial_statistics_adapters_averaged(
    raw_condMean,
    raw_condCov;
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    min_overdispersion::Real = 1e-10,
    relative_floor::Real = 1e-10
)

    # ========================================================
    # Recover common NB parameters from original observation
    # moments
    #
    # raw_condMean:
    #
    #     E[Y] = μ
    #
    # raw_condCov:
    #
    #     Var(Y) = μ + μ²/r
    #
    # Therefore
    #
    #     r = μ² / (Var(Y) - μ)
    #
    # Current implementation assumes common μ and r within
    # each group, matching the current no-regression model.
    # ========================================================

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


        g =
            length(μ)


        variance_y =
            _nb_variance_vector(
                covariance_raw,
                g
            )


        # ----------------------------------------------------
        # Common state within group
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Recover r
        #
        # Var(Y) - μ = μ²/r
        # ----------------------------------------------------

        overdispersion =
            variance_i -
            μi


        overdispersion =
            max(
                overdispersion,
                min_overdispersion
            )


        r =
            μi^2 /
            overdispersion


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


    # ========================================================
    # Conditional moments for
    #
    # z =
    #
    # [
    #     Y
    #     log(1+Y)
    # ]
    #
    #
    # E[z] =
    #
    # [
    #     μ
    #     E[log(1+Y)]
    # ]
    #
    #
    # Cov(z) =
    #
    # [
    #   Var(Y)                  Cov(Y,log(1+Y))
    #
    #   Cov(Y,log(1+Y))         Var(log(1+Y))
    # ]
    #
    # where the non-polynomial moments are evaluated using
    # the full NB probability mass function.
    # ========================================================

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


        # ----------------------------------------------------
        # Moments involving log(1+Y)
        # ----------------------------------------------------

        mean_log,
        second_log,
        mean_y_log =
            _nb_log1p_moments(
                μ,
                r
            )


        # ====================================================
        # Conditional mean
        # ====================================================

        mean_z[1] =
            μ


        mean_z[2] =
            mean_log


        # ====================================================
        # Conditional covariance for ONE NB observation
        # ====================================================

        variance_y =
            μ +
            μ^2 / r


        variance_log =
            second_log -
            mean_log^2


        covariance_y_log =
            mean_y_log -
            μ *
            mean_log


        # Numerical protection against tiny negative
        # round-off in Var(log(1+Y)).
        variance_log =
            max(
                variance_log,
                0.0
            )


        # ====================================================
        # Covariance of group AVERAGE
        #
        # For conditionally iid observations:
        #
        # Var(z̄) = Var(z) / g
        # ====================================================

        R[1, 1] =
            variance_y /
            g


        R[1, 2] =
            covariance_y_log /
            g


        R[2, 1] =
            R[1, 2]


        R[2, 2] =
            variance_log /
            g


        # ----------------------------------------------------
        # SPD protection
        # ----------------------------------------------------

        R .=
            _make_spd(
                R;
                relative_floor = relative_floor
            )


        return nothing
    end


    return condMoments_nb_factorial_averaged!
end


# ============================================================
# Constructor
#
# NAME AND ARGUMENTS KEPT UNCHANGED:
#
#     NBFactorialStatsAveraged()
# ============================================================

function NBFactorialStatsAveraged(;
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    min_overdispersion::Real = 1e-10,
    relative_floor::Real = 1e-10
)

    return NBSuffStats(
    y -> nb_factorial_observation_averaged(y),

    (condMean, condCov) ->
        make_nb_factorial_statistics_adapters_averaged(
            condMean,
            condCov;
            min_mean = min_mean,
            min_dispersion = min_dispersion,
            min_overdispersion = min_overdispersion,
            relative_floor = relative_floor
        ),

    nPerGroup -> 2
    )
end


# ============================================================
# Prepare observation transformation
#
# INTERFACE UNCHANGED.
# ============================================================

function prepare_observation_transform(
    transform::NBSuffStats,
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

    μ = max(float(μ), 1e-12)
    r = max(float(r), 1e-8)

    p = r / (r + μ)
    q = μ / (r + μ)

    # P(Y = 0)
    py = exp(r * log(p))
    cdf = py

    # Score wrt r for y = 0
    #
    # s_r(y) =
    # ψ(r+y) - ψ(r)
    # - log(1 + μ/r)
    # + (μ-y)/(r+μ)
    #
    # For y=0, the digamma difference is zero.
    score_r =
        -log1p(μ / r) +
        μ / (r + μ)

    Irr =
        py * score_r^2

    # ψ(r+y)-ψ(r)
    # = sum_{k=0}^{y-1} 1/(r+k)
    digamma_diff =
        0.0

    y =
        0

    while (
        max(1.0 - cdf, 0.0) > tol
    ) && (
        y < maxiter
    )

        y += 1

        # Recursive NB probability
        py *=
            ((y - 1 + r) / y) *
            q

        # Update ψ(r+y)-ψ(r)
        digamma_diff +=
            1.0 / (r + y - 1)

        score_r =
            digamma_diff -
            log1p(μ / r) +
            (μ - y) / (r + μ)

        Irr +=
            py * score_r^2

        cdf +=
            py
    end

    return max(Irr, 0.0)
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


# ============================================================
# GROUPED / REGRESSION VERSION
#
# Transform:
#
# z =
# [
#   Y_1, ..., Y_g,
#   log(1+Y_1), ..., log(1+Y_g)
# ]
#
# Dimension = 2g
# ============================================================

function nb_factorial_observation_grouped(y)

    y_group = _nb_vector(y, "y")
    g = length(y_group)

    z = Vector{Float64}(undef, 2g)

    @inbounds for i in 1:g

        yi = y_group[i]

        z[i] = yi
        z[g + i] = log1p(yi)
    end

    return z
end

# ============================================================
# Conditional moments for grouped observations
# ============================================================
function make_nb_factorial_statistics_adapters_grouped(
    raw_condMean,
    raw_condCov;
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    relative_floor::Real = 1e-10
)

    function condMoments_nb_grouped!(
        mean_z,
        R,
        param,
        state,
        t
    )

        # ----------------------------------------------------
        # Construct μ_i and r_i DIRECTLY from the model state
        # ----------------------------------------------------

        ημ =
            param.Z[1][t] *
            state[param.Zidx[1]]

        ηr =
            param.Z[2][t] *
            state[param.Zidx[2]]

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

        μ = _nb_vector(
            μ,
            "NB mean"
        )

        r = _nb_vector(
            r,
            "NB size"
        )

        g = length(μ)

        # Precision may be intercept-only, in which case
        # repeat the common r across observations.
        if length(r) == 1 && g > 1
            r = fill(r[1], g)
        end

        length(r) == g ||
            throw(DimensionMismatch(
                "length(μ)=$g but length(r)=$(length(r))."
            ))

        μ = max.(μ, min_mean)
        r = max.(r, min_dispersion)

        length(mean_z) == 2g ||
            throw(DimensionMismatch(
                "mean_z length $(length(mean_z)); expected $(2g)."
            ))

        size(R) == (2g, 2g) ||
            throw(DimensionMismatch(
                "R size $(size(R)); expected ($(2g),$(2g))."
            ))

        fill!(R, 0.0)

        # ----------------------------------------------------
        # One [Y_i, log(1+Y_i)] block per observation
        # ----------------------------------------------------


    ##################LOOOPP
        @inbounds for i in 1:g

            μi = μ[i]
            ri = r[i]

            # --------------------------------------------------------
            # Cached numerical NB moments
            # --------------------------------------------------------

            mean_log,
            second_log,
            mean_y_log =
                _nb_log1p_moments(
                    μi,
                    ri;
                    tol = 1e-8,
                    maxiter = 10_000
                )

            # --------------------------------------------------------
            # Conditional covariance components
            # --------------------------------------------------------

            variance_y =
                μi +
                μi^2 / ri

            variance_log =
                second_log -
                mean_log^2

            covariance_y_log =
                mean_y_log -
                μi * mean_log


            # Numerical protection
            variance_log =
                max(
                    variance_log,
                    eps(Float64)
                )


            # --------------------------------------------------------
            # Location in transformed observation
            # --------------------------------------------------------

            j = g + i


            # --------------------------------------------------------
            # Conditional mean
            # --------------------------------------------------------

            mean_z[i] = μi
            mean_z[j] = mean_log


            # --------------------------------------------------------
            # Cheap SPD protection for the 2×2 covariance block
            #
            # R_i =
            #
            # [ variance_y       covariance_y_log
            #   covariance_y_log variance_log      ]
            # --------------------------------------------------------

            v1 = variance_y
            v2 = variance_log
            c  = covariance_y_log

            detR =
                v1 * v2 -
                c^2

            if !isfinite(detR) || detR <= 0.0

                δ =
                    relative_floor *
                    max(
                        v1,
                        v2,
                        1.0
                    )

                v1 += δ
                v2 += δ

                # In the unlikely case that roundoff is stronger,
                # enforce |c| < sqrt(v1*v2)
                cmax =
                    sqrt(v1 * v2) *
                    (1.0 - 1e-10)

                c =
                    clamp(
                        c,
                        -cmax,
                        cmax
                    )
            end


            # --------------------------------------------------------
            # Insert block into full covariance matrix
            # --------------------------------------------------------

            R[i, i] = v1
            R[i, j] = c

            R[j, i] = c
            R[j, j] = v2
        end

        return nothing
    end

    return condMoments_nb_grouped!
end

# ============================================================
# Grouped constructor
# ============================================================

function NBFactorialStatsGrouped(;
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    relative_floor::Real = 1e-10
)

    return NBSuffStats(

        y ->
            nb_factorial_observation_grouped(y),

        (condMean, condCov) ->
            make_nb_factorial_statistics_adapters_grouped(
                condMean,
                condCov;
                min_mean = min_mean,
                min_dispersion = min_dispersion,
                relative_floor = relative_floor
            ),

        nPerGroup -> 2 * nPerGroup
    )
end