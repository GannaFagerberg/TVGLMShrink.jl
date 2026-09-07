# ============================================================
# Gamma sufficient-statistic helpers
#
# Parameterisation:
#
# Y | μ, κ ~ Gamma(shape = κ, scale = μ/κ)
#
# E[Y]   = μ
# Var[Y] = μ²/κ
#
# Sufficient statistics:
#
#     z(y) = [log(y), y]
#
# ============================================================


function _gamma_vector(value, name::AbstractString)

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

function _gamma_variance_vector(value, group_size::Int)

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


function gamma_sufficient_observation_averaged(
    y;
    clip::Bool = false,
    boundary::Real = 1e-12
)

    y_group = _gamma_vector(
        y,
        "y"
    )

    g = length(y_group)

    mean_log_y = 0.0
    mean_y     = 0.0

    @inbounds for i in eachindex(y_group)

        yi = y_group[i]

        isfinite(yi) ||
            throw(DomainError(
                yi,
                "Gamma observations must be finite."
            ))

        if yi <= 0
            if clip
                yi = max(
                    yi,
                    boundary
                )
            else
                throw(DomainError(
                    yi,
                    "Gamma observations must be strictly positive."
                ))
            end
        end

        mean_log_y += log(yi)
        mean_y     += yi
    end

    return [
        mean_log_y / g,
        mean_y / g
    ]
end


function make_gamma_sufficient_statistics_adapters_averaged(
    raw_condMean,
    raw_condCov;
    min_mean::Real = 1e-10,
    min_precision::Real = 1e-10
)

    # ==========================================================
    # Recover common Gamma parameters from raw observation moments
    # ==========================================================

    function gamma_parameters_from_original_model_averaged(
        param,
        state,
        t
    )

        μ_raw =raw_condMean(param,state,t)

        covariance_raw =raw_condCov(param,state,t)

        μ = _gamma_vector(
            μ_raw,
            "raw_condMean output"
        )

        g = length(μ)

        variance_y =_gamma_variance_vector(covariance_raw,g)

        # No-regression case:
        # all observations in the group share the same μ and κ
        μi =max(μ[1],min_mean)

        variance_i =
            variance_y[1]

        isfinite(μi) &&
            μi > 0 ||
            throw(DomainError(
                μi,
                "Conditional Gamma mean must be finite and positive."
            ))

        isfinite(variance_i) &&
            variance_i > 0 ||
            throw(DomainError(
                variance_i,
                "Conditional Gamma variance must be finite and positive."
            ))

        # Var(Y|x) = μ² / κ
        κ =μi^2 / variance_i

        isfinite(κ) &&
            κ > 0 ||
            throw(DomainError(
                κ,
                "The conditional moments imply non-positive Gamma precision."
            ))

        κ =max(κ,min_precision)

        if any(μ .< 1e-6) ||
        any(κ .< 1e-2) ||
        any(κ .> 1e3) ||
        any(.!isfinite.(μ)) ||
        any(.!isfinite.(κ))

            @show t
            @show extrema(ημ)
            @show extrema(μ)
            @show extrema(ηκ)
            @show extrema(κ)
        end

        return μi, κ, g
    end


    # ==========================================================
    # Averaged sufficient-statistic moments
    #
    # z_bar =
    # [
    #   (1/g) sum_i log(y_i),
    #   (1/g) sum_i y_i
    # ]
    #
    # Dimension is always 2.
    # ==========================================================

    function condMoments_gamma_sufficient_averaged!(
        mean_z,
        R,
        param,
        state,
        t
    )

        μ, κ, g =
            gamma_parameters_from_original_model_averaged(
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
        # Conditional mean
        # ------------------------------------------------------

        mean_z[1] =digamma(κ) +log(μ) -log(κ)
        mean_z[2] =μ

        # ------------------------------------------------------
        # Conditional covariance of the average
        # ------------------------------------------------------

        R[1, 1] =trigamma(κ) / g

        R[2, 2] =(μ^2 / κ) / g

        R[1, 2] =(μ / κ) / g

        R[2, 1] = R[1, 2]

        # Numerical SPD safeguard
        R .= _make_spd(
            R;
            relative_floor = 1e-10
        )

        return nothing
    end

    return condMoments_gamma_sufficient_averaged!
end


struct GammaSuffStats{F1,F2,F3} <: AbstractObsTransform
    transform_obs::F1
    make_cond_moments::F2
    obs_dim::F3
end

function GammaSuffStatsAveraged(;
    clip::Bool = false,
    boundary::Real = 1e-12,
    min_mean::Real = 1e-10,
    min_precision::Real = 1e-10
)

    return GammaSuffStats(

        # Observation transformation
        y ->
            gamma_sufficient_observation_averaged(
                y;
                clip = clip,
                boundary = boundary
            ),

        # Conditional moments of transformed observation
        (condMean, condCov) ->
            make_gamma_sufficient_statistics_adapters_averaged(
                condMean,
                condCov;
                min_mean = min_mean,
                min_precision = min_precision
            ),

        # Dimension of transformed observation
        nPerGroup -> 2
    )
end

function prepare_observation_transform(
    transform::GammaSuffStats,
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

function fisher_gamma_blocks(
    Xμ,
    Xκ,
    βμ,
    βκ,
    linkμ,
    linkκ
)

    # Linear predictors
    ημ = Xμ * βμ
    ηκ = Xκ * βκ

    ημ = clamp.(ημ, -20.0, 20.0)
    ηκ = clamp.(ηκ, -20.0, 20.0)

    # Gamma mean and precision
    μ = linkinv.(Ref(linkμ), ημ)
    κ = linkinv.(Ref(linkκ), ηκ)

    μ = max.(μ, 1e-8)
    κ = max.(κ, 1e-3)

    # Derivatives dμ/dημ and dκ/dηκ
    dμ = mueta.(Ref(linkμ), ημ)
    dκ = mueta.(Ref(linkκ), ηκ)

    # Fisher weights on predictor scale
    wμ =(κ ./ μ.^2) .* dμ.^2

    wκ =(trigamma.(κ) .- 1.0 ./ κ) .* dκ.^2

    # Mean and precision are Fisher-orthogonal
    Fμμ =Xμ' * Diagonal(vec(wμ)) * Xμ

    Fκκ =Xκ' * Diagonal(vec(wκ)) * Xκ

    p = size(Xμ, 2)
    q = size(Xκ, 2)

    F = zeros(
        promote_type(eltype(Fμμ), eltype(Fκκ)),
        p + q,
        p + q
    )

    F[1:p, 1:p] .= Fμμ
    F[(p + 1):(p + q), (p + 1):(p + q)] .= Fκκ

    return F
end

function FisherInfoGamma(param, μ, t)

    p = size(param.X[1], 2)

    return fisher_gamma_blocks(
        param.X[1],
        param.X[2],
        μ[1:p],
        μ[(p + 1):end],
        param.link[1],
        param.link[2]
    )
end

# ============================================================
# Gamma sufficient statistics -- GROUPED / REGRESSION VERSION
#
# Y_i | μ_i, κ_i ~ Gamma(shape = κ_i, scale = μ_i/κ_i)
#
# z =
# [
#   log(y_1)
#   ...
#   log(y_g)
#   y_1
#   ...
#   y_g
# ]
#
# Dimension = 2g
#
# This version allows μ_i and/or κ_i to vary within a group,
# e.g. because of regression covariates.
# ============================================================


function gamma_sufficient_observation_grouped(
    y;
    clip::Bool = false,
    boundary::Real = 1e-12
)

    y_group = _gamma_vector(
        y,
        "y"
    )

    g = length(y_group)

    z = Vector{Float64}(undef, 2 * g)

    @inbounds for i in 1:g

        yi = y_group[i]

        isfinite(yi) ||
            throw(DomainError(
                yi,
                "Gamma observations must be finite."
            ))

        if yi <= 0
            if clip
                yi = max(
                    yi,
                    boundary
                )
            else
                throw(DomainError(
                    yi,
                    "Gamma observations must be strictly positive."
                ))
            end
        end

        # First block: log(y_i)
        z[i] = log(yi)

        # Second block: y_i
        z[g + i] = yi
    end

    return z
end


# ============================================================
# Conditional moments for grouped Gamma sufficient statistics
# ============================================================

function make_gamma_sufficient_statistics_adapters_grouped(
    raw_condMean,
    raw_condCov;
    min_mean::Real = 1e-10,
    min_precision::Real = 1e-10
)

    # ========================================================
    # Construct μ_i and κ_i directly from the state.
    #
    # This is important for regression because μ_i may vary
    # across observations within the same group.
    # ========================================================

    function gamma_parameters_from_original_model_grouped(
        param,
        state,
        t
    )

        # Linear predictor for mean
        ημ =
            param.Z[1][t] *
            state[param.Zidx[1]]

        # Linear predictor for precision
        ηκ =
            param.Z[2][t] *
            state[param.Zidx[2]]

        # If desired:
        #
        # ημ = clamp.(ημ, -20.0, 20.0)
        # ηκ = clamp.(ηκ, -20.0, 20.0)

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

        # Make sure they are vectors also in scalar cases
        μ = _gamma_vector(
            μ,
            "Gamma conditional mean"
        )

        κ = _gamma_vector(
            κ,
            "Gamma conditional precision"
        )

        g = length(μ)

        # ----------------------------------------------------
        # Allow scalar κ with regression only in the mean.
        #
        # If Zκ gives a single common precision parameter,
        # repeat it across all observations.
        # ----------------------------------------------------

        if length(κ) == 1 && g > 1
            κ = fill(
                κ[1],
                g
            )
        end

        length(κ) == g ||
            throw(DimensionMismatch(
                "Gamma mean has length $g but precision has " *
                "length $(length(κ))."
            ))

        # Numerical safeguards
        μ = max.(
            μ,
            min_mean
        )

        κ = max.(
            κ,
            min_precision
        )

        all(isfinite, μ) ||
            throw(DomainError(
                μ,
                "Non-finite Gamma mean."
            ))

        all(isfinite, κ) ||
            throw(DomainError(
                κ,
                "Non-finite Gamma precision."
            ))

        return μ, κ
    end


    # ========================================================
    # Moments of
    #
    # z =
    # [
    #   log(Y_1), ..., log(Y_g),
    #   Y_1,     ..., Y_g
    # ]
    #
    # ========================================================

    function condMoments_gamma_sufficient_grouped!(
        mean_z,
        R,
        param,
        state,
        t
    )

        μ, κ =
            gamma_parameters_from_original_model_grouped(
                param,
                state,
                t
            )

        g = length(μ)

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

        # Conditional independence across observations means
        # all cross-observation covariance entries are zero.
        fill!(
            R,
            zero(eltype(R))
        )

        @inbounds for i in 1:g

            j = g + i

            μi = μ[i]
            κi = κ[i]

            # =================================================
            # Conditional mean
            # =================================================

            # E[log(Y_i)]
            mean_z[i] =
                digamma(κi) +
                log(μi) -
                log(κi)

            # E[Y_i]
            mean_z[j] =
                μi


            # =================================================
            # Conditional covariance
            # =================================================

            # Var(log(Y_i))
            R[i, i] =
                trigamma(κi)

            # Var(Y_i)
            R[j, j] =
                μi^2 / κi

            # Cov(log(Y_i), Y_i)
            cov_log_y =
                μi / κi

            R[i, j] =
                cov_log_y

            R[j, i] =
                cov_log_y
        end

        # Usually not necessary analytically, but retain your
        # numerical safeguard for extreme sigma points.
        #R .= _make_spd(R;relative_floor = 1e-10)
        fill!(R, 0)
        @inbounds for i in 1:g

            j = g + i

            μi = μ[i]
            κi = κ[i]

            mean_z[i] =
                digamma(κi) +
                log(μi) -
                log(κi)

            mean_z[j] = μi

            Ri = [
                trigamma(κi)    μi / κi
                μi / κi         μi^2 / κi
            ]

            Ri = _make_spd(
                Ri;
                relative_floor = 1e-8
            )

            R[i, i] = Ri[1, 1]
            R[i, j] = Ri[1, 2]
            R[j, i] = Ri[2, 1]
            R[j, j] = Ri[2, 2]
        end

        return nothing
    end

    return condMoments_gamma_sufficient_grouped!
end


# ============================================================
# Observation transform constructor
# ============================================================

function GammaSuffStatsGrouped(;
    clip::Bool = false,
    boundary::Real = 1e-12,
    min_mean::Real = 1e-10,
    min_precision::Real = 1e-10
)

    return GammaSuffStats(

        # Transform raw observations
        y ->
            gamma_sufficient_observation_grouped(
                y;
                clip = clip,
                boundary = boundary
            ),

        # Construct transformed conditional moments
        (condMean, condCov) ->
            make_gamma_sufficient_statistics_adapters_grouped(
                condMean,
                condCov;
                min_mean = min_mean,
                min_precision = min_precision
            ),

        # Two statistics per raw observation
        nPerGroup -> 2 * nPerGroup
    )
end