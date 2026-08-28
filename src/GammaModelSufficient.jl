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

