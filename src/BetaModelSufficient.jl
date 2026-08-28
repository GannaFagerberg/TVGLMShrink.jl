
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


### For regression
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



function make_beta_sufficient_statistics_adapters_grouped(
    raw_condMean,
    raw_condCov;
    variance_denominator_offset::Real = 1.0,
    mean_boundary::Real = 1e-12,
    min_concentration::Real = 1e-10,
    shape_floor::Real = 1e-6
)

    function beta_shapes_from_original_model(
            param,
            state,
            t
        )
            
            ημ =param.Z[1][t] *state[param.Zidx[1]]
            ηκ =param.Z[2][t] *state[param.Zidx[2]]

            #ημ = clamp(ημ, -20.0, 20.0)
            #ηκ = clamp(ηκ, -20.0, 20.0)
          
            μ =linkinv.(Ref(param.link[1]),ημ)
            κ =linkinv.(Ref(param.link[2]),ηκ)

            # Compare against old construction only for extreme sigma points
            μ = clamp.(μ,mean_boundary,1.0 - mean_boundary)

            κ = max.(κ,min_concentration)

            all(isfinite, μ) ||
                throw(DomainError(
                    μ,
                    "Non-finite Beta mean."
                ))

            all(isfinite, κ) ||
                throw(DomainError(
                    κ,
                    "Non-finite Beta concentration."
                ))

            # Raw Beta shapes implied by the model
            alpha_raw = μ .* κ
            beta_raw  = (1.0 .- μ) .* κ

            # Diagnostic: check when SLR sigma points approach the boundary
            alpha_raw = μ .* κ
            beta_raw  = (1.0 .- μ) .* κ

            #if any(alpha_raw .< shape_floor) ||
            #any(beta_raw .< shape_floor)

              #  @show t

               # @show extrema(ημ)
               # @show extrema(μ)

               # @show extrema(ηκ)
                #@show extrema(κ)

               # @show minimum(alpha_raw)
               # @show minimum(beta_raw)
            #end
            
            # Numerical regularization for sufficient-statistic moments
            alpha_shape =
                max.(
                    alpha_raw,
                    shape_floor
                )

            beta_shape =
                max.(
                    beta_raw,
                    shape_floor
                )

            κ_effective =
                alpha_shape .+ beta_shape

            return κ_effective,
                alpha_shape,
                beta_shape
        end

    # ==========================================================
    # BOTH sufficient-statistic moments in one call
    # ==========================================================

    # ==========================================================
    # BOTH sufficient-statistic moments in one in-place call
    # ==========================================================

    function condMoments_beta_sufficient_grouped!(
        mean_z,
        R,
        param,
        state,
        t
    )

        # Beta shapes are still constructed once per sigma point
        κ, alpha_shape, beta_shape =
            beta_shapes_from_original_model(
                param,
                state,
                t
            )

        g = length(κ)

        length(mean_z) == 2g ||
            throw(DimensionMismatch(
                "mean_z has length $(length(mean_z)), expected $(2g)."
            ))

        size(R) == (2g, 2g) ||
            throw(DimensionMismatch(
                "R has size $(size(R)), expected ($(2g), $(2g))."
            ))

        # Important because only selected entries of R are written below
        fill!(R, zero(eltype(R)))

        @inbounds for i in 1:g

            j = g + i

            κi = κ[i]
            αi = alpha_shape[i]
            βi = beta_shape[i]

            # ------------------------------------------------------
            # Conditional mean
            # ------------------------------------------------------

            digamma_κ = digamma(κi)

            mean_z[i] =
                digamma(αi) - digamma_κ

            mean_z[j] =
                digamma(βi) - digamma_κ

            # ------------------------------------------------------
            # Conditional covariance
            # ------------------------------------------------------

            trigamma_κ = trigamma(κi)

            variance_log_y =
                trigamma(αi) - trigamma_κ

            variance_log_one_minus_y =
                trigamma(βi) - trigamma_κ

            covariance_logs =
                -trigamma_κ

            R[i, i] =
                variance_log_y

            R[j, j] =
                variance_log_one_minus_y

            R[i, j] =
                covariance_logs

            R[j, i] =
                covariance_logs
        end

        return nothing
    end

    return condMoments_beta_sufficient_grouped!
end


#########################
### For no regression 
#########################

function beta_sufficient_observation_summed(
    y;
    clip::Bool = false,
    boundary::Real = 1e-12
)

    y_group = _beta_vector(y, "y")

    sum_log_y = 0.0
    sum_log_one_minus_y = 0.0

    @inbounds for i in eachindex(y_group)

        yi = y_group[i]

        isfinite(yi) ||
            throw(DomainError(
                yi,
                "Beta observations must be finite."
            ))

        if !(0 < yi < 1)
            if clip
                yi = clamp(
                    yi,
                    boundary,
                    1.0 - boundary
                )
            else
                throw(DomainError(
                    yi,
                    "Beta observations must lie in (0,1)."
                ))
            end
        end

        sum_log_y += log(yi)
        sum_log_one_minus_y += log1p(-yi)
    end

    return [
        sum_log_y,
        sum_log_one_minus_y
    ]
end

## Takes existing Beta model functions for the raw observation
## Constructs new functions giving the conditional moments of the transformed sufficient-statistic observation
##

function beta_sufficient_observation_averaged(
    y;
    clip::Bool = false,
    boundary::Real = 1e-12
)

    y_group = _beta_vector(y, "y")
    g = length(y_group)

    mean_log_y = 0.0
    mean_log_one_minus_y = 0.0

    @inbounds for i in eachindex(y_group)

        yi = y_group[i]

        isfinite(yi) ||
            throw(DomainError(
                yi,
                "Beta observations must be finite."
            ))

        if !(0 < yi < 1)
            if clip
                yi = clamp(
                    yi,
                    boundary,
                    1.0 - boundary
                )
            else
                throw(DomainError(
                    yi,
                    "Beta observations must lie in (0,1)."
                ))
            end
        end

        mean_log_y += log(yi)
        mean_log_one_minus_y += log1p(-yi)
    end

    return [
        mean_log_y / g,
        mean_log_one_minus_y / g
    ]
end



### make_beta_sufficient_statistics_adapters_grouped
function make_beta_sufficient_statistics_adapters_averaged(
    raw_condMean,
    raw_condCov;
    variance_denominator_offset::Real = 1.0,
    mean_boundary::Real = 1e-12,
    min_concentration::Real = 1e-10
)

    # ==========================================================
    # Construct common Beta parameters for the whole group
    # ==========================================================

    function beta_shapes_from_original_model_averaged(
        param,
        state,
        t
    )

        # Conditional moments of original Beta observations
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

        μ = _beta_vector(
            μ_raw,
            "raw_condMean output"
        )

        # Number of original observations in the group
        g = length(μ)

        variance_y =
            _beta_variance_vector(
                covariance_raw,
                g
            )

        # ------------------------------------------------------
        # Common Beta parameters within the group
        # ------------------------------------------------------

        μi =
            clamp(
                μ[1],
                mean_boundary,
                1.0 - mean_boundary
            )

        variance_i =
            variance_y[1]

        isfinite(variance_i) &&
            variance_i > 0 ||
            throw(DomainError(
                variance_i,
                "Conditional Beta variance must be finite and positive."
            ))

        # Var(Y|x) = μ(1-μ)/(κ+1)
        κ =
            μi * (1.0 - μi) / variance_i -
            variance_denominator_offset

        isfinite(κ) &&
            κ > 0 ||
            throw(DomainError(
                κ,
                "The conditional moments imply non-positive concentration."
            ))

        κ =
            max(
                κ,
                min_concentration
            )

        alpha_shape =
            max(
                μi * κ,
                min_concentration
            )

        beta_shape =
            max(
                (1.0 - μi) * κ,
                min_concentration
            )

        # Account for numerical flooring
        κ_effective =
            alpha_shape + beta_shape

        return (
            κ_effective,
            alpha_shape,
            beta_shape,
            g
        )
    end


    # ==========================================================
    # Averaged sufficient-statistic moments
    #
    # z_bar =
    # [
    #   (1/g) sum_i log(y_i),
    #   (1/g) sum_i log(1-y_i)
    # ]
    #
    # Dimension is always 2.
    # ==========================================================

    function condMoments_beta_sufficient_averaged!(
        mean_z,
        R,
        param,
        state,
        t
    )

        κ, alpha_shape, beta_shape, g =
            beta_shapes_from_original_model_averaged(
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
        #
        # E[average] = E[single observation]
        # ------------------------------------------------------

        digamma_κ =
            digamma(κ)

        mean_z[1] =
            digamma(alpha_shape) -
            digamma_κ

        mean_z[2] =
            digamma(beta_shape) -
            digamma_κ

        # ------------------------------------------------------
        # Conditional covariance for one observation
        # ------------------------------------------------------

        trigamma_κ =
            trigamma(κ)

        variance_log_y =
            trigamma(alpha_shape) -
            trigamma_κ

        variance_log_one_minus_y =
            trigamma(beta_shape) -
            trigamma_κ

        covariance_logs =
            -trigamma_κ

        # ------------------------------------------------------
        # Covariance of the average:
        #
        # Var(z_bar | x) = Var(z_i | x) / g
        # ------------------------------------------------------

        R[1, 1] =
            variance_log_y / g

        R[2, 2] =
            variance_log_one_minus_y / g

        R[1, 2] =
            covariance_logs / g

        R[2, 1] =
            covariance_logs / g

        return nothing
    end

    return condMoments_beta_sufficient_averaged!
end


function make_beta_sufficient_statistics_adapters_summed(
    raw_condMean,
    raw_condCov;
    variance_denominator_offset::Real = 1.0,
    mean_boundary::Real = 1e-12,
    min_concentration::Real = 1e-10
)

    # ==========================================================
    # Construct common Beta parameters for the whole group
    # ==========================================================

    function beta_shapes_from_original_model_summed(
        param,
        state,
        t
    )

        # Conditional moments of original Beta observations
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

        μ = _beta_vector(
            μ_raw,
            "raw_condMean output"
        )

        # Actual number of raw observations in this group
        g = length(μ)

        variance_y = _beta_variance_vector(
            covariance_raw,
            g
        )

        # ------------------------------------------------------
        # Common Beta parameters within the group
        # ------------------------------------------------------

        μi = clamp(
            μ[1],
            mean_boundary,
            1.0 - mean_boundary
        )

        variance_i = variance_y[1]

        isfinite(variance_i) &&
            variance_i > 0 ||
            throw(DomainError(
                variance_i,
                "Conditional Beta variance must be finite and positive."
            ))

        # Var(Y|x) = μ(1-μ)/(κ+1)
        κ =
            μi * (1.0 - μi) / variance_i -
            variance_denominator_offset

        isfinite(κ) &&
            κ > 0 ||
            throw(DomainError(
                κ,
                "The conditional moments imply non-positive concentration."
            ))

        κ = max(
            κ,
            min_concentration
        )

        alpha_shape =
            max(
                μi * κ,
                min_concentration
            )

        beta_shape =
            max(
                (1.0 - μi) * κ,
                min_concentration
            )

        # Account for numerical flooring
        κ_effective =
            alpha_shape + beta_shape

        return (
            κ_effective,
            alpha_shape,
            beta_shape,
            g
        )
    end


    # ==========================================================
    # Summed sufficient-statistic moments
    #
    # z =
    # [
    #   sum_i log(y_i),
    #   sum_i log(1-y_i)
    # ]
    #
    # Dimension is ALWAYS 2.
    # ==========================================================

    function condMoments_beta_sufficient_summed!(
        mean_z,
        R,
        param,
        state,
        t
    )

        κ, alpha_shape, beta_shape, g =
            beta_shapes_from_original_model_summed(
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
        # Conditional mean for ONE observation
        # ------------------------------------------------------

        digamma_κ =
            digamma(κ)

        mean_log_y =
            digamma(alpha_shape) -
            digamma_κ

        mean_log_one_minus_y =
            digamma(beta_shape) -
            digamma_κ

        # For sum of g conditionally independent observations
        mean_z[1] =
            g * mean_log_y

        mean_z[2] =
            g * mean_log_one_minus_y

        # ------------------------------------------------------
        # Conditional covariance for ONE observation
        # ------------------------------------------------------

        trigamma_κ =
            trigamma(κ)

        variance_log_y =
            trigamma(alpha_shape) -
            trigamma_κ

        variance_log_one_minus_y =
            trigamma(beta_shape) -
            trigamma_κ

        covariance_logs =
            -trigamma_κ

        # ------------------------------------------------------
        # Covariance of the SUM:
        #
        # Var(sum z_i | x) = g Var(z_i | x)
        # ------------------------------------------------------

        R[1, 1] =
            g * variance_log_y

        R[2, 2] =
            g * variance_log_one_minus_y

        R[1, 2] =
            g * covariance_logs

        R[2, 1] =
            g * covariance_logs

        return nothing
    end

    return condMoments_beta_sufficient_summed!
end

struct BetaSuffStats{F1,F2,F3} <: AbstractObsTransform
    transform_obs::F1
    make_cond_moments::F2
    obs_dim::F3
end

function BetaSuffStatsGrouped(;
    variance_denominator_offset::Real = 1.0,
    mean_boundary::Real = 1e-12,
    min_concentration::Real = 1e-10,
    shape_floor::Real = 1e-6
)

    return BetaSuffStats(

        # Transform raw observations
        y -> beta_sufficient_observation_grouped(y),

        # Construct conditional-moment function
        (condMean, condCov) ->
            make_beta_sufficient_statistics_adapters_grouped(
                condMean,
                condCov;
                variance_denominator_offset =
                    variance_denominator_offset,
                mean_boundary =
                    mean_boundary,
                min_concentration =
                    min_concentration,
                shape_floor =
                    shape_floor
            ),

        # Dimension of transformed observation
        nPerGroup -> 2 * nPerGroup
    )
end

function BetaSuffStatsAveraged(;
    variance_denominator_offset::Real = 1.0,
    mean_boundary::Real = 1e-12,
    min_concentration::Real = 1e-10
)

    return BetaSuffStats(

        # Transform raw observations
        y -> beta_sufficient_observation_averaged(y),

        # Construct conditional-moment function
        (condMean, condCov) ->
            make_beta_sufficient_statistics_adapters_averaged(
                condMean,
                condCov;
                variance_denominator_offset =
                    variance_denominator_offset,
                mean_boundary =
                    mean_boundary,
                min_concentration =
                    min_concentration
            ),

        # Averaged sufficient statistic always has dimension 2
        nPerGroup -> 2
    )
end


function prepare_observation_transform(
    transform::BetaSuffStats,
    Y,
    condMean,
    condCov,
    nPerGroup
)

    Y_transformed = [
        transform.transform_obs(Y[t])
        for t in eachindex(Y)
    ]

    condMoments = transform.make_cond_moments(
        condMean,
        condCov
    )

    nObs = transform.obs_dim(nPerGroup)

    return (
        Y = Y_transformed,
        condMoments = condMoments,
        nObs = nObs
    )
end