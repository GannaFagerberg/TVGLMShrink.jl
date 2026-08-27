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


# ============================================================
# Extract one state component for the no-regression case
# ============================================================

function _gamma_single_state(state, idx, name::AbstractString)

    if idx isa Integer
        return state[idx]
    end

    values = state[idx]

    length(values) == 1 ||
        throw(DimensionMismatch(
            "$name has $(length(values)) state components; " *
            "the no-regression Gamma helper expects exactly one."
        ))

    return only(values)
end


# ============================================================
# Gamma parameters from state
#
# No regression:
#
# η_μ = single mean state
# η_κ = single precision state
#
# μ = linkinv(mean link, η_μ)
# κ = linkinv(precision link, η_κ)
# ============================================================

function _gamma_parameters_noregression(
    param,
    state;
    min_mean::Real = 1e-10,
    min_precision::Real = 1e-10
)

    ημ = _gamma_single_state(
        state,
        param.Zidx[1],
        "Gamma mean predictor"
    )

    ηκ = _gamma_single_state(
        state,
        param.Zidx[2],
        "Gamma precision predictor"
    )

    μ = linkinv(
        param.link[1],
        ημ
    )

    κ = linkinv(
        param.link[2],
        ηκ
    )

    isfinite(μ) ||
        throw(DomainError(
            μ,
            "Non-finite Gamma mean."
        ))

    isfinite(κ) ||
        throw(DomainError(
            κ,
            "Non-finite Gamma precision."
        ))

    μ = max(
        μ,
        min_mean
    )

    κ = max(
        κ,
        min_precision
    )

    return μ, κ
end

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

    z = Vector{Float64}(
        undef,
        2 * g
    )

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

        z[i]     = log(yi)
        z[g + i] = yi
    end

    return z
end

function gamma_sufficient_observation_summed(
    y;
    clip::Bool = false,
    boundary::Real = 1e-12
)

    y_group = _gamma_vector(
        y,
        "y"
    )

    sum_log_y = 0.0
    sum_y     = 0.0

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

        sum_log_y += log(yi)
        sum_y     += yi
    end

    return [
        sum_log_y,
        sum_y
    ]
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

function make_gamma_sufficient_statistics_adapter_grouped_noregression(
    group_size::Integer;
    min_mean::Real = 1e-10,
    min_precision::Real = 1e-10
)

    g = Int(group_size)

    g > 0 ||
        throw(ArgumentError(
            "group_size must be positive."
        ))

    function condMoments_gamma_sufficient_grouped!(
        mean_z,
        R,
        param,
        state,
        t
    )

        μ, κ =
            _gamma_parameters_noregression(
                param,
                state;
                min_mean = min_mean,
                min_precision = min_precision
            )

        length(mean_z) == 2g ||
            throw(DimensionMismatch(
                "mean_z has length $(length(mean_z)), expected $(2g)."
            ))

        size(R) == (2g, 2g) ||
            throw(DimensionMismatch(
                "R has size $(size(R)), expected ($(2g), $(2g))."
            ))

        fill!(
            R,
            zero(eltype(R))
        )

        # ------------------------------------------------------
        # Moments for a single Gamma observation
        # ------------------------------------------------------

        mean_log_y =
            digamma(κ) +
            log(μ) -
            log(κ)

        variance_log_y =
            trigamma(κ)

        variance_y =
            μ^2 / κ

        covariance_logy_y =
            μ / κ

        # ------------------------------------------------------
        # Same μ and κ for all observations in the group
        #
        # z =
        # [
        #   log(y₁), ..., log(y_g),
        #   y₁,      ..., y_g
        # ]
        # ------------------------------------------------------

        @inbounds for i in 1:g

            j = g + i

            mean_z[i] =
                mean_log_y

            mean_z[j] =
                μ

            R[i, i] =
                variance_log_y

            R[j, j] =
                variance_y

            R[i, j] =
                covariance_logy_y

            R[j, i] =
                covariance_logy_y
        end

        return nothing
    end

    return condMoments_gamma_sufficient_grouped!
end

function make_gamma_sufficient_statistics_adapter_averaged_noregression(
    group_size::Integer;
    min_mean::Real = 1e-10,
    min_precision::Real = 1e-10
)

    g = Int(group_size)

    function condMoments_gamma_sufficient_averaged!(
        mean_z,
        R,
        param,
        state,
        t
    )

        μ, κ =
            _gamma_parameters_noregression(
                param,
                state;
                min_mean = min_mean,
                min_precision = min_precision
            )

        length(mean_z) == 2 ||
            throw(DimensionMismatch(
                "mean_z has length $(length(mean_z)), expected 2."
            ))

        size(R) == (2, 2) ||
            throw(DimensionMismatch(
                "R has size $(size(R)), expected (2, 2)."
            ))

        mean_z[1] =
            digamma(κ) +
            log(μ) -
            log(κ)

        mean_z[2] =
            μ

        R[1, 1] =
            trigamma(κ) / g

        R[2, 2] =
            (μ^2 / κ) / g

        R[1, 2] =
            (μ / κ) / g

        R[2, 1] =
            R[1, 2]

        return nothing
    end

    return condMoments_gamma_sufficient_averaged!
end

function make_gamma_sufficient_statistics_adapter_summed_noregression(
    group_size::Integer;
    min_mean::Real = 1e-10,
    min_precision::Real = 1e-10
)

    g = Int(group_size)

    function condMoments_gamma_sufficient_summed!(
        mean_z,
        R,
        param,
        state,
        t
    )

        μ, κ =
            _gamma_parameters_noregression(
                param,
                state;
                min_mean = min_mean,
                min_precision = min_precision
            )

        length(mean_z) == 2 ||
            throw(DimensionMismatch(
                "mean_z has length $(length(mean_z)), expected 2."
            ))

        size(R) == (2, 2) ||
            throw(DimensionMismatch(
                "R has size $(size(R)), expected (2, 2)."
            ))

        mean_log_y =
            digamma(κ) +
            log(μ) -
            log(κ)

        mean_z[1] =
            g * mean_log_y

        mean_z[2] =
            g * μ

        R[1, 1] =
            g * trigamma(κ)

        R[2, 2] =
            g * μ^2 / κ

        R[1, 2] =
            g * μ / κ

        R[2, 1] =
            R[1, 2]

        return nothing
    end

    return condMoments_gamma_sufficient_summed!
end