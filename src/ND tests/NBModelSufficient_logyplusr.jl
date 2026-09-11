


# ============================================================
# Negative-binomial shifted-log observation transformation
#
# Parameterisation:
#
#     Y | μ, r ~ NB(μ, r)
#
#     E[Y]   = μ
#     Var[Y] = μ + μ²/r
#
# Reference trajectory:
#
#     r0 > 0
#
# Observation transformation:
#
#     z(y) =
#     [
#         y
#         log(y + r0)
#     ]
#
# IMPORTANT:
#
# r0 defines the observation transformation and is fixed
# externally during the Gibbs run.
#
# The model dispersion r remains time-varying/state-dependent.
# ============================================================


# ============================================================
# Cache
# ============================================================

const NB_LOGR_CACHE =
    Dict{NTuple{3,Float64}, NTuple{3,Float64}}()


# ============================================================
# Conditional moments involving
#
#     log(Y + r0)
#
# for
#
#     Y ~ NB(μ,r).
#
# Calculate numerically
#
#     E[log(Y+r0)]
#     E[log(Y+r0)^2]
#     E[Y log(Y+r0)]
#
# Notice:
#
#     r  = current model dispersion
#     r0 = fixed transformation reference
# ============================================================

function _nb_vector(value, name::AbstractString)

    if value isa Real
        return [float(value)]

    elseif value isa AbstractVector
        return vec(float.(value))

    else
        throw(
            ArgumentError(
                "$name must be a scalar or vector; got $(typeof(value))."
            )
        )
    end
end


function _nb_check_r0(r0::Real)
    isfinite(r0) && r0 > 0 ||
        throw(DomainError(r0, "r0 must be finite and positive."))

    return Float64(r0)
end

function _nb_check_r0(r0::AbstractVector{<:Real})
    all(isfinite, r0) ||
        throw(DomainError(r0, "All elements of r0 must be finite."))

    all(r0 .> 0) ||
        throw(DomainError(r0, "All elements of r0 must be positive."))

    return Float64.(r0)
end

function _nb_logr_moments(
    μ::Real,
    r::Real,
    r0::Real;
    tol::Real = 1e-10,
    maxiter::Int = 100_000
)

    μ = max(float(μ), 1e-12)
    r = max(float(r), 1e-10)
    r0 = _nb_check_r0(r0)

    # NB parameterisation
    p = r / (r + μ)
    q = μ / (r + μ)

    # P(Y = 0)
    py = exp(r * log(p))
    cumulative_probability = py

    # At Y = 0:
    #
    #     log(Y+r0) = log(r0)
    #
    # so Y=0 contributes to E[log] and E[log²].
    log_y = log(r0)

    mean_log = py * log_y
    second_log = py * log_y^2
    mean_y_log = 0.0

    y = 0

    while (
        max(1.0 - cumulative_probability, 0.0) > tol
    ) && (
        y < maxiter
    )

        y += 1

        # Recursive NB pmf
        py *= ((y - 1 + r) / y) * q

        log_y = log(y + r0)

        mean_log += py * log_y
        second_log += py * log_y^2
        mean_y_log += py * y * log_y

        cumulative_probability += py
    end

    return (
        mean_log,
        second_log,
        mean_y_log
    )
end


# ============================================================
# Cached version
#
# Depends on μ, r and r0.
# ============================================================

function _nb_logr_moments_cached(
    μ::Real,
    r::Real,
    r0::Real
)

    μsafe = max(float(μ), 1e-12)
    rsafe = max(float(r), 1e-10)
    r0safe = _nb_check_r0(r0)

    key = (
        Float64(μsafe),
        Float64(rsafe),
        Float64(r0safe)
    )

    return get!(
        NB_LOGR_CACHE,
        key
    ) do

        _nb_logr_moments(
            μsafe,
            rsafe,
            r0safe;
            tol = 1e-10,
            maxiter = 100_000
        )
    end
end


# ============================================================
# GROUPED OBSERVATION TRANSFORMATION
#
# For group size g:
#
# z =
# [
#     Y_1
#     ...
#     Y_g
#     log(Y_1+r0_1)
#     ...
#     log(Y_g+r0_g)
# ]
#
# Dimension = 2g.
# ============================================================

function nb_factorial_observation_grouped(
    y,
    r0::AbstractVector{<:Real}
)

    length(y) == length(r0) ||
        throw(DimensionMismatch(
            "length(y)=$(length(y)), length(r0)=$(length(r0))."
        ))

    yv = Float64.(y)

    all(yv .>= 0) ||
        throw(DomainError(
            yv,
            "Negative-binomial observations must be non-negative."
        ))

    z1 = yv
    z2 = log.(yv .+ r0)

    return vcat(z1, z2)
end


# Scalar-reference version, useful if r0 is constant.
function nb_factorial_observation_grouped(
    y,
    r0::Real
)

    yv = Float64.(y)

    all(yv .>= 0) ||
        throw(DomainError(
            yv,
            "Negative-binomial observations must be non-negative."
        ))

    z1 = yv
    z2 = log.(yv .+ r0)

    return vcat(z1, z2)
end


# ============================================================
# GROUPED CONDITIONAL MOMENTS
# ============================================================

function make_nb_factorial_statistics_adapters_grouped(
    raw_condMean,
    raw_condCov;
    r0::Union{Real,AbstractVector{<:Real}},
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    relative_floor::Real = 1e-10
)

    r0 = _nb_check_r0(r0)

    function condMoments_nb_grouped!(
        mean_z,
        R,
        param,
        state,
        t
    )

        # ----------------------------------------------------
        # Current μ_i and r_i from current state.
        #
        # r0 only defines log(Y_i+r0_i).
        # It does NOT replace the current model r_i.
        # ----------------------------------------------------

        ημ = param.Z[1][t] * state[param.Zidx[1]]
        ηr = param.Z[2][t] * state[param.Zidx[2]]

        μ = linkinv.(Ref(param.link[1]), ημ)
        r = linkinv.(Ref(param.link[2]), ηr)

        μ = _nb_vector(μ, "NB mean")
        r = _nb_vector(r, "NB size")

        g = length(μ)

        # Size may be intercept-only.
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
        # Select r0 values corresponding to filtering group t.
        # ----------------------------------------------------

        if r0 isa Real

            r0_group = fill(r0, g)

        else

            first_idx = (t - 1) * g + 1
            last_idx = first_idx + g - 1

            last_idx <= length(r0) ||
                throw(DimensionMismatch(
                    "r0 has length $(length(r0)), but group $t " *
                    "requires indices $first_idx:$last_idx."
                ))

            r0_group = @view r0[first_idx:last_idx]
        end

        # ----------------------------------------------------
        # One [Y_i, log(Y_i+r0_i)] block per observation.
        # ----------------------------------------------------

        @inbounds for i in 1:g

            μi = μ[i]
            ri = r[i]
            r0i = r0_group[i]

            mean_log, second_log, mean_y_log =
                _nb_logr_moments_cached(
                    μi,
                    ri,
                    r0i
                )

            # Current NB variance
            variance_y = μi + μi^2 / ri

            # Moments of log(Y+r0)
            variance_log = second_log - mean_log^2
            covariance_y_log = mean_y_log - μi * mean_log

            variance_log = max(variance_log, eps(Float64))

            # Second statistic location
            j = g + i

            mean_z[i] = μi
            mean_z[j] = mean_log

            # ------------------------------------------------
            # 2 × 2 covariance block
            #
            # [ Var(Y_i)             Cov(Y_i,log_i) ]
            # [ Cov(Y_i,log_i)       Var(log_i)     ]
            # ------------------------------------------------

            v1 = variance_y
            v2 = variance_log
            c = covariance_y_log

            detR = v1 * v2 - c^2

            # Numerical SPD protection
            if !isfinite(detR) || detR <= 0.0

                δ = relative_floor * max(v1, v2, 1.0)

                v1 += δ
                v2 += δ

                cmax = sqrt(v1 * v2) * (1.0 - 1e-10)
                c = clamp(c, -cmax, cmax)
            end

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
# Observation-transform type
# ============================================================

struct NBSuffStats{F1,F2,F3} <: AbstractObsTransform
    transform_obs::F1
    make_cond_moments::F2
    obs_dim::F3
end


# ============================================================
# GROUPED CONSTRUCTOR
#
#     NBFactorialStatsGrouped(r0=rtime)
# ============================================================

function NBFactorialStatsGrouped(;
    r0::Union{Real,AbstractVector{<:Real}},
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    relative_floor::Real = 1e-10
)

    r0 = _nb_check_r0(r0)

    return NBSuffStats(

        (y, t, nPerGroup) -> begin

            if r0 isa Real
                r0_t = r0
            else
                i1 = (t - 1) * nPerGroup + 1
                i2 = i1 + length(y) - 1
                r0_t = @view r0[i1:i2]
            end

            nb_factorial_observation_grouped(y, r0_t)
        end,

        (condMean, condCov) ->
            make_nb_factorial_statistics_adapters_grouped(
                condMean,
                condCov;
                r0 = r0,
                min_mean = min_mean,
                min_dispersion = min_dispersion,
                relative_floor = relative_floor
            ),

        nPerGroup -> 2 * nPerGroup
    )
end


function prepare_observation_transform(
    transform::NBSuffStats,
    Y,
    condMean,
    condCov,
    nPerGroup
)

    Y_transformed = [
        transform.transform_obs(
            Y[t],
            t,
            nPerGroup
        )
        for t in eachindex(Y)
    ]

    condMoments = transform.make_cond_moments(
        condMean,
        condCov
    )

    nObs = transform.obs_dim(
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

    Irr =
        s -
        μ / (r * (μ + r))

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

