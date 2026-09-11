# ============================================================
# Negative-binomial reference-score observation transformation
#
# Parameterisation:
#
#     Y | μ, r ~ NB(μ, r)
#
#     E[Y]   = μ
#     Var[Y] = μ + μ²/r
#
# Fixed reference value:
#
#     r0 > 0
#
# Observation transformation:
#
#     z(y) =
#     [
#         y
#         digamma(y + r0)
#     ]
#
# IMPORTANT:
#
# r0 defines the observation transformation and is held fixed
# throughout the Gibbs sampler.
#
# The model dispersion r remains time-varying / state-dependent.
#
# At r = r0, the exact NB score in (μ,r) is contained in the
# linear span of the centered transformed observation.
#
# PUBLIC NAMES ARE KEPT COMPATIBLE WITH THE EXISTING
# IPLF INFRASTRUCTURE.
# ============================================================

using SpecialFunctions: digamma


# ============================================================
# Cache
#
# Do NOT round μ and r aggressively here.
#
# Coarse rounding can make the conditional-moment map
# artificially piecewise constant across IPLF sigma points.
# Exact floating-point values are therefore used as keys.
# ============================================================

const NB_PSI_CACHE =
    Dict{NTuple{3,Float64}, NTuple{3,Float64}}()


# ============================================================
# Observation-transform type
# ============================================================

struct NBSuffStats{F1,F2,F3} <: AbstractObsTransform
    transform_obs::F1
    make_cond_moments::F2
    obs_dim::F3
end


# ============================================================
# Helpers
# ============================================================

function _nb_vector(
    value,
    name::AbstractString
)

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


function _nb_variance_vector(
    value,
    group_size::Int
)

    if value isa Real

        group_size == 1 ||
            throw(
                DimensionMismatch(
                    "A scalar variance was returned for group size $group_size."
                )
            )

        return [float(value)]

    elseif value isa AbstractVector

        length(value) == group_size ||
            throw(
                DimensionMismatch(
                    "Variance vector has length $(length(value)); " *
                    "expected $group_size."
                )
            )

        return vec(float.(value))

    elseif value isa AbstractMatrix

        size(value) == (group_size, group_size) ||
            throw(
                DimensionMismatch(
                    "Conditional covariance has size $(size(value)); " *
                    "expected ($group_size, $group_size)."
                )
            )

        return float.(diag(value))

    else

        throw(
            ArgumentError(
                "Unsupported conditional covariance type $(typeof(value))."
            )
        )
    end
end

function _nb_check_r0(r0::Real)
    r0 > 0 || error("r0 must be positive")
    return Float64(r0)
end

function _nb_check_r0(r0::AbstractVector{<:Real})
    all(r0 .> 0) ||
        error("All elements of r0 must be positive")

    return Float64.(r0)
end


# ============================================================
# Conditional moments involving
#
#     ψ(Y + r0)
#
# for
#
#     Y ~ NB(μ,r).
#
# We calculate numerically
#
#     E[ψ(Y+r0)]
#
#     E[ψ(Y+r0)^2]
#
#     E[Y ψ(Y+r0)]
#
# using the complete NB pmf.
#
# Notice carefully:
#
#     r  = CURRENT model dispersion
#     r0 = FIXED transformation reference
#
# They are not the same quantity.
# ============================================================

function _nb_psi_moments(
    μ::Real,
    r::Real,
    r0::Real;
    tol::Real = 1e-10,
    maxiter::Int = 100_000
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

    r0 =
        _nb_check_r0(r0)


    # --------------------------------------------------------
    # NB parameterisation
    #
    # p = r/(r+μ)
    # q = μ/(r+μ)
    # --------------------------------------------------------

    p =
        r / (r + μ)

    q =
        μ / (r + μ)


    # --------------------------------------------------------
    # P(Y=0)
    # --------------------------------------------------------

    py =
        exp(
            r * log(p)
        )

    cumulative_probability =
        py


    # --------------------------------------------------------
    # Unlike log(1+Y), ψ(Y+r0) is generally nonzero at Y=0.
    #
    # Therefore Y=0 MUST contribute to E[ψ] and E[ψ²].
    # --------------------------------------------------------

    ψy =
        digamma(r0)

    mean_psi =
        py * ψy

    second_psi =
        py * ψy^2

    mean_y_psi =
        0.0


    y =
        0


    # --------------------------------------------------------
    # Recursive NB pmf
    #
    # P(Y=y) =
    # P(Y=y-1) *
    # ((y-1+r)/y) *
    # μ/(μ+r)
    # --------------------------------------------------------

    while (
        max(
            1.0 - cumulative_probability,
            0.0
        ) > tol
    ) && (
        y < maxiter
    )

        y += 1


        py *=
            ((y - 1 + r) / y) *
            q


        ψy =
            digamma(
                y + r0
            )


        mean_psi +=
            py *
            ψy


        second_psi +=
            py *
            ψy^2


        mean_y_psi +=
            py *
            y *
            ψy


        cumulative_probability +=
            py
    end


    return (
        mean_psi,
        second_psi,
        mean_y_psi
    )
end


# ============================================================
# Cached version
#
# Cache depends on THREE quantities:
#
#     μ, r, r0
#
# because the distribution uses (μ,r), while the statistic
# itself depends on r0.
# ============================================================

function _nb_psi_moments_cached(
    μ::Real,
    r::Real,
    r0::Real
)

    μsafe =
        max(
            float(μ),
            1e-12
        )

    rsafe =
        max(
            float(r),
            1e-10
        )

    r0safe =
        _nb_check_r0(r0)


    key = (
        Float64(μsafe),
        Float64(rsafe),
        Float64(r0safe)
    )


    return get!(
        NB_PSI_CACHE,
        key
    ) do

        _nb_psi_moments(
            μsafe,
            rsafe,
            r0safe;
            tol = 1e-10,
            maxiter = 100_000
        )
    end
end


# ============================================================
# AVERAGED TRANSFORMATION
#
# z̄ =
#
# [
#     mean(Y_i)
#     mean(ψ(Y_i+r0))
# ]
#
# This is included so that the old averaged interface also
# remains available.
# ============================================================

function nb_factorial_observation_averaged(
    y,
    r0::Real
)

    r0 =
        _nb_check_r0(r0)


    y_group =
        _nb_vector(
            y,
            "y"
        )


    g =
        length(y_group)


    mean_y =
        0.0

    mean_psi =
        0.0


    @inbounds for i in eachindex(y_group)

        yi =
            y_group[i]


        isfinite(yi) ||
            throw(
                DomainError(
                    yi,
                    "Negative-binomial observations must be finite."
                )
            )


        yi >= 0 ||
            throw(
                DomainError(
                    yi,
                    "Negative-binomial observations must be non-negative."
                )
            )


        isinteger(yi) ||
            throw(
                DomainError(
                    yi,
                    "Negative-binomial observations must be integer-valued."
                )
            )


        mean_y +=
            yi


        mean_psi +=
            digamma(
                yi + r0
            )
    end


    return [
        mean_y / g,
        mean_psi / g
    ]
end


# ============================================================
# AVERAGED CONDITIONAL MOMENTS
#
# This keeps the same logic as your old averaged implementation:
#
#     E[Y]   = μ
#     Var[Y] = μ + μ²/r
#
# so
#
#     r = μ²/(Var[Y]-μ).
#
# Current implementation assumes common μ and r within the
# group.
# ============================================================

function make_nb_factorial_statistics_adapters_averaged(
    raw_condMean,
    raw_condCov;
    r0::Real,
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    min_overdispersion::Real = 1e-10,
    relative_floor::Real = 1e-10
)

    r0 =
        _nb_check_r0(r0)


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


        μi =
            max(
                μ[1],
                min_mean
            )


        variance_i =
            variance_y[1]


        isfinite(μi) && μi > 0 ||
            throw(
                DomainError(
                    μi,
                    "Conditional negative-binomial mean must be finite and positive."
                )
            )


        isfinite(variance_i) && variance_i > 0 ||
            throw(
                DomainError(
                    variance_i,
                    "Conditional negative-binomial variance must be finite and positive."
                )
            )


        overdispersion =
            max(
                variance_i - μi,
                min_overdispersion
            )


        r =
            μi^2 /
            overdispersion


        isfinite(r) && r > 0 ||
            throw(
                DomainError(
                    r,
                    "The conditional moments imply invalid NB dispersion."
                )
            )


        r =
            max(
                r,
                min_dispersion
            )


        return μi, r, g
    end


    # --------------------------------------------------------
    # Moments of
    #
    # z =
    #
    # [
    #     Y
    #     ψ(Y+r0)
    # ]
    # --------------------------------------------------------

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
            throw(
                DimensionMismatch(
                    "mean_z has length $(length(mean_z)), expected 2."
                )
            )


        size(R) == (2, 2) ||
            throw(
                DimensionMismatch(
                    "R has size $(size(R)), expected (2,2)."
                )
            )


        mean_psi,
        second_psi,
        mean_y_psi =
            _nb_psi_moments_cached(
                μ,
                r,
                r0
            )


        # ----------------------------------------------------
        # Conditional mean
        # ----------------------------------------------------

        mean_z[1] =
            μ

        mean_z[2] =
            mean_psi


        # ----------------------------------------------------
        # Covariance for ONE observation
        # ----------------------------------------------------

        variance_y =
            μ +
            μ^2 / r


        variance_psi =
            second_psi -
            mean_psi^2


        covariance_y_psi =
            mean_y_psi -
            μ * mean_psi


        variance_psi =
            max(
                variance_psi,
                eps(Float64)
            )


        # ----------------------------------------------------
        # Covariance of group AVERAGE
        #
        # Var(z̄) = Var(z)/g
        # ----------------------------------------------------

        R[1, 1] =
            variance_y / g

        R[1, 2] =
            covariance_y_psi / g

        R[2, 1] =
            R[1, 2]

        R[2, 2] =
            variance_psi / g


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
# AVERAGED CONSTRUCTOR
# ============================================================

function NBFactorialStatsAveraged(;
    r0::Real,
    min_mean::Real = 1e-10,
    min_dispersion::Real = 1e-8,
    min_overdispersion::Real = 1e-10,
    relative_floor::Real = 1e-10
)

    r0 =
        _nb_check_r0(r0)


    return NBSuffStats(

        y ->
            nb_factorial_observation_averaged(
                y,
                r0
            ),

        (condMean, condCov) ->
            make_nb_factorial_statistics_adapters_averaged(
                condMean,
                condCov;
                r0 = r0,
                min_mean = min_mean,
                min_dispersion = min_dispersion,
                min_overdispersion = min_overdispersion,
                relative_floor = relative_floor
            ),

        nPerGroup ->
            2
    )
end


# ============================================================
# GROUPED / REGRESSION TRANSFORMATION
#
# For a group of size g:
#
# z =
#
# [
#     Y_1
#     ...
#     Y_g
#     ψ(Y_1+r0)
#     ...
#     ψ(Y_g+r0)
# ]
#
# Dimension = 2g.
# ============================================================
function nb_factorial_observation_grouped(
    y,
    r0::AbstractVector{<:Real}
)

    length(y) == length(r0) ||
        error(
            "Length mismatch: length(y)=$(length(y)), " *
            "length(r0)=$(length(r0))"
        )

    yv = Float64.(y)

    z1 = yv
    z2 = digamma.(yv .+ r0)

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
        # Current μ_i and r_i from the current state.
        #
        # r0 is only the reference appearing in
        #
        #     ψ(Y_i + r0_i)
        #
        # and does NOT replace the current model size r_i.
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
        # Select the reference r0 values belonging to group t.
        #
        # If r0 is scalar: same reference everywhere.
        #
        # If r0 is a vector:
        #
        # group 1 -> r0[1:g]
        # group 2 -> r0[g+1:2g]
        # etc.
        # ----------------------------------------------------

        if r0 isa Real
            r0_group = fill(r0, g)
        else
            first_idx = (t - 1) * g + 1
            last_idx = first_idx + g - 1

            last_idx <= length(r0) ||
                throw(DimensionMismatch(
                    "r0 has length $(length(r0)), but group $t requires indices $first_idx:$last_idx."
                ))

            r0_group = @view r0[first_idx:last_idx]
        end

        # ----------------------------------------------------
        # One
        #
        #     [Y_i, ψ(Y_i + r0_i)]
        #
        # block for each observation.
        # ----------------------------------------------------

        @inbounds for i in 1:g

            μi = μ[i]
            ri = r[i]
            r0i = r0_group[i]

            # ------------------------------------------------
            # Moments under
            #
            #     Y_i ~ NB(μ_i, r_i)
            #
            # of the transformed statistic ψ(Y_i + r0_i).
            # ------------------------------------------------

            mean_psi, second_psi, mean_y_psi =
                _nb_psi_moments_cached(μi, ri, r0i)

            # ------------------------------------------------
            # Conditional moments
            # ------------------------------------------------

            variance_y = μi + μi^2 / ri
            variance_psi = second_psi - mean_psi^2
            covariance_y_psi = mean_y_psi - μi * mean_psi

            variance_psi = max(variance_psi, eps(Float64))

            # Second statistic for observation i is at g+i.
            j = g + i

            mean_z[i] = μi
            mean_z[j] = mean_psi

            # ------------------------------------------------
            # 2 × 2 covariance block
            #
            # [ Var(Y_i)          Cov(Y_i, ψ_i) ]
            # [ Cov(Y_i, ψ_i)     Var(ψ_i)      ]
            # ------------------------------------------------

            v1 = variance_y
            v2 = variance_psi
            c = covariance_y_psi

            detR = v1 * v2 - c^2

            # ------------------------------------------------
            # Numerical SPD protection
            # ------------------------------------------------

            if !isfinite(detR) || detR <= 0.0

                δ = relative_floor * max(v1, v2, 1.0)

                v1 += δ
                v2 += δ

                cmax = sqrt(v1 * v2) * (1.0 - 1e-10)
                c = clamp(c, -cmax, cmax)
            end

            # ------------------------------------------------
            # Insert covariance block
            # ------------------------------------------------

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
# GROUPED CONSTRUCTOR
#
# This keeps the same external name as your previous code.
#
# The only new required argument is r0.
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

            nb_factorial_observation_grouped(
                y,
                r0_t
            )
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

        nPerGroup ->
            2 * nPerGroup
    )
end

# ============================================================
# Prepare observation transformation
#
# UNCHANGED interface.
# ============================================================

function _nb_r0_group(r0, t, nPerGroup)

    if r0 isa Real
        return r0
    end

    firstidx = (t - 1) * nPerGroup + 1
    lastidx  = firstidx + nPerGroup - 1

    return @view r0[firstidx:lastidx]
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

