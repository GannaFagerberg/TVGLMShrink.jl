
# ============================================================
# Negative Binomial:
# 3 factorial-statistic observation transform
#
# Parameterization:
#
#   E[Y]   = μ
#   Var[Y] = μ + μ^2/r
#
# Transform:
#
#   z1 = Y
#   z2 = Y(Y-1)
#   z3 = Y(Y-1)(Y-2)
#
# For grouped observations we use the average:
#
#   z̄ = (1/g) Σ z_i
#
# Hence
#
#   E[z̄]   = E[z]
#   Var[z̄] = Var[z] / g
# ============================================================


# ------------------------------------------------------------
# Extract scalar from scalar / length-one vector / 1×1 matrix
# ------------------------------------------------------------



# ============================================================
# NB factorial moments
#
# F_k = E[(Y)_k]
#
# where
#
#   (Y)_k = Y(Y-1)...(Y-k+1)
#
# For NB(μ,r):
#
#   F_k =
#       μ^k ∏_{j=0}^{k-1} (1 + j/r)
#
# We need moments through k=6 because z3² contains (Y)_6.
# ============================================================

function _nb_factorial_moments(
    μ::Real,
    r::Real,
    K::Int = 6
)

    μ > 0 ||
        throw(DomainError(μ, "NB mean μ must be positive."))

    r > 0 ||
        throw(DomainError(r, "NB size r must be positive."))


    # F[k+1] = F_k
    #
    # F_0 = 1

    F = Vector{Float64}(undef, K + 1)

    F[1] = 1.0

    for k in 1:K

        # recursion:
        #
        # F_k =
        # F_{k-1} * μ * (1 + (k-1)/r)

        F[k + 1] =
            F[k] *
            μ *
            (1.0 + (k - 1.0) / r)
    end

    return F
end


# ============================================================
# Product of two falling factorials
#
# Identity:
#
# (Y)_a (Y)_b
#
#   = Σ_{j=0}^{min(a,b)}
#
#       binomial(a,j)
#       binomial(b,j)
#       j!
#       (Y)_{a+b-j}
#
# Therefore its expectation follows directly from F_k.
# ============================================================

function _nb_factorial_product_moment(
    F,
    a::Int,
    b::Int
)

    out = 0.0

    for j in 0:min(a, b)

        coefficient =
            binomial(a, j) *
            binomial(b, j) *
            factorial(j)

        k =
            a + b - j

        # F[k+1] = F_k

        out +=
            coefficient *
            F[k + 1]
    end

    return out
end


# ============================================================
# Mean and covariance of
#
#   z =
#   [
#       (Y)_1
#       (Y)_2
#       (Y)_3
#   ]
#
# i.e.
#
#   [
#       Y
#       Y(Y-1)
#       Y(Y-1)(Y-2)
#   ]
# ============================================================

function _nb_factorial3_moments(
    μ::Real,
    r::Real
)

    # Need F_0,...,F_6

    F =
        _nb_factorial_moments(
            μ,
            r,
            6
        )


    # --------------------------------------------------------
    # Conditional mean
    # --------------------------------------------------------

    m =
        Vector{Float64}(undef, 3)

    m[1] = F[2]   # F1
    m[2] = F[3]   # F2
    m[3] = F[4]   # F3


    # --------------------------------------------------------
    # Conditional covariance
    # --------------------------------------------------------

    R =
        Matrix{Float64}(undef, 3, 3)


    for a in 1:3
        for b in a:3

            Eab =
                _nb_factorial_product_moment(
                    F,
                    a,
                    b
                )

            cab =
                Eab -
                m[a] * m[b]

            R[a, b] = cab
            R[b, a] = cab
        end
    end


    # Remove tiny negative diagonal values from floating
    # point arithmetic only.

    for j in 1:3
        R[j, j] =
            max(R[j, j], 0.0)
    end


    return m, R
end


# ============================================================
# Recover r from your EXISTING raw conditional moments
#
# Your ordinary NB functions give
#
#   E[Y]   = μ
#
#   Var[Y] = μ + μ²/r
#
# hence
#
#   r = μ² / (Var[Y] - μ)
#
# This means the transform does NOT require any changes to
# your existing condMean / condCov functions.
# ============================================================

function _nb_r_from_mean_variance(
    μ::Real,
    varY::Real
)

    excess =
        varY - μ


    # NB requires Var[Y] > μ.
    #
    # This tiny safeguard is only for roundoff close to
    # the Poisson limit r -> Inf.

    tol =
        100 *
        eps(Float64) *
        max(1.0, abs(μ), abs(varY))


    if excess <= tol

        # Numerically Poisson.
        #
        # A very large r reproduces the limiting moments
        # without introducing Inf into later calculations.

        return 1e12
    end


    return μ^2 / excess
end


# ============================================================
# Observation transform object
#
# This follows the SAME interface as your existing
# BetaSuffStatsAveraged / GammaSuffStatsAveraged setup.
# ============================================================

struct NBFactorialStats{F1,F2,F3} <: AbstractObsTransform
    transform_obs::F1
    make_cond_moments::F2
    obs_dim::F3
end


# ============================================================
# Constructor
#
# IMPORTANT:
#
# Keep the SAME name as in your current script:
#
#     obsTransform = NBFactorialStatsAveraged()
#
# so you do not need to change the rest of the script.
# ============================================================

function NBFactorialStatsAveraged()


    # ========================================================
    # Transform actual observations
    #
    # Handles both
    #
    #   scalar y
    #
    # and
    #
    #   vector y = observations in one group
    #
    # For a group we return the average sufficient-like
    # statistic.
    # ========================================================

    transform_obs =
        function(y)

            # ------------------------------------------------
            # Single observation
            # ------------------------------------------------

            if y isa Number

                yf =
                    Float64(y)

                z1 =
                    yf

                z2 =
                    yf *
                    (yf - 1.0)

                z3 =
                    yf *
                    (yf - 1.0) *
                    (yf - 2.0)

                return [
                    z1,
                    z2,
                    z3
                ]
            end


            # ------------------------------------------------
            # Group of observations
            # ------------------------------------------------

            g =
                length(y)

            g > 0 ||
                error("Empty observation group.")


            z1 = 0.0
            z2 = 0.0
            z3 = 0.0


            @inbounds for yi in y

                yf =
                    Float64(yi)

                z1 +=
                    yf

                z2 +=
                    yf *
                    (yf - 1.0)

                z3 +=
                    yf *
                    (yf - 1.0) *
                    (yf - 2.0)
            end


            invg =
                1.0 / g


            return [
                z1 * invg,
                z2 * invg,
                z3 * invg
            ]
        end


    # ========================================================
    # Construct transformed conditional moments from your
    # EXISTING raw condMean and condCov.
    #
    # prepare_observation_transform passes these in.
    # ========================================================
make_cond_moments =
    function(
        condMean,
        condCov,
        nPerGroup
    )

        function condMoments(
            mean_z,
            R,
            param,
            state,
            t
        )

            # =================================================
            # Raw NB conditional means for observations
            # inside group t
            #
            # Usually length = nPerGroup
            # =================================================

            μraw =
                condMean(
                    param,
                    state,
                    t
                )

            μvec =
                μraw isa Number ?
                [Float64(μraw)] :
                vec(Float64.(μraw))


            # =================================================
            # Raw NB conditional covariance
            #
            # We only need the marginal variances Var(Y_i).
            #
            # Handles:
            #   scalar
            #   vector of variances
            #   diagonal/full covariance matrix
            # =================================================

            Vraw =
                condCov(
                    param,
                    state,
                    t
                )


            varvec =
                if Vraw isa Number

                    fill(
                        Float64(Vraw),
                        length(μvec)
                    )

                elseif Vraw isa AbstractVector

                    vec(Float64.(Vraw))

                elseif Vraw isa AbstractMatrix

                    Float64.(diag(Vraw))

                else

                    error(
                        "Unsupported condCov output type: $(typeof(Vraw))"
                    )
                end


            length(varvec) == length(μvec) ||
                error(
                    "NB condMean/condCov dimension mismatch: " *
                    "length(μ)=$(length(μvec)), " *
                    "length(var)=$(length(varvec))"
                )


            # =================================================
            # Number of observations in this actual group
            # =================================================

            g =
                length(μvec)


            # =================================================
            # Accumulate moments for the group average
            # =================================================

            fill!(mean_z, 0.0)
            fill!(R, 0.0)


            @inbounds for i in 1:g

                μi =
                    μvec[i]

                varYi =
                    varvec[i]


                # ---------------------------------------------
                # NB:
                #
                # Var(Y) = μ + μ²/r
                #
                # therefore
                #
                # r = μ² / (Var(Y)-μ)
                # ---------------------------------------------

                ri =
                    _nb_r_from_mean_variance(
                        μi,
                        varYi
                    )


                # ---------------------------------------------
                # Exact moments of
                #
                # [
                #   Y
                #   Y(Y-1)
                #   Y(Y-1)(Y-2)
                # ]
                # ---------------------------------------------

                mi,
                Vi =
                    _nb_factorial3_moments(
                        μi,
                        ri
                    )


                mean_z .+= mi
                R      .+= Vi
            end


            # =================================================
            # Moments of the AVERAGE
            #
            # mean:
            #
            #   E[z̄] = Σ E[z_i] / g
            #
            # covariance:
            #
            #   Var(z̄) = Σ Var(z_i) / g²
            #
            # =================================================

            mean_z ./= g
            R      ./= g^2


            return nothing
        end


        return condMoments
    end

    return NBFactorialStats(
        transform_obs,
        make_cond_moments,
        nPerGroup -> 3
    )
end


# ============================================================
# Prepare observation transformation
#
# INTERFACE UNCHANGED.
# ============================================================

function prepare_observation_transform_ref(
    transform::NBFactorialStats,
    Y,
    condMean,
    condCov,
    nPerGroup
)

    # --------------------------------------------------------
    # Transform observations
    # --------------------------------------------------------

    Y_transformed = [
        transform.transform_obs(Y[t])
        for t in eachindex(Y)
    ]


    # --------------------------------------------------------
    # Conditional moments for transformed observation
    #
    # nPerGroup is required because for averaged statistics
    #
    #     Var(z̄) = Var(z) / nPerGroup
    # --------------------------------------------------------

    condMoments =
        transform.make_cond_moments(
            condMean,
            condCov,
            nPerGroup
        )


    # --------------------------------------------------------
    # Dimension of transformed observation
    #
    # z = [
    #       Y,
    #       Y(Y-1),
    #       Y(Y-1)(Y-2)
    #     ]
    #
    # therefore dimension = 3
    # --------------------------------------------------------

    nObs =
        transform.obs_dim


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







# ============================================================
# Negative Binomial factorial-statistic transform
# GROUPED / REGRESSION VERSION
#
# Y_i | μ_i, r_i ~ NB(μ_i, r_i)
#
# E[Y_i]   = μ_i
# Var[Y_i] = μ_i + μ_i^2 / r_i
#
# Transform:
#
# z =
# [
#   Y_1, ..., Y_g,
#   Y_1(Y_1-1), ..., Y_g(Y_g-1),
#   Y_1(Y_1-1)(Y_1-2), ...,
#   Y_g(Y_g-1)(Y_g-2)
# ]
#
# Dimension = 3g
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


# ============================================================
# Raw observation transformation
# ============================================================

function nb_factorial_observation_grouped(y)

    y_group =
        _nb_vector(
            y,
            "y"
        )

    g = length(y_group)

    z =
        Vector{Float64}(
            undef,
            3 * g
        )

    @inbounds for i in 1:g

        yi = y_group[i]

        isfinite(yi) ||
            throw(DomainError(
                yi,
                "Negative Binomial observations must be finite."
            ))

        yi >= 0 ||
            throw(DomainError(
                yi,
                "Negative Binomial observations must be non-negative."
            ))

        # (Y)_1
        z[i] =
            yi

        # (Y)_2
        z[g + i] =
            yi *
            (yi - 1.0)

        # (Y)_3
        z[2g + i] =
            yi *
            (yi - 1.0) *
            (yi - 2.0)
    end

    return z
end


# ============================================================
# Conditional moments
# ============================================================

function make_nb_factorial_statistics_adapters_grouped(
    raw_condMean,
    raw_condCov;
    min_mean::Real = 1e-10,
    min_size::Real = 1e-3,
    relative_floor::Real = 1e-10
)

    # ========================================================
    # Construct μ_i and r_i directly from the state
    # ========================================================

    function nb_parameters_from_original_model_grouped(
        param,
        state,
        t
    )

        # Mean linear predictor
        ημ =
            param.Z[1][t] *
            state[param.Zidx[1]]

        # Size / precision linear predictor
        ηr =
            param.Z[2][t] *
            state[param.Zidx[2]]

        # ----------------------------------------------------
        # Transform to parameter scale
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
                "Negative Binomial mean"
            )

        r =
            _nb_vector(
                r,
                "Negative Binomial size"
            )


        # ----------------------------------------------------
        # Handle regression in only one parameter
        #
        # e.g. μ_i varies but r is common within group
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
                min_size
            )


        all(isfinite, μ) ||
            throw(DomainError(
                μ,
                "Non-finite Negative Binomial mean."
            ))

        all(isfinite, r) ||
            throw(DomainError(
                r,
                "Non-finite Negative Binomial size."
            ))


        return μ, r
    end


    # ========================================================
    # Conditional moments of grouped transformed observation
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


        # ----------------------------------------------------
        # Dimension checks
        # ----------------------------------------------------

        length(mean_z) == 3g ||
            throw(DimensionMismatch(
                "mean_z has length $(length(mean_z)), " *
                "expected $(3g)."
            ))

        size(R) == (3g, 3g) ||
            throw(DimensionMismatch(
                "R has size $(size(R)), " *
                "expected ($(3g), $(3g))."
            ))


        # Observations are conditionally independent
        fill!(
            R,
            zero(eltype(R))
        )


        # ====================================================
        # One 3x3 factorial-statistic block per observation
        # ====================================================

        @inbounds for i in 1:g

            μi =
                μ[i]

            ri =
                r[i]


            # ------------------------------------------------
            # Exact NB moments of
            #
            # [
            #   (Y)_1
            #   (Y)_2
            #   (Y)_3
            # ]
            # ------------------------------------------------

            mi,
            Vi =
                _nb_factorial3_moments(
                    μi,
                    ri
                )


            # Optional numerical SPD regularization
            Vi =
                _make_spd(
                    Vi;
                    relative_floor =
                        relative_floor
                )


            # ------------------------------------------------
            # Locations in full transformed vector
            #
            # i       -> (Y_i)_1
            # g+i     -> (Y_i)_2
            # 2g+i    -> (Y_i)_3
            # ------------------------------------------------

            i1 =
                i

            i2 =
                g + i

            i3 =
                2g + i


            # ------------------------------------------------
            # Conditional mean
            # ------------------------------------------------

            mean_z[i1] =
                mi[1]

            mean_z[i2] =
                mi[2]

            mean_z[i3] =
                mi[3]


            # ------------------------------------------------
            # Conditional covariance block
            # ------------------------------------------------

            idx =
                (i1, i2, i3)

            for a in 1:3
                for b in 1:3

                    R[
                        idx[a],
                        idx[b]
                    ] =
                        Vi[a, b]
                end
            end
        end


        return nothing
    end


    return condMoments_nb_factorial_grouped!
end

function NBFactorialStatsGrouped(;
    min_mean::Real = 1e-10,
    min_size::Real = 1e-3,
    relative_floor::Real = 1e-10
)

    return NBFactorialStats(

        # ----------------------------------------------------
        # Observation transformation
        # ----------------------------------------------------

        y ->
            nb_factorial_observation_grouped(
                y
            ),


        # ----------------------------------------------------
        # Conditional moments
        # ----------------------------------------------------

        (condMean, condCov, nPerGroup) ->
            make_nb_factorial_statistics_adapters_grouped(
                condMean,
                condCov;
                min_mean =
                    min_mean,
                min_size =
                    min_size,
                relative_floor =
                    relative_floor
            ),


        # ----------------------------------------------------
        # Three transformed observations per original y_i
        # ----------------------------------------------------

        nPerGroup ->
            3 * nPerGroup
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
            condCov,
            nPerGroup
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