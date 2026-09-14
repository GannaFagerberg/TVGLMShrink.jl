mutable struct TVGLMmodel{T,S<:AbstractMatrix{T},C<:Cholesky{T},L<:Tuple{Vararg{Link}}}
    Σᵥ::Vector{PDMat{T,S,C}}
    Z::Vector{Vector{Matrix{T}}}
    X::Vector{Matrix{T}}
    link::L
    Zidx::Vector{UnitRange{Int}}
end

# extending GLM link to have exp-lin inverse link function
struct LogLinLink{T<:Real} <: Link
    threshold::T
end

LogLinLink() = LogLinLink(5.0)

# linkfun: φ -> η 
function linkfun(link::LogLinLink, φ::Real)
    η₀ = link.threshold
    l₀ = exp(η₀)
    φ <= l₀ ? log(φ) : η₀ + φ / l₀ - 1
end

function linkinv(link::LogLinLink, η::Real)
    η₀ = link.threshold
    η <= η₀ ? exp(η) : exp(η₀) * (1 + (η - η₀))
end

# derivative of the inverse link function with respect to η
function mueta(link::LogLinLink, η::Real)
    η₀ = link.threshold
    η <= η₀ ? exp(η) : exp(η₀)
end


### Hardtanh
struct PositiveHardLink{T<:Real} <: Link
    floor::T
end

PositiveHardLink() = PositiveHardLink(1e-6)

# inverse link: η -> κ
function linkinv(link::PositiveHardLink, η::Real)
    return max(η, link.floor)
end

# derivative dκ/dη
function mueta(link::PositiveHardLink, η::Real)
    return η > link.floor ? one(η) : zero(η)
end

# link: κ -> η
function linkfun(link::PositiveHardLink, κ::Real)
    κ >= link.floor ||
        throw(DomainError(
            κ,
            "κ must be at least $(link.floor)."
        ))

    return κ
end

### SoftplusLink
struct ShiftedSoftplusLink{T<:Real} <: Link
    floor::T
end

ShiftedSoftplusLink() = ShiftedSoftplusLink(1e-6)


# η -> κ
function linkinv(
    link::ShiftedSoftplusLink,
    η::Real
)
    # numerically stable softplus
    softplus =
        log1p(exp(-abs(η))) +
        max(η, zero(η))

    return link.floor + softplus
end


# dκ/dη = logistic(η)
function mueta(
    link::ShiftedSoftplusLink,
    η::Real
)
    if η >= 0
        return inv(1 + exp(-η))
    else
        eη = exp(η)
        return eη / (1 + eη)
    end
end


# κ -> η
function linkfun(
    link::ShiftedSoftplusLink,
    κ::Real
)
    κ > link.floor ||
        throw(DomainError(
            κ,
            "κ must be greater than $(link.floor)."
        ))

    y = κ - link.floor

    # inverse softplus, numerically stable
    return y > 20 ?
        y :
        log(expm1(y))
end


### Woodart
# -------------------------------------------------------
# Woodard et al. link
#
# g(κ)     = log(exp(ακ) - 1) / α
# g⁻¹(η)   = log(1 + exp(αη)) / α
# -------------------------------------------------------

struct WoodardLink{T<:Real} <: Link
    α::T
end

WoodardLink() = WoodardLink(1.0)


# inverse link: η -> κ
function linkinv(
    link::WoodardLink,
    η::Real
)
    α = link.α
    z = α * η

    # numerically stable softplus
    if z > 0
        return (z + log1p(exp(-z))) / α
    else
        return log1p(exp(z)) / α
    end
end


# derivative dκ/dη
function mueta(
    link::WoodardLink,
    η::Real
)
    α = link.α
    z = α * η

    # numerically stable logistic(αη)
    if z >= 0
        return inv(1 + exp(-z))
    else
        ez = exp(z)
        return ez / (1 + ez)
    end
end


# link: κ -> η
function linkfun(
    link::WoodardLink,
    κ::Real
)
    κ > 0 ||
        throw(DomainError(
            κ,
            "κ must be positive."
        ))

    α = link.α
    z = α * κ

    # log(exp(z) - 1), evaluated stably
    if z > log(2)
        return (z + log1p(-exp(-z))) / α
    else
        return log(expm1(z)) / α
    end
end

struct UnitHardLink{T<:Real} <: Link
    floor::T
    ceiling::T
end

UnitHardLink() = UnitHardLink(1e-3, 1 - 1e-3)


# inverse link: η -> μ
function linkinv(
    link::UnitHardLink,
    η::Real
)
    return clamp(η, link.floor, link.ceiling)
end


# derivative dμ/dη
function mueta(
    link::UnitHardLink,
    η::Real
)
    return (link.floor < η < link.ceiling) ?
        one(η) :
        zero(η)
end


# link: μ -> η
function linkfun(
    link::UnitHardLink,
    μ::Real
)
    link.floor <= μ <= link.ceiling ||
        throw(DomainError(
            μ,
            "μ must be between $(link.floor) and $(link.ceiling)."
        ))

    return μ
end


function make_identity_cond_moments(
    raw_condMean,
    raw_condCov
)

    function condMoments_identity!(
        mean_z,
        R,
        param,
        state,
        t
    )

        μ = raw_condMean(param, state, t)
        V = raw_condCov(param, state, t)

        # Conditional mean
        if μ isa Real
            mean_z[1] = μ
        else
            copyto!(mean_z, vec(μ))
        end

        # Conditional covariance
        if V isa Real
            R[1, 1] = V
        else
            copyto!(R, V)
        end

        return nothing
    end

    return condMoments_identity!
end


function prepare_observation_transform(
    ::IdentityTransform,
    Y,
    condMean,
    condCov,
    nPerGroup
)

    # No transformation of observations
    Y_transformed = Y

    # Combine existing raw conditional mean/covariance
    # into the interface expected by IPLF
    condMoments =
        make_identity_cond_moments(
            condMean,
            condCov
        )

    # Dimension of one observation supplied to IPLF
    y₁ = first(Y)

    nObs =
        y₁ isa Real ?
        1 :
        length(vec(y₁))

    return (
        Y = Y_transformed,
        condMoments = condMoments,
        nObs = nObs
    )
end
