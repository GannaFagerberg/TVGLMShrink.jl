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
LogLinLink() = LogLinLink(8.0)

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