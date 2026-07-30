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
LogLinLink() = LogLinLink(7.0)

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


