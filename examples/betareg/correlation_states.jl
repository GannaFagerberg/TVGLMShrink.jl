using Random
using Distributions
using Statistics
using LinearAlgebra
using Plots

Random.seed!(123)

N = 100_000

logistic(x) = 1 / (1 + exp(-x))

# ============================================================
# CASE 1: fix mu, vary precision
# ============================================================

μ_fixed = 0.4

ηκ_1 = randn(N)
κ_1  = exp.(ηκ_1)

y_1 = [
    rand(Beta(μ_fixed * κ_1[i], (1 - μ_fixed) * κ_1[i]))
    for i in 1:N
]

println("Fixed μ:")
println("Cor(ηκ, Y) = ", cor(ηκ_1, y_1))


# ============================================================
# CASE 2a: independent mean and precision states
# ============================================================

ημ_2 = randn(N)
ηκ_2 = randn(N)

μ_2 = logistic.(ημ_2)
κ_2 = exp.(ηκ_2)

y_2 = [
    rand(Beta(μ_2[i] * κ_2[i], (1 - μ_2[i]) * κ_2[i]))
    for i in 1:N
]

println("\nIndependent states:")
println("Cor(ημ, ηκ) = ", cor(ημ_2, ηκ_2))
println("Cor(ηκ, Y)  = ", cor(ηκ_2, y_2))


# ============================================================
# CASE 2b: correlated mean and precision states
# ============================================================

ρ = 0.8

Σ = [
    1.0  ρ
    ρ    1.0
]

X = rand(MvNormal(zeros(2), Σ), N)

ημ_3 = X[1, :]
ηκ_3 = X[2, :]

μ_3 = logistic.(ημ_3)
κ_3 = exp.(ηκ_3)

y_3 = [
    rand(Beta(μ_3[i] * κ_3[i], (1 - μ_3[i]) * κ_3[i]))
    for i in 1:N
]

println("\nCorrelated states:")
println("Cor(ημ, ηκ) = ", cor(ημ_3, ηκ_3))
println("Cor(ηκ, Y)  = ", cor(ηκ_3, y_3))

scatter(ηκ_3, y_3)
scatter(ηκ_2, y_2)


# ============================================================
# Remove the part of ηκ induced by ημ
#
# For jointly Gaussian states:
#
# E[ηκ | ημ] = ρ ημ
#
# since both marginal variances equal 1.
# ============================================================

ηκ_res = ηκ_3 .- ρ .* ημ_3

println("Cor(residual ηκ, ημ) = ", cor(ηκ_res, ημ_3))
println("Cor(residual ηκ, Y)  = ", cor(ηκ_res, y_3))