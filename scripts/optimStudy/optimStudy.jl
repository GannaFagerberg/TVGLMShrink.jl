using Optim, ForwardDiff, Plots, Random

# Simulate poisson regression data
β = [1,-0.3]
T = 500
X = [ones(T) randn(T)]
y = zeros(Int, T)
for t in 1:T
    λ = exp(X[t,:]'*β)
    y[t] = rand(Poisson(λ))
end


logposterior(β) = sum(logpdf.(Poisson.(exp.(X * β)), y))

θ₀ = randn(2)
f = θ -> -logposterior(θ)

# optimize generates an AD cache internally, but this can't be used across calls
@btime optimize($f, $θ₀, method = NewtonTrustRegion(); autodiff = :forward);

# Explicit Twice-differentiated AD cache, this can be reused across calls
td = Optim.TwiceDifferentiable(f, θ₀; autodiff = :forward)
@btime optimize($td, $θ₀, NewtonTrustRegion());

# This pattern can be used when optimizing for t = 1:T using the TwiceDifferentiable cache
# Prebuild TDs once. Each TD captures `state` (mutable) + its own block.
tds = [Optim.TwiceDifferentiable(θ -> f(θ, param, data[i]), θ₀;
        autodiff = :forward) for i in 1:1000]

# Safer BFGS

res = optimize(td, β0, BFGS())