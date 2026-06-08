using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
cd(joinpath(@__DIR__, "../.."))
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using PDMats, LogExpFunctions, ForwardDiff, Measures
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim
using Utils: mvcolors as colors

dataFolder = joinpath(@__DIR__, "figs/")
figFolder = joinpath(@__DIR__, "figs/")

Plots.reset_defaults()
gr(legend=nothing, grid=false, color=colors[1], lw=4,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=14, yguidefontsize=14,
    legendfontsize=14, titlefontsize=14)

# Linear approximation of a function for EKF - small variance
μ = 0.0
σ = 0.2
origDens = Normal(μ, σ)
origQuantiles = quantile(origDens, [0.025, 0.975])
xGrid = -(μ + 4 * σ):0.01:(μ+4*σ)
x₀ = μ # expand around the mode
p1 = plot(xGrid, pdf(origDens, xGrid), label="normal pdf", xlabel=L"z", ylabel=L"p(x)",
    title="Original density " * L"N(0,%$σ^2)", legend=false)
plot!([x₀, x₀], [0, pdf(origDens, x₀)], color=colors[2], linestyle=:dash, lw=2,
    label="expansion point")

func(x) = exp(x)
invfunc(x) = log(x)
linapprox(x, x₀) = exp(x₀) + exp(x₀) * (x - x₀)
p2 = vspan(origQuantiles, linecolor=:lightgray, fillcolor=:lightgray, opacity=0.3, lw=0, linestyle=:dash,
    label="95% interval", xlabel=L"z", ylabel=L"f(x)", title="Transformation",
    legend=:topleft)
p2 = plot!(xGrid, func.(xGrid), label=L"\exp(x)")
plot!(xGrid, linapprox.(xGrid, x₀), color=colors[2],
    label="linear approx", linestyle=:solid)
vline!([x₀], color=colors[2], linestyle=:dash, lw=2, label="expansion point")

transDensPDF(x) = pdf(origDens, invfunc(x)) * (1 / x)
xTransGrid = func.(xGrid)
p3 = plot(xTransGrid, transDensPDF.(xTransGrid), label="true", xlabel=L"z",
    ylabel=L"p(x)", title="Transformed density", legend=:topright)

μtransApprox = linapprox(μ, x₀)
σtransApprox = exp(x₀) * σ
plot!(xTransGrid, pdf(Normal(μtransApprox, σtransApprox), xTransGrid),
    label="approx", color=colors[2], linestyle=:solid)

plot(p1, p2, p3, layout=(1, 3), size=(1200, 400), bottommargin=5mm)
savefig(figFolder * "EKFapprox_small.png")

# Linear approximation of a function for EKF - moderate variance
μ = 0.0
σ = 0.5
origDens = Normal(μ, σ)
origQuantiles = quantile(origDens, [0.025, 0.975])
xGrid = -(μ + 4 * σ):0.01:(μ+4*σ)
x₀ = μ # expand around the mode
p1 = plot(xGrid, pdf(origDens, xGrid), label="normal pdf", xlabel=L"z", ylabel=L"p(x)",
    title="Original density " * L"N(0,%$σ^2)", legend=false)
plot!([x₀, x₀], [0, pdf(origDens, x₀)], color=colors[2], linestyle=:dash, lw=2,
    label="expansion point")

func(x) = exp(x)
invfunc(x) = log(x)
linapprox(x, x₀) = exp(x₀) + exp(x₀) * (x - x₀)
p2 = vspan(origQuantiles, linecolor=:lightgray, fillcolor=:lightgray, opacity=0.3, lw=0, linestyle=:dash,
    label="95% interval", xlabel=L"z", ylabel=L"f(x)", title="Transformation",
    legend=:topleft)
p2 = plot!(xGrid, func.(xGrid), label=L"\exp(x)")
plot!(xGrid, linapprox.(xGrid, x₀), color=colors[2],
    label="linear approx", linestyle=:solid)
vline!([x₀], color=colors[2], linestyle=:dash, lw=2, label="expansion point")

transDensPDF(x) = pdf(origDens, invfunc(x)) * (1 / x)
xTransGrid = func.(xGrid)
p3 = plot(xTransGrid, transDensPDF.(xTransGrid), label="true", xlabel=L"z",
    ylabel=L"p(x)", title="Transformed density", legend=:topright)

μtransApprox = linapprox(μ, x₀)
σtransApprox = exp(x₀) * σ
plot!(xTransGrid, pdf(Normal(μtransApprox, σtransApprox), xTransGrid),
    label="approx", color=colors[2], linestyle=:solid)

plot(p1, p2, p3, layout=(1, 3), size=(1200, 400), bottommargin=5mm)
savefig(figFolder * "EKFapprox_moderate.png")

# Linear approximation of a function for EKF - large variance
μ = 0.0
σ = 1
origDens = Normal(μ, σ)
origQuantiles = quantile(origDens, [0.025, 0.975])
xGrid = -(μ + 4 * σ):0.01:(μ+4*σ)
x₀ = μ # expand around the mode
p1 = plot(xGrid, pdf(origDens, xGrid), label="normal pdf", xlabel=L"z", ylabel=L"p(x)",
    title="Original density " * L"N(0,%$σ^2)", legend=false)
plot!([x₀, x₀], [0, pdf(origDens, x₀)], color=colors[2], linestyle=:dash, lw=2,
    label="expansion point")

func(x) = exp(x)
invfunc(x) = log(x)
linapprox(x, x₀) = exp(x₀) + exp(x₀) * (x - x₀)
p2 = vspan(origQuantiles, linecolor=:lightgray, fillcolor=:lightgray, opacity=0.3, lw=0, linestyle=:dash,
    label="95% interval", xlabel=L"z", ylabel=L"f(x)", title="Transformation",
    legend=:topleft)
p2 = plot!(xGrid, func.(xGrid), label=L"\exp(x)")
plot!(xGrid, linapprox.(xGrid, x₀), color=colors[2],
    label="linear approx", linestyle=:solid)
vline!([x₀], color=colors[2], linestyle=:dash, lw=2, label="expansion point")

transDensPDF(x) = pdf(origDens, invfunc(x)) * (1 / x)
xTransGrid = func.(xGrid)
p3 = plot(xTransGrid, transDensPDF.(xTransGrid), label="true", xlabel=L"z",
    ylabel=L"p(x)", title="Transformed density", legend=:topright)

μtransApprox = linapprox(μ, x₀)
σtransApprox = exp(x₀) * σ
plot!(xTransGrid, pdf(Normal(μtransApprox, σtransApprox), xTransGrid),
    label="approx", color=colors[2], linestyle=:solid)

plot(p1, p2, p3, layout=(1, 3), size=(1200, 400), bottommargin=5mm)
savefig(figFolder * "EKFapprox_large.png")


# Linear approximation of a function for EKF - moderate variance and wrong expansion point
μ = 0.0
σ = 0.2
origDens = Normal(μ, σ)
origQuantiles = quantile(origDens, [0.025, 0.975])
xGrid = -(μ + 4 * σ):0.01:(μ+4*σ)
x₀ = μ - 1.96 * σ # expand around the mode
p1 = plot(xGrid, pdf(origDens, xGrid), label="normal pdf", xlabel=L"z", ylabel=L"p(x)",
    title="Original density " * L"N(0,%$σ^2)", legend=false)
plot!([x₀, x₀], [0, pdf(origDens, x₀)], color=colors[2], linestyle=:dash, lw=2,
    label="expansion point")

func(x) = exp(x)
invfunc(x) = log(x)
linapprox(x, x₀) = exp(x₀) + exp(x₀) * (x - x₀)
p2 = vspan(origQuantiles, linecolor=:lightgray, fillcolor=:lightgray, opacity=0.3, lw=0, linestyle=:dash,
    label="95% interval", xlabel=L"z", ylabel=L"f(x)", title="Transformation",
    legend=:topleft)
p2 = plot!(xGrid, func.(xGrid), label=L"\exp(x)")
plot!(xGrid, linapprox.(xGrid, x₀), color=colors[2],
    label="linear approx", linestyle=:solid)
vline!([x₀], color=colors[2], linestyle=:dash, lw=2, label="expansion point")

transDensPDF(x) = pdf(origDens, invfunc(x)) * (1 / x)
xTransGrid = func.(xGrid)
p3 = plot(xTransGrid, transDensPDF.(xTransGrid), label="true", xlabel=L"z",
    ylabel=L"p(x)", title="Transformed density", legend=:topright)

μtransApprox = linapprox(μ, x₀)
σtransApprox = exp(x₀) * σ
plot!(xTransGrid, pdf(Normal(μtransApprox, σtransApprox), xTransGrid),
    label="approx", color=colors[2], linestyle=:solid)

plot(p1, p2, p3, layout=(1, 3), size=(1200, 400), bottommargin=5mm)
savefig(figFolder * "EKFapprox_moderate_poorpoint.png")


#####################
# UNSCENTED TRANSFORM
#####################

function CovMatEquiCorr(σₓ, ρ, pBlock)
    p = sum(pBlock)
    corrMat = zeros(p, p)
    count = 0
    println(length(ρ))
    for i in 1:length(ρ)
        idx = (1+count):(count+pBlock[i])
        corrMat[idx, idx] = ρ[i] * ones(pBlock[i], pBlock[i])
        count = count + pBlock[i]
    end
    corrMat[diagind(corrMat)] .= 1
    return diagm(σₓ) * corrMat * diagm(σₓ)
end

func(x) = exp.(x)
jacobianfunc(x) = ForwardDiff.jacobian(x -> func(x), x)

n = 2 # dimension of state
# α = 1; β = 0; κ = 0;
α = √(3 / n);
β = (3 / n) - 1;
κ = 0;
# α = 10^(-3); β = 2; κ = 0;
λ = α^2 * (n + κ) - n
γ = sqrt(n + λ)

pBlock = [2]

σₓ = [0.2, 0.3]
ρ = [0.9]
Ω̄ = CovMatEquiCorr(σₓ, ρ, pBlock)
μ̄ = 1 * ones(n)
#L̄ = cholesky(Ω̄).L;
#L̄ = sqrt(Ω̄)
V = eigvecs(Ω̄)
Λ = eigvals(Ω̄)
L̄ = V * Diagonal(sqrt.(Λ))
ωₘ = [λ / (n + λ); ones(2 * n) / (2 * (n + λ))]
ωₛ = [λ / (n + λ) + (1 - α^2 + β); ωₘ[2:end]]
X̄ = [μ̄ (μ̄ .+ γ * L̄) (μ̄ .- γ * L̄)] # sigma points
Ȳ = func.(X̄) # transformed sigma points

function UnscentedTransform(μ̄, Ω̄, func; α=1, β=0, κ=0, squareroot="eigen")
    n = length(μ̄)
    #α = √(3/n); β = (3/n)-1; κ = 0;
    λ = α^2 * (n + κ) - n
    γ = sqrt(n + λ)
    ωₘ = [λ / (n + λ); ones(2 * n) / (2 * (n + λ))]
    ωₛ = [λ / (n + λ) + (1 - α^2 + β); ωₘ[2:end]]

    if squareroot == "eigen"
        V = eigvecs(Ω̄)
        Λ = eigvals(Ω̄)
        L̄ = V * Diagonal(sqrt.(Λ))
    elseif squareroot == "chol"
        L̄ = cholesky(Ω̄).L
    elseif squareroot == "matrixsqrt"
        L̄ = sqrt(Ω̄)
    end

    X̄ = [μ̄ (μ̄ .+ γ * L̄) (μ̄ .- γ * L̄)]       # sigma points
    Ȳ = hcat(map(func, eachcol(X̄))...)  # transformed sigma points
    μUKF = zeros(n)
    for i in 1:length(ωₘ)
        μUKF += ωₘ[i] * Ȳ[:, i]
    end

    ΣUKF = zeros(n, n)
    for i in 1:length(ωₘ)
        ΣUKF += ωₛ[i] * (Ȳ[:, i] .- μUKF) * (Ȳ[:, i] .- μUKF)'
    end
    ΣUKF = Symmetric(ΣUKF)

    return μUKF, ΣUKF, X̄, Ȳ, ωₘ, ωₛ
end

# Simulated draws for moment matching
xDraws = rand(MvNormal(μ̄, Ω̄), 10000)
yDraws = func.(xDraws)
meanY = mean(yDraws, dims=2)[:]
covY = cov(yDraws, dims=2)

# Plot bivariate normal contours and sigma points

# hack: smallest ellipse is never plotted
chisqVals = quantile.(Chisq(length(μ̄)), reverse([eps() 0.1 0.25 0.5 0.75 0.95]))
pdfAtContours = (1 / √det(2 * π * Ω̄)) * exp.(-0.5 * chisqVals)

α = √(3 / n);
β = (3 / n) - 1;
κ = 0;
μUKF, ΣUKF, X̄, Ȳ, ωₘ, ωₛ = UnscentedTransform(μ̄, Ω̄, func; α=α, β=β, κ=κ)
σ = sqrt.(diag(Ω̄))
x1grid = range(μ̄[1] - 3 * σ[1], μ̄[1] + 3 * σ[1], length=200)
x2grid = range(μ̄[2] - 3 * σ[2], μ̄[2] + 3 * σ[2], length=200)
p1 = contour(size=(450, 450), x1grid, x2grid, (x1, x2) -> pdf(MvNormal(μ̄, Ω̄), [x1, x2]),
    levels=pdfAtContours[:], fill=true,
    color=:Blues,
    xlabel=L"z_1", ylabel=L"z_2", lw=1, linecolor=:black,
    xticks=0.5:0.5:2, yticks=0.5:0.5:2,
    colorbar=false, margin=5mm,
    title="original " * L"\alpha = %$(round(α, digits = 3)), \beta = %$(round(β, digits = 3))")
scatter!(X̄[1, :], X̄[2, :], markersize=8 * ωₛ / maximum(ωₛ),
    markerstrokecolor=colors[2], markercolor=colors[2],
    label="σ-points")

# Plot transformed variable contours and sigma points
transpdf(y1, y2) = pdf(MvNormal(μ̄, Ω̄), log.([y1, y2])) * (1 / y1) * (1 / y2)
σ = sqrt.(diag(Ω̄))
y1grid = func.(range(μ̄[1] - 5 * σ[1], μ̄[1] + 3 * σ[1], length=100))
y2grid = func.(range(μ̄[2] - 5 * σ[2], μ̄[2] + 3 * σ[2], length=100))
pdfAtContours = (1 / √det(2 * π * covY)) * exp.(-0.5 * chisqVals)
p2 = contour(size=(450, 450), y1grid, y2grid, (y1, y2) -> transpdf(y1, y2),
    levels=pdfAtContours[:], fill=true, color=:Blues,
    xlabel=L"y_1", ylabel=L"y_2", lw=1, linecolor=:black,
    colorbar=false, ylims=(0, 6), xlims=(0, 5),
    margin=5mm, title="transformed " * L"\alpha = %$(round(α, digits = 3)), \beta = %$(round(β, digits = 3))")
scatter!(Ȳ[1, :], Ȳ[2, :], markersize=20 * ωₛ / sum(ωₛ),
    markerstrokecolor=colors[2], markercolor=colors[2], label="σ-points")

# Plot linear (EKF) approximation
linfunc(x, x₀) = func(x₀) .+ jacobianfunc(x₀) * (x .- x₀)
μEKF = linfunc(μ̄, μ̄)
ΣEKF = jacobianfunc(μ̄) * Ω̄ * jacobianfunc(μ̄)'
pdfAtContours = (1 / √det(2 * π * ΣEKF)) * exp.(-0.5 * chisqVals)
p3 = contour(size=(450, 450), y1grid, y2grid,
    (y1, y2) -> pdf(MvNormal(μEKF, ΣEKF), [y1, y2]),
    levels=pdfAtContours[:], fill=true, color=:Blues,
    xlabel=L"y_1", ylabel=L"y_2", lw=1, linecolor=:black,
    colorbar=false, ylims=(0, 6), xlims=(0, 5),
    margin=5mm, title="EKF")


# Moment matching by simulation
pdfAtContours = (1 / √det(2 * π * covY)) * exp.(-0.5 * chisqVals)
p4 = contour(size=(450, 450), y1grid, y2grid,
    (y1, y2) -> pdf(MvNormal(meanY, covY), [y1, y2]),
    levels=pdfAtContours[:], fill=true, color=:Blues,
    xlabel=L"y_1", ylabel=L"y_2", lw=1, linecolor=:black,
    colorbar=false, ylims=(0, 6), xlims=(0, 5),
    margin=5mm, title="moment matched")


# UKF - first parameter scheme
α = √(3 / n);
β = (3 / n) - 1;
κ = 0;
μUKF, ΣUKF, X̄, Ȳ, ωₘ, ωₛ = UnscentedTransform(μ̄, Ω̄, func; α=α, β=β, κ=κ)
pdfAtContours = (1 / √det(2 * π * ΣUKF)) * exp.(-0.5 * chisqVals)
p5 = contour(size=(450, 450), y1grid, y2grid,
    (y1, y2) -> pdf(MvNormal(μUKF, ΣUKF), [y1, y2]),
    levels=pdfAtContours[:], fill=true, color=:Blues,
    xlabel=L"y_1", ylabel=L"y_2", lw=1, linecolor=:black,
    colorbar=false, ylims=(0, 6), xlims=(0, 5),
    margin=5mm,
    title="UKF " * L"\alpha = %$(round(α, digits = 3)), \beta = %$(round(β, digits = 3))")
scatter!(Ȳ[1, :], Ȳ[2, :], markersize=20 * ωₛ / sum(ωₛ),
    markerstrokecolor=colors[2], markercolor=colors[2],
    label="σ-points")

# UKF - second parameter scheme
α = 1;
β = 0;
κ = 0;
μUKF, ΣUKF, X̄, Ȳ, ωₘ, ωₛ = UnscentedTransform(μ̄, Ω̄, func; α=α, β=β, κ=κ)
pdfAtContours = (1 / √det(2 * π * ΣUKF)) * exp.(-0.5 * chisqVals)
p6 = contour(size=(450, 450), y1grid, y2grid,
    (y1, y2) -> pdf(MvNormal(μUKF, ΣUKF), [y1, y2]),
    levels=pdfAtContours[:], fill=true, color=:Blues,
    xlabel=L"y_1", ylabel=L"y_2", lw=1, linecolor=:black,
    colorbar=false, ylims=(0, 6), xlims=(0, 5),
    margin=5mm,
    title="UKF " * L"\alpha = %$(round(α, digits = 3)), \beta = %$(round(β, digits = 3))")
scatter!(Ȳ[1, :], Ȳ[2, :], markersize=20 * ωₛ / sum(ωₛ),
    markerstrokecolor=colors[2], markercolor=colors[2],
    label="σ-points")


plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), size=(1200, 800), margin=5mm)
savefig(figFolder * "UKFapprox.pdf")


## Grouping

# Plot Poisson likelihood contributions for different y values on the same plot
θgrid = range(0.001, 10, length=1000)
p1 = plot(θgrid, pdf.(Poisson.(θgrid), 0), xlabel=L"\theta", ylabel=L"p(y|\theta)",
    label=L"y=0", title="Likelihood contributions", legend=:topright, size=(500, 400), color=colors[1], xticks=0:2:10)
plot!(θgrid, pdf.(Poisson.(θgrid), 2), color=colors[2], label=L"y=2")
plot!(θgrid, pdf.(Poisson.(θgrid), 5), color=colors[3], label=L"y=5")
savefig(figFolder * "PoissonLikeContributions.svg")

# Posterior distribution from the above likelihood contributions with a Gamma(2, 1) prior
prior = Gamma(2, 1)
# plot the prior
p1 = plot(θgrid, pdf(prior, θgrid), xlabel=L"\theta", ylabel=L"p(\theta \vert y)",
    label=L"\mathrm{Gamma}(2, 1)" * " prior", title="Posterior - informative prior", legend=:topright, size=(500, 400), color=:black, xticks=0:2:10)
# plot the posterior for y=0, 2, 5 normalized to have area 1
for (j, y) in enumerate([0, 2, 5])
    likelihood = pdf.(Poisson.(θgrid), y)
    unnormalizedPosterior = likelihood .* pdf.(prior, θgrid)
    posterior = unnormalizedPosterior / sum(unnormalizedPosterior) / (θgrid[2] - θgrid[1])
    plot!(θgrid, posterior, label="y=$y", color=colors[j])
end
p1
savefig(figFolder * "PoissonPosteriorInformativePrior.svg")

# Posterior distribution with a Gamma(0.01, 0.01) prior
prior = Gamma(0.02, 1 / 0.01)
# plot the prior
p1 = plot(θgrid, pdf(prior, θgrid), xlabel=L"\theta", ylabel=L"p(\theta \vert y)",
    label=L"\mathrm{Gamma}(0.02, 0.01)" * " prior", title="Posterior - non-informative prior", legend=:topright, size=(500, 400), color=:black, ylim=(0, 0.7), xlim=(0, 10),
    xticks=0:2:10)
# plot the posterior for y=0, 2, 5 normalized to have area 1
for (j, y) in enumerate([0, 2, 5])
    likelihood = pdf.(Poisson.(θgrid), y)
    unnormalizedPosterior = likelihood .* pdf.(prior, θgrid)
    posterior = unnormalizedPosterior / sum(unnormalizedPosterior) / (θgrid[2] - θgrid[1])
    plot!(θgrid, posterior, label="y=$y", color=colors[j])
end
p1
savefig(figFolder * "PoissonPosteriorNonInformativePrior.svg")

# plot the likelihood contributions from groups of observation
# normalize the likelihood contributions to have area 1 for better visualization
pdfs = pdf.(Poisson.(θgrid), 0)
normpdf = pdfs / sum(pdfs) / (θgrid[2] - θgrid[1])
p1 = plot(θgrid, normpdf, xlabel=L"\theta", ylabel=L"p(y|\theta)" * " (normalized)",
    label=L"\mathbf{y}=(0)", title="Likelihood contributions - grouping", legend=:topright, size=(500, 400), color=colors[1], xticks=0:2:10)
pdfs = pdf.(Poisson.(θgrid), 0) .* pdf.(Poisson.(θgrid), 1)
normpdf = pdfs / sum(pdfs) / (θgrid[2] - θgrid[1])
plot!(θgrid, normpdf, color=colors[2], label=L"\mathbf{y}=(0,1)")
# add a three observation case with 0,1,2
pdfs = pdf.(Poisson.(θgrid), 0) .* pdf.(Poisson.(θgrid), 1) .* pdf.(Poisson.(θgrid), 2)
normpdf = pdfs / sum(pdfs) / (θgrid[2] - θgrid[1])
plot!(θgrid, normpdf, color=colors[3], label=L"\mathbf{y}=(0,1,2)")

savefig(figFolder * "PoissonLikeContributionsGroup.svg")

## Scaling 

# KL divergence between two Poisson distributions
function KLPoisson(λ₁, λ₂)
    return λ₁ * log(λ₁ / λ₂) + λ₂ - λ₁
end

function plotPoisson(λ₁, ϵ, scaling, ygrid; legend=false)

    if scaling
        ϵ = sqrt(λ₁) * ϵ
        titleadd = "Scaled unit change: ϵ = $(round(ϵ, digits=2))\n"
    else
        titleadd = "Unscaled unit change: ϵ = $(ϵ)\n"
    end
    λ₂ = λ₁ + ϵ
    kl = KLPoisson(λ₁, λ₂)

    ygrid = 0:1:10
    p1 = plot(ygrid, pdf.(Poisson(λ₁), ygrid), xlabel=L"y", ylabel=L"p(y|\lambda)",
        label=L"\lambda_1=%$(round(λ₁, digits = 3))",
        title=titleadd * "KL = $(round(kl, digits = 3))",
        legend=legend, size=(500, 400), color=colors[1],
        xticks=0:2:10)
    scatter!(ygrid, pdf.(Poisson(λ₁), ygrid), markerstrokecolor=colors[1],
        color=colors[1], label="")
    plot!(ygrid, pdf.(Poisson(λ₂), ygrid), color=colors[2],
        label=L"\lambda_2=%$(round(λ₂, digits = 3))")
    scatter!(ygrid, pdf.(Poisson(λ₂), ygrid), markerstrokecolor=colors[2],
        color=colors[2], label="")

end

ϵ = 1
ygrid = 0:1:10
λs = [1.0, 2.0, 5.0]
scaling = false
plts = []
for (j, λ₁) in enumerate(λs)
    if j == 1
        legend = :topright
    else
        legend = :topright
    end
    push!(plts, plotPoisson(λ₁, ϵ, scaling, ygrid; legend=legend))
end
plot(plts..., layout=(1, 3), size=(1200, 400), margin=5mm)
savefig(figFolder * "KLPoissonLambdaNoScaling.svg")

scaling = true
plts = []
for (j, λ₁) in enumerate(λs)
    if j == 1
        legend = :topright
    else
        legend = :topright
    end
    push!(plts, plotPoisson(λ₁, ϵ, scaling, ygrid; legend=legend))
end
plot(plts..., layout=(1, 3), size=(1200, 400), margin=5mm)
savefig(figFolder * "KLPoissonLambdaScaling.svg")



## now with λ = exp(θ) and a change of ϵ in theta instead of λ
# KL divergence between two Poisson distributions
function KLPoissonLog(θ₁, θ₂)
    return exp(θ₁) * (θ₁ - θ₂) + exp(θ₂) - exp(θ₁)
end

function plotPoissonLog(θ₁, ϵ, scaling, ygrid; legend=false)

    if scaling
        ϵ = exp(-θ₁ / 2) * ϵ
        titleadd = "Scaled unit change: ϵ = $(round(ϵ, digits=2))\n"
    else
        titleadd = "Unscaled unit change: ϵ = $(ϵ)\n"
    end
    θ₂ = θ₁ + ϵ
    kl = KLPoissonLog(θ₁, θ₂)
    λ₁ = exp(θ₁)
    λ₂ = exp(θ₂)

    ygrid = 0:1:10
    p1 = plot(ygrid, pdf.(Poisson(λ₁), ygrid), xlabel=L"y", ylabel=L"p(y|\lambda)",
        label=L"\lambda_1=%$(round(λ₁, digits = 3))",
        title=titleadd * "KL = $(round(kl, digits = 3))",
        legend=legend, size=(500, 400), color=colors[1],
        xticks=0:2:10)
    scatter!(ygrid, pdf.(Poisson(λ₁), ygrid), markerstrokecolor=colors[1],
        color=colors[1], label="")
    plot!(ygrid, pdf.(Poisson(λ₂), ygrid), color=colors[2],
        label=L"\lambda_2=%$(round(λ₂, digits = 3))")
    scatter!(ygrid, pdf.(Poisson(λ₂), ygrid), markerstrokecolor=colors[2],
        color=colors[2], label="")

end

ϵ = 1
ygrid = 0:1:10
θs = log.([1.0, 3.0, 5.0])
scaling = false
plts = []
for (j, θ₁) in enumerate(θs)
    if j == 1
        legend = :topright
    else
        legend = :topright
    end
    push!(plts, plotPoissonLog(θ₁, ϵ, scaling, ygrid; legend=legend))
end
plot(plts..., layout=(1, 3), size=(1200, 400), margin=5mm)
savefig(figFolder * "KLPoissonLogLambdaNoScaling.svg")

scaling = true
plts = []
for (j, θ₁) in enumerate(θs)
    if j == 1
        legend = :topright
    else
        legend = :topright
    end
    push!(plts, plotPoissonLog(θ₁, ϵ, scaling, ygrid; legend=legend))
end
plot(plts..., layout=(1, 3), size=(1200, 400), margin=5mm)
savefig(figFolder * "KLPoissonLogLambdaScaling.svg")
