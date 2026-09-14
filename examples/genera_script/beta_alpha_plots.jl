
θpost=θpost_laplace
Tg = length(Z[1])              # probably 87
M  = size(θpost, 3)            # 2000

# Check whether posterior contains t = 0
offset = size(θpost, 1) == Tg + 1 ? 1 : 0

# Number of original observations represented by groups
N = sum(size(Z[1][t], 1) for t in 1:Tg)

# posterior arrays:
# observation × posterior draw
mu_post    = Matrix{Float64}(undef, N, M)
kappa_post = Matrix{Float64}(undef, N, M)
alpha_post = Matrix{Float64}(undef, N, M)
beta_post  = Matrix{Float64}(undef, N, M)

row = 1

for t in 1:Tg

    Zmu = Z[1][t]       # g × 2
    Zk  = Z[2][t]       # g × 1

    g = size(Zmu, 1)

    for m in 1:M

        # state corresponding to observation group t
        tt = t + offset

        beta_t =
            θpost[tt, 1:2, m]

        gamma_t =
            θpost[tt, 3:3, m]

        # linear predictors
        eta_mu =
            Zmu * beta_t

        eta_kappa =
            Zk * gamma_t

        # links:
        # mu    = logistic(eta_mu)
        # kappa = exp(eta_kappa)
        mu =
            1.0 ./ (1.0 .+ exp.(-eta_mu))

        kappa =
            exp.(eta_kappa)

        # Beta shape parameters
        alpha =
            mu .* kappa

        beta =
            (1.0 .- mu) .* kappa

        idx = row:(row + g - 1)

        mu_post[idx, m]    .= mu
        kappa_post[idx, m] .= kappa
        alpha_post[idx, m] .= alpha
        beta_post[idx, m]  .= beta
    end

    row += g
end

size(mu_post)
size(kappa_post)
size(alpha_post)
size(beta_post)

using Statistics

mu_q =
    mapslices(
        x -> quantile(x, [0.025, 0.5, 0.975]),
        mu_post;
        dims = 2
    )

kappa_q =
    mapslices(
        x -> quantile(x, [0.025, 0.5, 0.975]),
        kappa_post;
        dims = 2
    )

alpha_q =
    mapslices(
        x -> quantile(x, [0.025, 0.5, 0.975]),
        alpha_post;
        dims = 2
    )

beta_q =
    mapslices(
        x -> quantile(x, [0.025, 0.5, 0.975]),
        beta_post;
        dims = 2
    )

using Statistics
using Plots

function posterior_summary(A)

    n = size(A, 1)

    q025 = zeros(n)
    q50  = zeros(n)
    q975 = zeros(n)

    for t in 1:n
        q025[t], q50[t], q975[t] =
            quantile(
                view(A, t, :),
                [0.025, 0.5, 0.975]
            )
    end

    return q025, q50, q975
end


mu_lo, mu_med, mu_hi =
    posterior_summary(mu_post)

kappa_lo, kappa_med, kappa_hi =
    posterior_summary(kappa_post)

alpha_lo, alpha_med, alpha_hi =
    posterior_summary(alpha_post)

beta_lo, beta_med, beta_hi =
    posterior_summary(beta_post)

p_mu = plot(
    mu_med,
    ribbon = (
        mu_med .- mu_lo,
        mu_hi .- mu_med
    ),
    label = "Posterior median",
    xlabel = "Observation",
    ylabel = "μ",
    title = "Posterior of μ",
    linewidth = 2
)

p_kappa = plot(
    kappa_med,
    ribbon = (
        kappa_med .- kappa_lo,
        kappa_hi .- kappa_med
    ),
    label = "Posterior median",
    xlabel = "Observation",
    ylabel = "κ",
    title = "Posterior of κ",
    linewidth = 2
)


p_alpha = plot(
    alpha_med,
    ribbon = (
        alpha_med .- alpha_lo,
        alpha_hi .- alpha_med
    ),
    label = "Posterior median",
    xlabel = "Observation",
    ylabel = "α",
    title = "Posterior of α",
    linewidth = 2
)

p_beta = plot(
    beta_med,
    ribbon = (
        beta_med .- beta_lo,
        beta_hi .- beta_med
    ),
    label = "Posterior median",
    xlabel = "Observation",
    ylabel = "β",
    title = "Posterior of β",
    linewidth = 2
)


plot(
    p_mu,
    p_kappa,
    p_alpha,
    p_beta;
    layout = (2, 2),
    size = (1000, 700)
)


###################################
p_alpha_log = plot(
    alpha_med,
    ribbon = (
        alpha_med .- alpha_lo,
        alpha_hi .- alpha_med
    ),
    yscale = :log10,
    ylabel = "α",
    xlabel = "Observation",
    label = "Posterior median",
    title = "Posterior of α"
)

hline!(p_alpha_log, [1.0], linestyle = :dash, label = "α = 1")


sd_post =
    sqrt.(
        mu_post .* (1 .- mu_post) ./
        (kappa_post .+ 1)
    )

    sd_lo, sd_med, sd_hi =
    posterior_summary(sd_post)

    p_sd = plot(
    sd_med,
    ribbon = (
        sd_med .- sd_lo,
        sd_hi .- sd_med
    ),
    label = "Posterior median",
    xlabel = "Observation",
    ylabel = "Conditional SD",
    title = "Posterior conditional standard deviation"
)

    abs_resid =
    abs.(y .- mu_med)

    std_resid =
    (y .- mu_med) ./
    sqrt.(
        mu_med .* (1 .- mu_med) ./
        (kappa_med .+ 1)
    )


    p_resid = plot(
    std_resid,
    label = false,
    xlabel = "Observation",
    ylabel = "Standardized residual",
    title = "Standardized residuals"
)

hline!(
    p_resid,
    [0.0],
    linestyle = :dash,
    label = false
)



lower_band = mu_med .- 2 .* sd_med
upper_band = mu_med .+ 2 .* sd_med

p = plot(
    mu_med,
    ribbon = (
        mu_med .- lower_band,
        upper_band .- mu_med
    ),
    label = "μ ± 2 SD",
    xlabel = "Observation",
    ylabel = "y"
)

scatter!(
    p,
    y,
    label = "Observed y",
    markersize = 2
)



std_resid =
    (y .- mu_med) ./
    sqrt.(
        mu_med .* (1 .- mu_med) ./
        (kappa_med .+ 1)
    )

p = scatter(
    std_resid,
    markersize = 2,
    xlabel = "Observation",
    ylabel = "Standardized residual",
    title = "Standardized residuals",
    label = false
)

hline!(p, [0.0], linestyle=:solid, label=false)
hline!(p, [-2.0, 2.0], linestyle=:dash, label=["-2" "+2"])
hline!(p, [-3.0, 3.0], linestyle=:dot, label=["-3" "+3"])