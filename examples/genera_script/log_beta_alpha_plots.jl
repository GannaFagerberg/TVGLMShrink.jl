p_beta_log = plot(
    beta_med,
    ribbon = (
        beta_med .- beta_lo,
        beta_hi .- beta_med
    ),
    yscale = :log10,
    xlabel = "Observation",
    ylabel = "β",
    title = "Posterior of β",
    label = "Posterior median",
    linewidth = 2
)

hline!(
    p_beta_log,
    [1.0],
    linestyle = :dash,
    label = "β = 1"
)


p_kappa_log = plot(
    kappa_med,
    ribbon = (
        kappa_med .- kappa_lo,
        kappa_hi .- kappa_med
    ),
    yscale = :log10,
    xlabel = "Observation",
    ylabel = "κ",
    title = "Posterior of κ",
    label = "Posterior median",
    linewidth = 2
)

hline!(
    p_kappa_log,
    [1.0],
    linestyle = :dash,
    label = "κ = 1"
)




gamma_post =
    log.(kappa_post)

gamma_lo, gamma_med, gamma_hi =
    posterior_summary(gamma_post)

p_gamma = plot(
    gamma_med,
    ribbon = (
        gamma_med .- gamma_lo,
        gamma_hi .- gamma_med
    ),
    xlabel = "Observation",
    ylabel = "γ = log κ",
    title = "Posterior of log-precision",
    label = "Posterior median",
    linewidth = 2
)


p_alpha_log = plot(
    alpha_med,
    ribbon = (
        alpha_med .- alpha_lo,
        alpha_hi .- alpha_med
    ),
    yscale = :log10,
    xlabel = "Observation",
    ylabel = "α",
    title = "Posterior of α",
    label = "Posterior median",
    linewidth = 2
)

hline!(
    p_alpha_log,
    [1.0],
    linestyle = :dash,
    label = "α = 1"
)

plot(
    p_alpha_log,
    p_beta_log,
    p_kappa_log,
    p_gamma;
    layout = (2, 2),
    size = (1100, 750)
)