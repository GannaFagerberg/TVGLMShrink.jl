# ============================================================
# Separate IPLF vs Laplace comparison for each seed
# ============================================================

comparison_plots = Dict()

for seed in seeds

    # Start from truth only
    plt_seed = plot_param_path_betareg(β, γ)

    # --------------------------------------------------------
    # IPLF
    # --------------------------------------------------------

    PlotPostParamEvolution!(
        plt_seed,
        iplf_quantiles[seed],
        "IPLF",
        iplf_results[seed].groupSizes;
        dateVec = dateVec,
        interpMethod = interpMethod,
        plot_t0 = keep_t0,
        interval_style = :solid,
        lw = 2,
        c = colors[4]
    )

    # --------------------------------------------------------
    # Laplace
    # --------------------------------------------------------

    PlotPostParamEvolution!(
        plt_seed,
        laplace_quantiles[seed],
        "Laplace",
        laplace_results[seed].groupSizes;
        dateVec = dateVec,
        interpMethod = interpMethod,
        plot_t0 = keep_t0,
        interval_style = :solid,
        lw = 2,
        c = colors[2]
    )

    # Keep plot in memory
    comparison_plots[seed] = plt_seed

    # Display
    display(plt_seed)

    # Save
    savefig(
        plt_seed,
        joinpath(
            save_dir,
            "IPLF_vs_Laplace_seed_$(seed).pdf"
        )
    )
end

### One plot
# ============================================================
# Combine seed comparisons:
#   column 1 = mean
#   column 2 = precision
#   one legend only
# ============================================================

panels = []

for (i, seed) in enumerate(seeds)

    plt_seed = comparison_plots[seed]

    # --------------------------------------------------------
    # Mean: first subplot
    # --------------------------------------------------------
    p_mean = plot(
        plt_seed.subplots[1];
        legend = false,
        #title = "Seed $(seed)"
        title = ""
    )

    # --------------------------------------------------------
    # Precision: second subplot
    # --------------------------------------------------------
    p_precision = plot(
        plt_seed.subplots[2];
        legend = (i == 1),          # legend only once
        title = i == 1 ? "Precision" : ""
    )

    push!(panels, p_mean)
    push!(panels, p_precision)
end


# ------------------------------------------------------------
# Add column title to mean column only once
# ------------------------------------------------------------

plot!(
    panels[1];
    title = "Mean"
)

# ============================================================
# Final figure
# ============================================================

plt_all_seeds = plot(
    panels...;
    layout = (length(seeds), 2),
    size = (1000, 300 * length(seeds))
)

display(plt_all_seeds)

savefig(
    plt_all_seeds,
    joinpath(
        save_dir,
        "IPLF_vs_Laplace_all_seeds_panels.pdf"
    )
)