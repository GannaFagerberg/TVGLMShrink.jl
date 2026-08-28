using Random
using Serialization

# ============================================================
# Gamma IPLF: repeat over different random seeds
# NO scaling
# ============================================================

methodlabel = "IPLF"

scaling = :none

algoSettings = (
    ;
    algoSettings...,
    scaling = scaling,
    stateSamplingMethod = :ffbs_slr
)

# ============================================================
# Group raw observations outside Gibbs
# ============================================================

obsTransform = GammaSuffStatsAveraged()

slrObs = prepare_observation_transform(
    obsTransform,
    Y,
    condMean,
    condCov,
    nPerGroup
)

modelSettings = (
    ;
    modelSettings...,
    slrObs = slrObs
)

dataSettings = (
    y = y,
    X = X,
    covSel = covSel,
    nPerGroup = nPerGroup
)

# ============================================================
# Seeds
# ============================================================

seeds = [234, 345, 456, 567, 678]

# ============================================================
# Save folder
# ============================================================

save_dir = "results/gamma_iplf_noscaling_seeds"
mkpath(save_dir)

# Keep results in memory
iplf_results = Dict()
iplf_quantiles = Dict()

# ============================================================
# Plot container
# ============================================================
# ============================================================
# Plot container: one complete figure per seed
# ============================================================
# ============================================================
# Plot settings
# ============================================================

plot_truth = true

# ============================================================
# Plot settings
# ============================================================

plot_truth = true

iplf_plots = Dict()

# Construct truth plot only once
plt_truth =
    plot_truth ? plot_param_path_betareg(β, γ) : nothing


for (k, seed) in enumerate(seeds)

    println("\n============================================")
    println("Running Gamma IPLF with seed = $seed")
    println("Scaling = $scaling")
    println("============================================")

    Random.seed!(seed)

    θpost, groupSizes, nFailure =
        GibbsTVGLM(
            dataSettings,
            priorSettings,
            modelSettings,
            algoSettings
        )

    # --------------------------------------------------------
    # Failure percentage
    # --------------------------------------------------------

    prcFailure =
        100 * nFailure[] /
        (algoSettings.nBurn + algoSettings.nIter)

    println(
        "$(algoSettings.stateSamplingMethod), seed $seed: " *
        "failed at $(prcFailure)% of the simulated trajectories"
    )

    # --------------------------------------------------------
    # Posterior quantiles
    # --------------------------------------------------------

    quant_paramtime =
        quantile_multidim(
            θpost,
            [0.025, 0.5, 0.975],
            dims = 3
        )

    # --------------------------------------------------------
    # Store in memory
    # --------------------------------------------------------

    iplf_results[seed] = (
        θpost = θpost,
        groupSizes = groupSizes,
        nFailure = nFailure[],
        prcFailure = prcFailure
    )

    iplf_quantiles[seed] = quant_paramtime

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result_seed = (
        model = :gamma,
        method = :IPLF,
        scaling = scaling,
        seed = seed,
        θpost = θpost,
        groupSizes = groupSizes,
        nFailure = nFailure[],
        prcFailure = prcFailure,
        quant_paramtime = quant_paramtime
    )

    serialize(
        joinpath(
            save_dir,
            "Gamma_IPLF_noscaling_seed_$(seed).jls"
        ),
        result_seed
    )

    # --------------------------------------------------------
    # Posterior plot for THIS SEED
    # --------------------------------------------------------

    plt_seed = PlotPostParamEvolution(
        quant_paramtime,
        "",
        groupSizes;
        titles = titles,
        dateVec = dateVec,
        interpMethod = interpMethod,
        plot_t0 = keep_t0,
        interval_style = :solid,
        lw = 2,
        c = seed_colors[k]
    )

    # --------------------------------------------------------
    # Add true parameter paths
    # --------------------------------------------------------

    if plot_truth

        for j in eachindex(plt_seed)

            truth_subplot = plt_truth.subplots[j]

            for s in truth_subplot.series_list

                plot!(
                    plt_seed[j],
                    s[:x],
                    s[:y];
                    c = :black,
                    lw = 2,
                    ls = :dash,
                    label = "Truth"
                )

            end

        end

    end

    # --------------------------------------------------------
    # Combine parameter panels for THIS seed
    # --------------------------------------------------------

    plt_seed_combined = plot(
        plt_seed...;
        layout = (length(plt_seed), 1),
        size = (1000, 400 * length(plt_seed)),
        margin = 3mm,
        plot_title = "Gamma IPLF — seed $seed"
    )

    # Store plot
    iplf_plots[seed] = plt_seed_combined

    # Display
    display(plt_seed_combined)

    # Save plot for this seed
    savefig(
        plt_seed_combined,
        joinpath(
            save_dir,
            "Gamma_IPLF_noscaling_seed_2$(seed).pdf"
        )
    )

end
# ============================================================
# Individual plots
# ============================================================

display(iplf_plots[234])
display(iplf_plots[345])
display(iplf_plots[456])
display(iplf_plots[567])
display(iplf_plots[678])

# ============================================================
# Combined plot
# ============================================================
# ============================================================
# Combined plot: all parameter panels for all seeds
# ============================================================
# ============================================================
# Overlay all seeds
# ============================================================
# ============================================================
# Build NEW overlay from current seed results
# ============================================================

plt_iplf_overlay = nothing

for (k, seed) in enumerate(seeds)

    quant_paramtime = iplf_quantiles[seed]
    groupSizes_seed = iplf_results[seed].groupSizes

    if k == 1

        plt_iplf_overlay = PlotPostParamEvolution(
            quant_paramtime,
            "",
            groupSizes_seed;
            titles = titles,
            dateVec = dateVec,
            interpMethod = interpMethod,
            plot_t0 = keep_t0,
            interval_style = :solid,
            lw = 2,
            c = seed_colors[k]
        )

    else

        PlotPostParamEvolution!(
            plt_iplf_overlay,
            quant_paramtime,
            "",
            groupSizes_seed;
            dateVec = dateVec,
            interpMethod = interpMethod,
            plot_t0 = keep_t0,
            interval_style = :solid,
            lw = 2,
            c = seed_colors[k]
        )

    end
end

# ============================================================
# Add truth
# ============================================================

if plot_truth

    plt_truth = plot_param_path_betareg(β, γ)

    for j in eachindex(plt_iplf_overlay)

        truth_subplot = plt_truth.subplots[j]

        # Usually the truth plot has one trajectory per subplot
        for (i, s) in enumerate(truth_subplot.series_list)

            plot!(
                plt_iplf_overlay[j],
                s[:x],
                s[:y];
                c = :black,
                lw = 2,
                ls = :dash,
                label = i == 1 ? "Truth" : ""
            )

        end
    end
end

# ============================================================
# Combined overlay
# ============================================================

plt_iplf_combined = plot(
    plt_iplf_overlay...;
    layout = (length(plt_iplf_overlay), 1),
    size = (1000, 400 * length(plt_iplf_overlay)),
    margin = 3mm
)

display(plt_iplf_combined)
# ============================================================
# Save figure
# ============================================================

savefig(
    plt_iplf_combined,
    joinpath(
        save_dir,
        "Gamma_IPLF_noscaling_seeds2.pdf"
    )
)

# ============================================================
# Save all seeds + true simulated quantities
# ============================================================

serialize(
    joinpath(
        save_dir,
        "Gamma_IPLF_all_seeds_noscaling.jls"
    ),
    (
        model = :gamma,
        method = :IPLF,
        scaling = scaling,
        results = iplf_results,
        quantiles = iplf_quantiles,
        seeds = seeds,

        β_true = β,
        γ_true = γ,

        μtime = μtime,
        ψtime = ψtime,
        αtime = αtime,
        βtime = βtime,

        y = y
    )
)

