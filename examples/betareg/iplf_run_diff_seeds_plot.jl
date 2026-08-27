using Random
using Serialization

# ============================================================
# IPLF: repeat over different random seeds
# ============================================================

methodlabel = "IPLF"

algoSettings = (;
    algoSettings...,
    scaling = scaling,
    stateSamplingMethod = :ffbs_slr
)

dataSettings = (
    y = y,
    X = X,
    covSel = covSel,
    nPerGroup = nPerGroup
)

# Five seeds
seeds = [234, 345, 456, 567, 678]

# Folder for saved MCMC results
save_dir = "results/iplf_seeds"
mkpath(save_dir)

# Keep summaries/results available in memory too
iplf_results = Dict()
iplf_quantiles = Dict()

# ------------------------------------------------------------
# Start plot ONCE: only true parameter paths
# ------------------------------------------------------------

#plt_iplf = plot_param_path_betareg(β, γ)
plt_iplf = nothing

# Use different colors for the five seeds
seed_colors = [
    colors[mod1(k, length(colors))]
    for k in eachindex(seeds)
]

# ============================================================
# Run IPLF for each seed
# ============================================================

for (k, seed) in enumerate(seeds)

    println("\n============================================")
    println("Running IPLF with seed = $seed")
    println("============================================")

    Random.seed!(seed)

    θpost, nFailure =
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
        Hpost = Hpost,
        ϕpost = ϕpost,
        σ²ₙpost = σ²ₙpost,
        μpost = μpost,
        groupSizes = groupSizes,
        nFailure = nFailure[],
        prcFailure = prcFailure
    )

    iplf_quantiles[seed] = quant_paramtime

    # --------------------------------------------------------
    # Save complete result for this seed
    # --------------------------------------------------------

    result_seed = (
        seed = seed,
        θpost = θpost,
        Hpost = Hpost,
        ϕpost = ϕpost,
        σ²ₙpost = σ²ₙpost,
        μpost = μpost,
        groupSizes = groupSizes,
        nFailure = nFailure[],
        prcFailure = prcFailure,
        quant_paramtime = quant_paramtime
    )

    serialize(
        joinpath(
            save_dir,
            "IPLF_seed_$(seed).jls"
        ),
        result_seed
    )

    # --------------------------------------------------------
    # Add THIS seed to the SAME plot
    # --------------------------------------------------------

    # --------------------------------------------------------
# Initialize plot with first seed, then overlay the others
# --------------------------------------------------------

    if k == 1

        plt_iplf = PlotPostParamEvolution(
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

    else

        PlotPostParamEvolution!(
            plt_iplf,
            quant_paramtime,
            "",
            groupSizes;
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
# Show combined plot
# ============================================================

display(
    plot(
        plt_iplf...,
        layout = (length(plt_iplf), 1),
        size = (1000, 400 * length(plt_iplf)),
        margin = 3mm
    )
)

# Save combined figure
plt_iplf_combined = plot(
    plt_iplf...;
    layout = (length(plt_iplf), 1),
    size = (1000, 1200)
)

savefig(
    plt_iplf_combined,
    joinpath(save_dir, "IPLF_layoff_noscaling_standardised_stability.pdf")
)

serialize(
    joinpath(save_dir, "IPLF_all_seeds_diag_scaling_std.jls"),
    (
        results = iplf_results,
        quantiles = iplf_quantiles,
        seeds = seeds
    )
)

# under full now