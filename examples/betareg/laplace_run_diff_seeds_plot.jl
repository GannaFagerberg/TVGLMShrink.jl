using Random
using Serialization

# ============================================================
# Laplace: repeat over seeds
# ============================================================

methodlabel = "Laplace"

algoSettings_la = (;
    algoSettings...,
    scaling = scaling,
    stateSamplingMethod = :ffbs_laplace
)

dataSettings = (
    y = y,
    X = X,
    covSel = covSel,
    nPerGroup = nPerGroup
)

save_dir = "results/beta_seed_comparison"
mkpath(save_dir)

laplace_results   = Dict()
laplace_quantiles = Dict()

# One plot containing all Laplace runs
plt_la = plot_param_path_betareg(β, γ)

for (k, seed) in enumerate(seeds)

    println("\n============================================")
    println("Running Laplace with seed = $seed")
    println("============================================")

    Random.seed!(seed)

    θpost,
    Hpost,
    ϕpost,
    σ²ₙpost,
    μpost,
    groupSizes,
    nFailure =
        GibbsTVGLM(
            dataSettings,
            priorSettings,
            modelSettings,
            algoSettings_la
        )

    prcFailure =
        100 * nFailure[] /
        (algoSettings_la.nBurn + algoSettings_la.nIter)

    println(
        "$(algoSettings_la.stateSamplingMethod), seed $seed: " *
        "failed at $(prcFailure)% of simulated trajectories"
    )

    # Posterior quantiles
    quant_paramtime =
        quantile_multidim(
            θpost,
            [0.025, 0.5, 0.975],
            dims = 3
        )

    # ---------------------------------------------
    # Store in memory
    # ---------------------------------------------

    laplace_results[seed] = (
        θpost = θpost,
        Hpost = Hpost,
        ϕpost = ϕpost,
        σ²ₙpost = σ²ₙpost,
        μpost = μpost,
        groupSizes = groupSizes,
        nFailure = nFailure[],
        prcFailure = prcFailure
    )

    laplace_quantiles[seed] = quant_paramtime

    # ---------------------------------------------
    # Save this seed to disk
    # ---------------------------------------------

    result_seed = (
        method = :laplace,
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
            "Laplace_seed_$(seed).jls"
        ),
        result_seed
    )

    # ---------------------------------------------
    # Add to Laplace-only plot
    # ---------------------------------------------

    PlotPostParamEvolution!(
        plt_la,
        quant_paramtime,
        #"Laplace seed $seed",
        "",
        groupSizes;
        dateVec = dateVec,
        interpMethod = interpMethod,
        plot_t0 = keep_t0,
        interval_style = :solid,
        lw = 2,
        c = colors[mod1(k, length(colors))]
    )
end

display(plt_la)

savefig(
    plt_la,
    joinpath(
        save_dir,
        "Laplace_five_seeds_overlay.pdf"
    )
)


serialize(
    joinpath(save_dir, "Laplace_all_seeds.jls"),
    (
        results = laplace_results,
        quantiles = laplace_quantiles,
        seeds = seeds
    )
)