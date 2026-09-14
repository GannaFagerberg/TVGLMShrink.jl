using Random
using Plots

# ============================================================
# Seeds
# ============================================================

seeds = [10, 20, 30, 40]
nSeeds = length(seeds)

# Store complete posterior draws and posterior quantiles
θpost_seed = Vector{Any}(undef, nSeeds)
quant_seed = Vector{Any}(undef, nSeeds)

# ============================================================
# Run Gibbs sampler for all seeds
# ============================================================

for (i, seed) in enumerate(seeds)

    println()
    println("==============================================")
    println("Running seed = $seed")
    println("==============================================")

    Random.seed!(seed)

    # --------------------------------------------------------
    # YOUR CURRENT GIBBS CALL
    # Keep all arguments exactly as in your present script
    # --------------------------------------------------------

    θpost_seed[i],_,_,_ = GibbsTVGLM(
        dataSettings_laplace,priorSettings_laplace,modelSettings,algoSettings_laplace
    )

    # --------------------------------------------------------
    # Posterior quantiles
    #
    # 1 = 2.5%
    # 2 = median
    # 3 = 97.5%
    # --------------------------------------------------------

    quant_seed[i] =
        quantile_multidim(
            θpost_seed[i],
            [0.025, 0.5, 0.975],
            dims = 3
        )
end


# ============================================================
# Start from your existing true parameter paths
# ============================================================
seeds = [1, 2, 3, 4]

for (i, seed) in enumerate(seeds)

    Random.seed!(seed)

    quant_seed[i] = quantile_multidim(
        θpost_seed[i],
        [0.025, 0.5, 0.975],
        dims = 3
    )
end


# ------------------------------------------------------------
# Overlay all seeds
# ------------------------------------------------------------

plt_overlay = plot(layout = (3, 1),size = (900, 650),legend = :topright)

for i in eachindex(seeds)

    PlotPostParamEvolution!(
        plt_overlay,
        quant_seed[i],
        "",
        groupSizes_laplace;
        dateVec = dateVec,
        interpMethod = interpMethod,
        plot_t0 = keep_t0,
        interval_style = :solid
    )

end

display(plt_overlay)

### cehcking with shoulders of 0.01 - diaglocal much better
# next chek fulllocal with the shoulders
#netx chekm iekf with scaling