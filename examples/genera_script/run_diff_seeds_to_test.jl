using Random
using Plots

# ============================================================
# Seeds
# ============================================================

seeds = [1, 2, 3, 4, 5]
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

    # Laplace
    #θpost_seed[i],_,_,_ = GibbsTVGLM(dataSettings_laplace,priorSettings_laplace,modelSettings,algoSettings_laplace) 
    
    #IPLF
    θpost_seed[i],_,_,_ = GibbsTVGLM(dataSettings_iplf,priorSettings,modelSettings_iplf,algoSettings_iplf)

    #IEKF
    #θpost_seed[i],_,_,_ =GibbsTVGLM(dataSettings_iekf,priorSettings,modelSettings_iekf,algoSettings_iekf)

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
#plt_overlay          = plot_param_path_betareg(β,γ)
for i in eachindex(seeds)

    PlotPostParamEvolution!(
        plt_overlay,
        quant_seed[i],
        "",
        groupSizes;
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

#savefig(plt_overlay,joinpath(save_dir, "iplf_none.pdf"))
#savefig(plt_overlay,joinpath(save_dir, "iplf_diag.pdf"))
#savefig(plt_overlay,joinpath(save_dir, "iplf_full.pdf"))
#savefig(plt_overlay,joinpath(save_dir, "iplf_fulllocal.pdf"))

savefig(plt_overlay,joinpath(save_dir, "iplf_simInvGass_none.pdf"))


#savefig(plt_overlay,joinpath(save_dir, "iplf_betasim_none_bounded.pdf"))
#quant_sbetasim_none_bounded = copy(quant_seed)
#savefig(plt_overlay,joinpath(save_dir, "iplf_betasim_none.pdf"))
#quant_sbetasim_none = copy(quant_seed)

#quant_seed = quant_sbetasim_none_bounded


#@save joinpath(save_dir, "quant_sbetasim_none.jld2") quant_sbetasim_none
#@save joinpath(save_dir, "quant_sbetasim_none_bounded.jld2") quant_sbetasim_none_bounded

#using JLD2
#@load joinpath(save_dir, "quant_sbetasim_none.jld2") quant_sbetasim_none

# ------------------------------------------------------------
# Overlay 5 seeds: old vs new full Laplace
# ------------------------------------------------------------

# ------------------------------------------------------------
# One plot per seed:
# old full Laplace vs new full Laplace
# ------------------------------------------------------------

plots_seed = Vector{Plots.Plot}(undef, length(seeds))

for i in eachindex(seeds)

    plt = plot(
        layout = (3, 1),
        size = (900, 650),
        legend = :topright,
        #xlim= (-0.025, 1)
    )

    # OLD method
    PlotPostParamEvolution!(
        plt,
        quant_seed_full_laplace_old[i],
        "Old",
        groupSizes;
        dateVec = dateVec,
        interpMethod = interpMethod,
        plot_t0 = keep_t0,
        interval_style = :solid,
        c = colors[1],
        lw = 2
    )

    # NEW method
    PlotPostParamEvolution!(
        plt,
        quant_seed_full_laplace[i],
        "New",
        groupSizes;
        dateVec = dateVec,
        interpMethod = interpMethod,
        plot_t0 = keep_t0,
        interval_style = :solid,
        c = colors[2],
        lw = 2
    )

    plot!(
        plt,
        plot_title = "Seed $(seeds[i])"
    )

    plots_seed[i] = plt

    plot!(plt[1], ylims = (-3, -0.5))
plot!(plt[2], ylims = (-0.035, 0.1))
plot!(plt[3], ylims = (0, 25))

end



display(plots_seed[1])
display(plots_seed[2])
display(plots_seed[3])
display(plots_seed[4])
display(plots_seed[5])


### IEKF much stabler with bounded region and full covariance. Naturally linearises locally
### IPLF with full covraince - diverges. Maybe try centered mean?

### Now I use loglin in the observation, and log in Fisher