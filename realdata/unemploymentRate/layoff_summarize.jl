# US layoff proportions

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
cd(joinpath(@__DIR__, "../.."))
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using PDMats, LogExpFunctions
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim, get_slurm_id
using Utils: mvcolors as colors
using CSV, DataFrames, Dates, JLD2
#using BetaRegression

slurm_id = get_slurm_id() # get slurm ID, if on cluster

Random.seed!(slurm_id);

includet(joinpath(@__DIR__, "../..") * "/examples/betareg/BetaModel.jl") # BetaReg stuff

gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
    titlefontsize=18, markerstrokecolor=:auto)
mainFolder = @__DIR__
dataFolder = joinpath(@__DIR__, "data")
figFolder = joinpath(@__DIR__, "figs/")
resFolder = joinpath(@__DIR__, "results/")

@load resFolder * "layoff_results.jld2" results
@load resFolder * "layoff_results.jld2" pbetadens

