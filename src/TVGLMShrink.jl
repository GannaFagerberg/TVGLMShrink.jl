module TVGLMShrink

using Distributions, LinearAlgebra, ProgressMeter, BandedMatrices, Plots, LaTeXStrings
using PDMats, LogExpFunctions
using SMCsamplers, DynamicGlobalLocalShrinkage, Utils
using GLM: Link
import GLM
import GLM: linkfun, linkinv, mueta
export linkfun, linkinv, mueta
export LogLinLink,
       PositiveHardLink,
       ShiftedSoftplusLink
using SpecialFunctions: digamma, trigamma
using Roots: find_zero

include("TEST.jl")
include("TVGLM_Gibbs.jl")

export FFBS_SLR_test!, GibbsTVGLM

include("TVGLMPlots.jl")
export PlotPostParamEvolution, PlotPostParamEvolution!, postPredCheckDistr

include("TVGLMUtils.jl")
export exp_lin, exp_lin_inv, simulateAR, simulateVAR, interpParam2Obs, scalingLabel
export XDiagX, XDiagZ, densityScores

include("TVGLMModels.jl")
export TVGLMmodel, LogLinLink, PositiveHardLink

end
