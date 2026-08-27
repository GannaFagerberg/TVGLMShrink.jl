module TVGLMShrink

using Distributions, LinearAlgebra, ProgressMeter, BandedMatrices, Plots, LaTeXStrings
using PDMats, LogExpFunctions
using SMCsamplers, DynamicGlobalLocalShrinkage, Utils
using GLM: Link, CauchitLink, ProbitLink,CloglogLink
import GLM
import GLM: linkfun, linkinv, mueta
export linkfun, linkinv, mueta
export LogLinLink,
       PositiveHardLink,
       ShiftedSoftplusLink,
       WoodardLink,
       UnitHardLink
using SpecialFunctions: digamma, trigamma
using Roots: find_zero

include("TVGLM_TransformedFFBS_KF.jl")
include("TVGLM_Gibbs.jl")
include("BetaModelSufficient.jl")

export FFBS_SLR_transformed!, FFBS_SLR_transformed_scaling!!, GibbsTVGLM

include("TVGLMPlots.jl")
export PlotPostParamEvolution, PlotPostParamEvolution!, postPredCheckDistr

include("TVGLMUtils.jl")
export exp_lin, exp_lin_inv, simulateAR, simulateVAR, interpParam2Obs, scalingLabel
export XDiagX, XDiagZ, densityScores

include("TVGLMModels.jl")
export TVGLMmodel, LogLinLink, PositiveHardLink

end
