module TVGLMShrink

using Distributions, LinearAlgebra, ProgressMeter, BandedMatrices, Plots, LaTeXStrings
using PDMats, LogExpFunctions
using SMCsamplers, DynamicGlobalLocalShrinkage, Utils
using GLM: Link, CauchitLink, ProbitLink,CloglogLink
import GLM
import Optim
import ForwardDiff
import GLM: linkfun, linkinv, mueta
export linkfun, linkinv, mueta
export LogLinLink,
       PositiveHardLink,
       ShiftedSoftplusLink,
       WoodardLink,
       UnitHardLink
using SpecialFunctions: digamma, trigamma
using Roots: find_zero


# ============================================================
# Observation transformation interface
#
# MUST be defined before BetaModelSufficient.jl,
# GammaModelSufficient.jl, NBModelSufficient.jl, etc.
# ============================================================

abstract type AbstractObsTransform end
struct IdentityTransform <: AbstractObsTransform end


include("TVGLM_TransformedFFBS_KF.jl")
include("TVGLM_Gibbs.jl")
include("BetaModelSufficient.jl")
#include("BetaModelSufficient_single.jl")
include("GammaModelSufficient.jl")
include("NBModelSufficient.jl")
include("TVGLM_IEKF_ffbs_kf.jl")

export FFBS_SLR_transformed!, FFBS_SLR_transformed_scaling!, GibbsTVGLM
export BetaSuffStats, BetaSuffStatsGrouped, BetaSuffStatsAveraged, prepare_observation_transform
export BetaSingleSuffStatGrouped
export prepare_observation_transform
export GammaSuffStatsAveraged,  GammaSuffStatsGrouped
export fisher_gamma_blocks, FisherInfoGamma
export fisher_nb_blocks, FisherInfoNB, NBFactorialStatsAveraged, NBFactorialStatsGrouped
export FFBS_IEKF_transformed!, FFBS_IEKF_transformed_scaled!,
       kalmanfilter_update_transformed_IEKF, 
       BetaSuffStatsCondMoments, BetaSuffStatsJacobian
export FFBS_laplace_constrained!

export BetaLogYCondMoments,
       BetaLogYJacobian,
       BetaLog1mYCondMoments,
       BetaLog1mYJacobian

include("TVGLMPlots.jl")
export PlotPostParamEvolution, PlotPostParamEvolution!, postPredCheckDistr

include("TVGLMUtils.jl")
export exp_lin, exp_lin_inv, simulateAR, simulateVAR, interpParam2Obs, scalingLabel
export XDiagX, XDiagZ, densityScores,update_homoscedastic_uni!

include("TVGLMModels.jl")
export TVGLMmodel, LogLinLink, PositiveHardLink
export AbstractObsTransform,IdentityTransform, 
       make_identity_cond_moments, prepare_observation_transform

end
