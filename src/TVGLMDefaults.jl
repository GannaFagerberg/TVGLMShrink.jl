# Default settings for TVGLM model, prior and algorithm
#TODO: covSel and FisherInfo prior

defaultDataSettings(y, X; covSel=axes(X, 2), nPerGroup=1) =
    (y=y, X=X, covSel=covSel, nPerGroup=nPerGroup)

function defaultPriorSettings(dataSettings; p=size(dataSettings.X, 2))
    (
        ϕ₀=0.5, κ₀=0.3,
        m₀=-15.0, σ₀=3.0,
        ν₀=3.0, ψ₀=1.0,
        μ₀=zeros(p), Σ₀=Symmetric(10.0 * Matrix(I, p, p)),
    )
end

function default_modelSettings(family::Symbol; kwargs...)
    common = (innovModel=:dsp, α=1 / 2, β=1 / 2, updateσₙ=false, nMixComp=10)

    family === :poisson && return (;
        observation=poisson_observation, link=(LogLink(),),
        condMean=poisson_condMean, condCov=poisson_condCov,
        common..., kwargs...)

    family === :beta && return (;
        observation=beta_observation, link=(LogitLink(), LogLinLink()),
        condMean=beta_condMean, condCov=beta_condCov,
        common..., kwargs...)

    error("Unknown family :$family — supported: :poisson, :beta. " *
          "For a custom family, build modelSettings as a NamedTuple directly.")
end

function defaultAlgoSettings()
    (
        stateSamplingMethod=:ffbs_laplace,
        nParticles=100, nIter=10_000, nBurn=3_000, nMaxIter=10,
        nPrePGAS=500, offsetMethod=eps(), h_upper=Inf, polyaoffset=0.0,
        scaling=:full, FisherInfo=FisherInfoPois, nCalibScale=1_000, verbose=true,
    )
end



function GibbsTVGLM(dataSettings;
    priorSettings=defaultPriorSettings(dataSettings),
    modelSettings=defaultModelSettings(:poisson),
    algoSettings=defaultAlgoSettings(),
    show_progress=true)
    # ... existing body, completely unchanged ...
end


dataSettings = defaultDataSettings(y, X)
θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure =
    GibbsTVGLM(dataSettings; modelSettings=defaultModelSettings(:poisson))



# Make more general with distributions

## ---- Block types: each positional slot of `distfun` is either time-varying or fixed ----

struct TVPBlock{T,TZ<:AbstractVector{<:AbstractMatrix{T}},
    TI<:AbstractUnitRange{<:Integer},
    TL<:Link}
    Z::TZ           # AbstractVector{<:AbstractMatrix{T}} — design matrix at each t
    idx::TI         # AbstractUnitRange{<:Integer}        — slice into `state`
    link::TL
end

struct FixedBlock{T,TV<:AbstractVector{T}}
    value::TV
end

# distparam() dispatches per block type — fully resolved at compile time
distparam(blk::TVPBlock, state, t) = linkinv.(blk.link, blk.Z[t] * @view state[blk.idx])
distparam(blk::FixedBlock, state, t) = blk.value


## ---- TVGLMmodel: holds everything needed to evaluate the model at any (state, t) ----

mutable struct TVGLMmodel{T,S<:AbstractMatrix{T},B<:Tuple,F}
    Σᵥ::Vector{PDMat{T,S}}   # state innov covariances, overwritten each Gibbs iteration
    X::Vector{Matrix{T}}     # covariates feeding the state-innovation scaling, if used
    blocks::B                # Tuple of TVPBlock/FixedBlock, ordered to match distfun's args
    distfun::F               # e.g. BetaMean, Poisson, (μ,σ²)->Normal(μ,sqrt(σ²))
end


## ---- Generic observation: same source for every family, every TVP/fixed combination ----

function observation(param::TVGLMmodel, state, t)
    μs = map(blk -> distparam(blk, state, t), param.blocks)
    product_distribution(param.distfun.(μs...))
end



# Beta, both blocks TVP — or one fixed, doesn't matter, distparam() handles both
function beta_condMean(param::TVGLMmodel, state, t)
    distparam(param.blocks[1], state, t)   # μ
end

function beta_condCov(param::TVGLMmodel, state, t)
    μ = distparam(param.blocks[1], state, t)
    ψ = distparam(param.blocks[2], state, t)
    diagm(μ .* (1 .- μ) ./ (1 .+ ψ))
end

# Poisson, K=1
poisson_condMean(param::TVGLMmodel, state, t) = distparam(param.blocks[1], state, t)
poisson_condCov(param::TVGLMmodel, state, t) = diagm(distparam(param.blocks[1], state, t))


## Poisson, K=1, fully TVP
blocks_pois = (TVPBlock(Z, 1:p, LogLink()),)
model_pois = TVGLMmodel(Σᵥ, X, blocks_pois, Poisson)

## Beta, K=2, both TVP (your original model)
blocks_beta = (TVPBlock(Z_mean, idx_mean, LogitLink()),
    TVPBlock(Z_prec, idx_prec, LogLinLink()))
model_beta = TVGLMmodel(Σᵥ, X, blocks_beta, BetaMean)

## Beta, K=2, TVP mean + fixed precision
blocks_betaφ = (TVPBlock(Z_mean, idx_mean, LogitLink()),
    FixedBlock([φ_init]))
model_betaφ = TVGLMmodel(Σᵥ, X, blocks_betaφ, BetaMean)

## Beta, K=2, fixed mean + TVP precision (the order-flip case)
blocks_betaμ = (FixedBlock([μ_init]),
    TVPBlock(Z_prec, idx_prec, LogLinLink()))
model_betaμ = TVGLMmodel(Σᵥ, X, blocks_betaμ, BetaMean)