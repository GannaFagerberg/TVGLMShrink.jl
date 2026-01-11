# Set up the Gibbs sampler for a state-space model with:
# - non-linear/non-Gaussian (NN) observation model
# - linear Gaussian (LG) state evolution (conditional on Polya-Gamma latents)
# - dynamic shrinkage process prior for the state innovations 
function GibbsTVGLM(Y, priorSettings, modelSettings, algoSettings)

    T = length(Y)
    ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀, μ₀, Σ₀ = priorSettings
    stateSamplingMethod, nParticles, nIter, nBurn, offsetMethod = algoSettings 
    observation, param, condMean, condCov, α, β, updateσₙ, nMixComp = modelSettings
    p = length(μ₀) # number of states

    ## Approximate the log χ²₁ distribution with a mixture of normals
    mixture = SetUpLogChi2Mixture(nMixComp) # Only 5 and 10 component supported

    ## Initial values          
    S = zeros(Int8, T, p)    # Mixture allocation for logχ²₁ - this is updated first
    μ = fill(m₀, p)
    updateσₙ ? σ²ₙ = fill(ψ₀, p) : σ²ₙ = fill(1, p)
    ϕ = fill(ϕ₀, p)
    H = fill(m₀, T, p)
    H̃ = H .- μ'
    ξ = ones(T, p)
    θ = zeros(T+1, p) # Regression coefficients evolution
    Dᵩ = BandedMatrix(-1 => repeat([-ϕ[1]], T-1), 0 => Ones(T)) # Init D matrix for h_t

    ## Storage
    θpost = zeros(T+1, p, nIter) # Store regression coefficients
    Hpost = zeros(T, p, nIter) # Store log-volatility evolution
    ϕpost = zeros(p, nIter) # Store AR coefficients
    σₙpost = zeros(p, nIter) # Store variance in log-volatility evolution
    μpost = zeros(p, nIter) # Store mean in log-volatility evolution
    
    offset = (offsetMethod == "kowal") ? eps()*ones(T,p) : offsetMethod 
    P = zeros(T, nMixComp) # storage for mixture component probabilities

    ## Set up state-space model
    prior = MvNormal(μ₀, Σ₀)
    transition(param, state, t) = (p == 1) ? 
        Normal(state, sqrt(param.Σᵥ[t][1])) : 
        MvNormal(state, param.Σᵥ[t])

    A = collect(I(p))
    B = 0.0
    U = zeros(T,1)

    if stateSamplingMethod == :pgas
        initialization = prior
        θparticles = zeros(nParticles, p, T+1) # Initialize PGAS particle container.
        θ = PGASsimulate!(θparticles, Y, p, nParticles, param, 
                prior, transition, observation, initialization, systematic)
    end

    @showprogress for i in 1:(nBurn + nIter)
        
        ## Draw state 
        LogVol2Covs!(param.Σᵥ, H) 

        if stateSamplingMethod == :ffbs_laplace
            θ = FFBS_laplace(U, Y, A, B, param.Σᵥ, μ₀, Σ₀, observation, param)
        elseif stateSamplingMethod == :ffbs_slr
            θ = FFBS_SLR(U, Y, A, B, μₖ_x, Pₖʸ_x, Cargs, param.Σᵥ, μ₀, Σ₀,
                    maxIter, 1; α = 1, β = 0, κ = 0, sample_t0 = true)
        elseif stateSamplingMethod == :pgas
            θ = PGASsimulate!(θparticles, Y, p, nParticles, param, 
                prior, transition, observation, initialization, systematic, θ) 
        else
            error("Only :ffbs_laplace or :pgas are implemented yet.")
        end

        ## Update the log-volatility evolution
        ν = diff(θ, dims = 1)
        setOffset!(offset, ν, offsetMethod)
        update_dsp!(ν, S, P, H, H̃, ξ, ϕ, μ, σ²ₙ, priorSettings, mixture, Dᵩ)
        
        if i > nBurn
            θpost[:, :, i - nBurn] = θ
            Hpost[:, :, i - nBurn] = H 
            ϕpost[:, i - nBurn] = ϕ
            σₙpost[:, i - nBurn] = σ²ₙ
            μpost[:, i - nBurn] = μ
        end
    end
    
    return θpost, Hpost, ϕpost, σₙpost, μpost

end