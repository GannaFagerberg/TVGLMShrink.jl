# Set up the Gibbs sampler for a state-space model with:
# - non-linear/non-Gaussian (NN) observation model
# - linear Gaussian (LG) state evolution (conditional on Polya-Gamma latents)
# - dynamic shrinkage process prior for the state innovations 
function GibbsTVGLM(Y, priorSettings, modelSettings, algoSettings)

    T = length(Y)
    ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀, μ₀, Σ₀ = priorSettings
    stateSamplingMethod, nParticles, nIter, nBurn, nMaxIter, nPrePGAS, 
        offsetMethod, h_upper, polyaoffset, scaling = algoSettings 
    observation, param, condMean, condCov, α, β, updateσₙ, nMixComp = modelSettings
    p = length(μ₀) # number of states

    ## Approximate the log χ²₁ distribution with a mixture of normals
    mixture = SetUpLogChi2Mixture(nMixComp) # Only 5 and 10 component supported

    ## Initial values          
    S = zeros(Int8, T, p)    # Mixture allocation for logχ²₁ - this is updated first
    μ = fill(m₀, p)
    σ²ₙ = fill(ψ₀, p)
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
    σ²ₙpost = zeros(p, nIter) # Store variance in log-volatility evolution
    μpost = zeros(p, nIter) # Store mean in log-volatility evolution
    
    offset = (offsetMethod == "kowal") ? eps()*ones(T,p) : fill(offsetMethod, T, p)

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
        θparticles = zeros(nParticles, p, T+1) # Initialize PGAS particle container.
        if nPrePGAS > 0
            println("Getting initial values from Laplace")
            algoSettingsInit = (stateSamplingMethod = :ffbs_laplace, 
                nParticles = nParticles, nIter = nPrePGAS, 
                nBurn = round(Int, 0.1*nPrePGAS), nMaxIter = nMaxIter, 
                nPrePGAS = 0, offsetMethod = offsetMethod, h_upper = h_upper, polyaoffset = polyaoffset, scaling = scaling)
            θpost0, Hpost0, ϕpost0, σ²ₙpost0, μpost0 = GibbsTVGLM(Y, priorSettings, 
                modelSettings, algoSettingsInit);
            μ_prop = median(θpost0[1,:,:], dims = 2)[:]
            Σ_prop = PDMat(cov(θpost0[1,:,:], dims = 2))
            initialization = MvNormal(μ_prop, Σ_prop)
            θ = median(θpost0, dims = 3) # Initial reference particle for pgas
            H = median(Hpost0, dims = 3)
            ϕ = median(ϕpost0, dims = 2)
            μ = median(μpost0, dims = 2)
            updateσₙ ? σ²ₙ = median(σ²ₙpost0, dims = 2) : σ²ₙ = fill(ψ₀, p)
        else
            initialization = prior
            θ = PGASsimulate!(θparticles, Y, p, nParticles, param, 
                prior, transition, observation, initialization, systematic)
        end
    end

    nFailure = Ref(0) 

    progressMessage = "Sampling progress: "
    @showprogress desc=progressMessage for i in 1:(nBurn + nIter)
        
        ## Draw state 
        LogVol2Covs!(param.Σᵥ, H) 

        if scaling != I(size(scaling, 1))
            for j in 1:size(H,1)
                Σ_scaled = scaling * Matrix(param.Σᵥ[j]) * scaling'
                Σ_scaled = 0.5 * (Σ_scaled + Σ_scaled')
                param.Σᵥ[j] = PDMat(Σ_scaled)
            end
        end

        
        if stateSamplingMethod == :ffbs_laplace
            FFBS_laplace!(θ, U, Y, A, B, param.Σᵥ, μ₀, Σ₀, observation, param; 
                max_iter = nMaxIter, nFailure = nFailure)
        elseif stateSamplingMethod == :ffbs_slr
            FFBS_SLR!(θ, U, Y, A, B, condMean, condCov, param, param.Σᵥ, μ₀, Σ₀,
                    nMaxIter; α = 1, β = 0, κ = 0, sample_t0 = true, nFailure = nFailure)
        elseif stateSamplingMethod == :pgas
            θ = PGASsimulate!(θparticles, Y, p, nParticles, param, 
                prior, transition, observation, initialization, systematic, θ, 
                nFailure = nFailure) 
        elseif stateSamplingMethod == :montecarlo
            FFBS_montecarlo!(θ, U, Y, A, B, param.Σᵥ, μ₀, Σ₀, observation, param; 
                nMC = 500, nFailure = nFailure)
        else
            error("Only :ffbs_laplace or :pgas are implemented yet.")
        end

        ## Update the log-volatility evolution
        ν = diff(θ, dims = 1) * inv(scaling)
        
        setOffset!(offset, ν, offsetMethod)

        update_dsp!(ν, S, P, H, H̃, ξ, ϕ, μ, σ²ₙ, priorSettings, mixture, Dᵩ,
            offset, α, β, updateσₙ, h_upper, polyaoffset)

        if i > nBurn
            θpost[:, :, i - nBurn] = θ
            Hpost[:, :, i - nBurn] = H 
            ϕpost[:, i - nBurn] = ϕ
            σ²ₙpost[:, i - nBurn] = σ²ₙ
            μpost[:, i - nBurn] = μ
        end
    end
    
    return θpost, Hpost, ϕpost, σ²ₙpost, μpost, nFailure

end