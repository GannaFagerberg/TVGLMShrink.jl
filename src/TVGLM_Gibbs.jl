# Set up the Gibbs sampler for a state-space model with:
# - non-linear/non-Gaussian (NN) observation model
# - linear Gaussian (LG) state evolution (conditional on Polya-Gamma latents)
# - dynamic shrinkage process prior for the state innovations 
function GibbsTVGLM(dataSettings, priorSettings, modelSettings, algoSettings;
    show_progress=true)

    # Unpack settings
    y, X, covSel, nPerGroup = dataSettings
    ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀, μ₀, Σ₀ = priorSettings
    stateSamplingMethod, nParticles, nIter, nBurn, nMaxIter, nPrePGAS, offsetMethod,
    h_upper, polyaoffset, scaling, FisherInfo = algoSettings
    observation, staticParam, condMean, condCov, α, β, updateσₙ, nMixComp = modelSettings

    p = length(μ₀) # number of states
    Tobs = length(y)

    # Setting up data as grouped data
    Y, Z, groupSizes = splitEqualGroups(y, X, covSel, nPerGroup)
    T = length(Y)
    groupsize_common = ceil(Int, mean(groupSizes)) # Assuming common group size.

    # Instantiate model parameters (Σᵥ = I for all t), overwritten at each Gibbs iteration
    
    param = staticParam(LogVol2Covs(zeros(length(groupSizes), p)), Z...)

    # Define the scaling matrix
    ScaleMat =
        if scaling == :full
            (θ, μ, t) -> inv(sqrt(Symmetric(FisherInfo(θ, μ, t) / Tobs)))
        elseif scaling == :diagonal
            (θ, μ, t) -> Diagonal(diag(inv(sqrt(Symmetric(FisherInfo(θ, μ, t) / Tobs)))))
        elseif scaling == :none
            (θ, μ, t) -> zeros(p, p)
        else
            error("Invalid scaling option. Choose :full, :diagonal or :none.")
        end

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
    θ = zeros(T + 1, p) # Regression coefficients evolution
    Dᵩ = BandedMatrix(-1 => repeat([-ϕ[1]], T - 1), 0 => Ones(T)) # Init D matrix for h_t

    Svec = zeros(p, p, T) # Storage for scaling matrices in Laplace FFBS
    ## Storage
    θpost = zeros(T + 1, p, nIter) # Store regression coefficients
    Hpost = zeros(T, p, nIter) # Store log-volatility evolution
    ϕpost = zeros(p, nIter) # Store AR coefficients
    σ²ₙpost = zeros(p, nIter) # Store variance in log-volatility evolution
    μpost = zeros(p, nIter) # Store mean in log-volatility evolution

    offset = (offsetMethod == "kowal") ? eps() * ones(T, p) : fill(offsetMethod, T, p)

    P = zeros(T, nMixComp) # storage for mixture component probabilities

    ## Set up transition model for pgas
    prior = MvNormal(μ₀, Σ₀)

    transition =
        if p == 1
            if scaling === :none
                (param, state, t) -> Normal(state, sqrt(param.Σᵥ[t][1]))
            else
                (param, state, t) -> begin
                    Smat = ScaleMat(param, state, t)
                    Normal(state, Smat[1, 1] * sqrt(param.Σᵥ[t][1]))
                end
            end
        else
            if scaling === :none
                (param, state, t) -> MvNormal(state, param.Σᵥ[t])
            else
                (param, state, t) -> begin
                    Smat = ScaleMat(param, state, t)
                    MvNormal(state, Hermitian(Smat * param.Σᵥ[t] * Smat'))
                end
            end
        end

    A = collect(I(p))
    B = 0.0
    U = zeros(T, 1)

    if stateSamplingMethod == :pgas
        θparticles = zeros(nParticles, p, T + 1) # Initialize PGAS particle container.
        if nPrePGAS > 0
            println("Getting initial values from Laplace")
            algoSettingsInit = (stateSamplingMethod=:ffbs_laplace,
                nParticles=nParticles, nIter=nPrePGAS,
                nBurn=round(Int, 0.1 * nPrePGAS), nMaxIter=nMaxIter,
                nPrePGAS=0, offsetMethod=offsetMethod, h_upper=h_upper, polyaoffset=polyaoffset, scaling=scaling, FisherInfo=FisherInfo)
            θpost0, Hpost0, ϕpost0, σ²ₙpost0, μpost0 = GibbsTVGLM(dataSettings, priorSettings,
                modelSettings, algoSettingsInit)
            μ_prop = median(θpost0[1, :, :]; dims=2)[:]
            Σ_prop = PDMat(cov(θpost0[1, :, :]; dims=2))
            initialization = MvNormal(μ_prop, Σ_prop)
            θ = median(θpost0; dims=3) # Initial reference particle for pgas
            H = median(Hpost0; dims=3)
            ϕ = median(ϕpost0; dims=2)
            μ = median(μpost0; dims=2)
            updateσₙ ? σ²ₙ = median(σ²ₙpost0; dims=2) : σ²ₙ = fill(ψ₀, p)
        else
            initialization = prior
            θ = PGASsimulate!(θparticles, Y, p, nParticles, param,
                prior, transition, observation, initialization, systematic)
        end
    end

    nFailure = Ref(0)

    if haskey(ENV, "SLURM_JOB_ID")
        show_progress = false
    end # No progress bar on cluster
    progressMessage = "Sampling progress: "
    @showprogress desc = progressMessage enabled = show_progress for i in 1:(nBurn+nIter)

        ## Draw state 
        LogVol2Covs!(param.Σᵥ, H)

        if stateSamplingMethod == :ffbs_laplace
            if scaling === :none
                FFBS_laplace!(θ, U, Y, A, B, param.Σᵥ, μ₀, Σ₀, observation, param; max_iter=nMaxIter, nFailure=nFailure)
            else
                FFBS_laplace!(θ, U, Y, A, B, param.Σᵥ, μ₀, Σ₀, observation, param,
                    ScaleMat, Svec; max_iter=nMaxIter, nFailure=nFailure)
            end
        elseif stateSamplingMethod == :ffbs_slr
            if scaling === :none
                 FFBS_SLR!(θ, U, Y, A, B, condMean, condCov, param, param.Σᵥ, μ₀, Σ₀,
                    nMaxIter; α=1, β=0, κ=0, sample_t0=true, nFailure=nFailure)
            else
                FFBS_SLR!(θ, U, Y, A, B, condMean, condCov, param, param.Σᵥ, μ₀, Σ₀,
                    nMaxIter, ScaleMat, Svec; α=1, β=0, κ=0, sample_t0=true, nFailure=nFailure)
            end
        elseif stateSamplingMethod == :pgas
            θ = PGASsimulate!(θparticles, Y, p, nParticles, param,
                prior, transition, observation, initialization, systematic, θ;
                nFailure=nFailure)
            for t in 1:T
                Svec[:, :, t] = ScaleMat(param, θ[t, :], t)
            end
        elseif stateSamplingMethod == :montecarlo
            if scaling === :none
                FFBS_montecarlo!(θ, U, Y, A, B, param.Σᵥ, μ₀, Σ₀, observation, param;
                    nMC=500, nFailure=nFailure)
            else
                error("Monte Carlo FFBS not implemented with scaling yet.")
            end
        else
            error("the chosen sampling method is not implemented yet.")
        end

        ## Update the log-volatility evolution
        ν = diff(θ; dims=1)
        if scaling !== :none
            for t in 1:T
                if i == 1 && det(Svec[:, :, t]) == 0
                    Svec[:, :, t] = Svec[:, :, t-1]
                end
                ν[t, :] .= Svec[:, :, t] \ ν[t, :]
            end
        end

        setOffset!(offset, ν, offsetMethod)
        if groupsize_common == 1
            update_dsp!(ν, S, P, H, H̃, ξ, ϕ, μ, σ²ₙ, priorSettings, mixture, Dᵩ,
                offset, α, β, updateσₙ, h_upper, polyaoffset)
        else
            update_dsp!(groupsize_common, ν, S, P, H, H̃, ξ, ϕ, μ, σ²ₙ, priorSettings,
                mixture, Dᵩ, offset, α, β, updateσₙ, h_upper, polyaoffset)
        end

        if i > nBurn
            θpost[:, :, i-nBurn] = θ
            Hpost[:, :, i-nBurn] = H
            ϕpost[:, i-nBurn] = ϕ
            σ²ₙpost[:, i-nBurn] = σ²ₙ
            μpost[:, i-nBurn] = μ
        end
    end

    return θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure
end