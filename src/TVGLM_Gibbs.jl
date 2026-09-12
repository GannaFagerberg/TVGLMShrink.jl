
# Set up the Gibbs sampler for a state-space model with:
# - non-linear/non-Gaussian (NN) observation model
# - linear Gaussian (LG) state evolution (conditional on Polya-Gamma latents)
# - dynamic shrinkage process prior for the state innovations 
function GibbsTVGLM(dataSettings, priorSettings, modelSettings, algoSettings;
    progessbar=(status=true, message="Sampling progress: "))

    # Unpack settings
    y, X, covSel, nPerGroup = dataSettings
    ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀, μ₀, Σ₀, n₀ = priorSettings
    stateSamplingMethod, nParticles, nIter, nBurn, nMaxIter, nPrePGAS, offsetMethod,
    h_upper, polyaoffset, scaling, FisherInfo, nCalibScale, fixed_scaling, verbose = algoSettings
    #observation, link, condMean, condCov, innovModel, α, β, updateσₙ, nMixComp = modelSettings
     (;observation,link,condMean,condCov,slrObs,innovModel,α,β,updateσₙ,nMixComp) = modelSettings

    if verbose
        println("$stateSamplingMethod using scaling = $scaling with $nPerGroup obs per group.")
    end

    nState = length(μ₀) # number of states

    if isa(m₀, Number)
        ϕ₀ = ϕ₀*ones(nState)
        κ₀ = κ₀*ones(nState)
        m₀ = m₀*ones(nState)
        σ₀ = σ₀*ones(nState)
        ν₀ = ν₀*ones(nState)
        ψ₀ = ψ₀*ones(nState)
        priorSettings = (
            ϕ₀=ϕ₀, κ₀=κ₀,             
            m₀=m₀, σ₀=σ₀,           
            ν₀=ν₀, ψ₀=ψ₀,               
            μ₀=μ₀, Σ₀=Σ₀, n₀=n₀,
        );
    end

    
    # Setting up data as grouped data
    Tobs = length(y)
    Y, Z, Xsel, groupSizes = splitEqualGroups(y, X, covSel, nPerGroup)
    T = length(Y)
    groupsize_common = ceil(Int, mean(groupSizes)) # Assuming common group size.

    # Zidx[k] is the index in the state vector for the covariates in parameter k
    Zidx = Vector{UnitRange{Int}}(undef, length(Z))
    start = 1
    for k in 1:length(Z)
        Zidx[k] = start:(start+size(Z[k][1], 2)-1)
        start += size(Z[k][1], 2)
    end

    # Instantiate model parameters (Σᵥ = I for all t), overwritten at each Gibbs iteration
    param = TVGLMmodel(LogVol2Covs(zeros(length(groupSizes), nState)), Z, Xsel, link, Zidx)

    # Set up prior cov for t=0 state, with option to use Fisher info based prior
    if Σ₀ == :fisherinfo
        Σ₀ = Hermitian((size(X, 1) / n₀) * inv(FisherInfo(param, μ₀, 1)))
        if verbose
            println("Prior at t=0 based on Fisher info with n₀ = $n₀")
            priorStd = sqrt.(diag(Σ₀))
            println("Prior 95% interval for the state at time t=0:")
            for j in 1:length(μ₀)
                println("State $j: [", round(μ₀[j] - 1.96 * priorStd[j], digits=3), ", ",
                    round(μ₀[j] + 1.96 * priorStd[j], digits=3), "]")
            end
        end
    end

    collectScaling = false
    if collectScaling
        Svec_collect = zeros(nState, nState, T, nIter) # Storage for scaling matrices
    else
        Svec_collect = zeros(0, 0, 0, 0) # Empty array if not collecting scaling matrices
    end
    Svec = zeros(nState, nState, T) # Storage for scaling matrices
    if scaling !== :none
        algoSettingsCalibrate = (; algoSettings..., scaling=:none,
            stateSamplingMethod=:ffbs_laplace, nIter=nCalibScale,
            nBurn=round(Int, 0.1 * nCalibScale), verbose=false)

        θpost0, _, _ = GibbsTVGLM(dataSettings, priorSettings, modelSettings, algoSettingsCalibrate, progessbar=(status=progessbar.status, message="Calibrating scaling matrix: "))
        for t in 1:T
            θmedian_t = median(θpost0[t, :, :]; dims=2)
            if scaling == :full
                Svec[:, :, t] = sqrt(inv(Symmetric(groupSizes[t] * 
                    FisherInfo(param, θmedian_t, t) / Tobs)))
            elseif scaling == :diag
                Svec[:, :, t] = Diagonal(sqrt(inv(Symmetric(groupSizes[t] * 
                    FisherInfo(param, θmedian_t, t) / Tobs)))) 
            elseif scaling == :fulllocal 
                Svec[:, :, t] = sqrt(pinv(Symmetric(FisherInfo(param, θmedian_t, t))))
            elseif scaling == :diaglocal
                Svec[:, :, t] = Diagonal(sqrt(pinv(Symmetric(FisherInfo(param, 
                    θmedian_t, t)))))
            else

            end
        end

    end

    # Define the scaling matrix
    ScaleMat =
        if fixed_scaling
            (par, state, t) -> Svec[:, :, t]
        elseif scaling == :full
            (par, state, t) -> sqrt(inv(Symmetric(groupSizes[t] * FisherInfo(par, state, t) / Tobs)))
        elseif scaling == :diag
            (par, state, t) -> Diagonal(sqrt(inv(Symmetric(groupSizes[t] * FisherInfo(par, state, t) / Tobs))))
        elseif scaling == :fulllocal
            (par, state, t) -> sqrt(pinv(Symmetric(FisherInfo(par, state, t))))
        elseif scaling == :diaglocal
            (par, state, t) -> Diagonal(sqrt(pinv(Symmetric(FisherInfo(par, state, t)))))
        elseif scaling == :none
            (par, state, t) -> I(nState)
        else
            error("Invalid scaling option. Choose :full, :diag,     
                :fulllocal, diaglocal or :none.")
        end

    ## Approximate the log χ²₁ distribution with a mixture of normals
    mixture = SetUpLogChi2Mixture(nMixComp) # Only 5 and 10 component supported

    ## Initial values          
    S = zeros(Int8, T, nState)    # Mixture allocation for logχ²₁ - this is updated first
    μ = copy(m₀)
    σ²ₙ = copy(ψ₀)
    ϕ = copy(ϕ₀)
    H = repeat(μ', T)
    H̃ = H .- μ'
    ξ = ones(T, nState)
    θ = zeros(T + 1, nState) # Regression coefficients evolution
    Dᵩ = BandedMatrix(-1 => repeat([-ϕ[1]], T - 1), 0 => Ones(T)) # Init D matrix for h_t

    ## Storage
    θpost = zeros(T + 1, nState, nIter) # Store regression coefficients
    Hpost = zeros(T, nState, nIter) # Store log-volatility evolution
    ϕpost = zeros(nState, nIter) # Store AR coefficients
    σ²ₙpost = zeros(nState, nIter) # Store variance in log-volatility evolution
    μpost = zeros(nState, nIter) # Store mean in log-volatility evolution

    offset = (offsetMethod == "kowal") ? eps() * ones(T, nState) : fill(offsetMethod, T, nState)
    P = zeros(T, nMixComp) # storage for mixture component probabilities
    prior = MvNormal(μ₀, Σ₀)

    ## Set up transition model for pgas
    transition =
        if nState == 1
            if scaling === :none
                (param, state, t) -> Normal(state, sqrt(param.Σᵥ[t][1] + eps()))
            else
                (param, state, t) -> begin
                    Smat = ScaleMat(param, state, t)
                    Normal(state, Smat[1, 1] * sqrt(param.Σᵥ[t][1] + eps()))
                end
            end
        else
            if scaling === :none
                (param, state, t) -> MvNormal(state, param.Σᵥ[t] + eps() * I)
            else
                (param, state, t) -> begin
                    Smat = ScaleMat(param, state, t)
                    MvNormal(state, Hermitian(Smat * param.Σᵥ[t] * Smat' + eps() * I))
                end
            end
        end

    A = collect(I(nState))
    B = 0.0
    U = zeros(T, 1)

    if stateSamplingMethod == :pgas
        θparticles = zeros(nParticles, nState, T + 1) # Initialize PGAS particle container.
        if nPrePGAS > 0
            algoSettingsInit = (; algoSettings..., stateSamplingMethod=:ffbs_laplace,
                nIter=nPrePGAS, nBurn=round(Int, 0.1 * nPrePGAS), nPrePGAS=0, scaling=:none,
                verbose=false)
            θpost0, Hpost0, ϕpost0, σ²ₙpost0, μpost0 = GibbsTVGLM(dataSettings,
                priorSettings, modelSettings, algoSettingsInit, progessbar=(status=progessbar.status, message="Getting initial values from Laplace: "))
            μ_prop = median(θpost0[1, :, :]; dims=2)[:]
            Σ_prop = PDMat(cov(θpost0[1, :, :]; dims=2))
            initialization = MvNormal(μ_prop, Σ_prop)
            θ = median(θpost0; dims=3) # Initial reference particle for pgas
            H = median(Hpost0; dims=3)
            ϕ = median(ϕpost0; dims=2)
            μ = median(μpost0; dims=2)
            updateσₙ ? σ²ₙ = median(σ²ₙpost0; dims=2) : σ²ₙ = ψ₀#fill(ψ₀, nState)
        else
            initialization = prior
            θ = PGASsimulate!(θparticles, Y, nState, nParticles, param,
                prior, transition, observation, initialization, systematic)
        end
    end

    nFailure = Ref(0)
    nLaplaceFailure =  Ref(0)

    ### For transformed-observation filters
    if stateSamplingMethod == :ffbs_slr ||
    stateSamplingMethod == :ffbs_iekf

        Y_sufficient = slrObs.Y
        nObs = slrObs.nObs

        # IPLF-specific
        if stateSamplingMethod == :ffbs_slr

            sufficient_condMoments = slrObs.condMoments

            ws = TVGLMShrink.IPLFWorkspace(
                Float64,
                nState,
                nObs
            )
        end
    end

    ### Begin the LOOP
    if haskey(ENV, "SLURM_JOB_ID")
        progessbar = (; progessbar..., status=false)
    end
    @showprogress desc = progessbar.message enabled = progessbar.status for i in 1:(nBurn+nIter)

        ## Draw state 
        LogVol2Covs!(param.Σᵥ, H) # Update the covariance matrices for the state 

        if stateSamplingMethod == :ffbs_laplace
            if scaling === :none
                #FFBS_laplace!(θ, U, Y, A, B, param.Σᵥ, μ₀, Σ₀, observation, param; max_iter=nMaxIter, nFailure=nFailure)
                FFBS_laplace_constrained!(θ, U, Y, A, B, param.Σᵥ, μ₀, Σ₀, observation, param; max_iter=nMaxIter, nFailure=nFailure, nLaplaceFailure=nLaplaceFailure)
            else
                FFBS_laplace!(θ, U, Y, A, B, param.Σᵥ, μ₀, Σ₀, observation, param, ScaleMat, Svec; max_iter=nMaxIter, nFailure=nFailure)
            end
        elseif stateSamplingMethod == :ffbs_slr
            if scaling === :none
                #FFBS_SLR_transformed!(θ, U, Y_sufficient, A, B, sufficient_condMoments, param, param.Σᵥ, μ₀, Σ₀,nMaxIter, ws; α=1, β=0, κ=0, sample_t0=true, nFailure=nFailure)
                FFBS_SLR_transformed!(θ, U, Y_sufficient, A, B, sufficient_condMoments, param, param.Σᵥ, μ₀, Σ₀,nMaxIter, ws; α=0.001, β=2, κ=0, sample_t0=true, nFailure=nFailure)
            else
                FFBS_SLR_transformed_scaling!(θ, U, Y_sufficient, A, B, sufficient_condMoments, param, param.Σᵥ, μ₀, Σ₀, nMaxIter, ScaleMat, Svec, ws; α=1, β=0, κ=0, sample_t0=true, nFailure=nFailure)
            end
        elseif stateSamplingMethod == :pgas
            θ = PGASsimulate!(θparticles, Y, nState, nParticles, param,
                prior, transition, observation, initialization, systematic, θ;
                nFailure=nFailure)
            for t in 1:T
                Svec[:, :, t] = ScaleMat(param, θ[t, :], t)
            end

        elseif stateSamplingMethod == :ffbs_iekf

        sufficient_condMoments_IEKF = modelSettings.sufficient_condMoments_IEKF
        sufficient_condJacobian = modelSettings.sufficient_condJacobian

        if scaling === :none
            FFBS_IEKF_transformed!(θ,U,Y_sufficient,A,B,sufficient_condMoments_IEKF,sufficient_condJacobian,param,param.Σᵥ,μ₀,Σ₀,nMaxIter;sample_t0=true,nFailure=nFailure)
        else
            error("Scaling is not yet implemented for stateSamplingMethod = :ffbs_iekf")
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
                    Svec[:, :, t] = t > 1 ? Svec[:, :, t-1] : ScaleMat(param, μ₀, t)
                end
                ν[t, :] .= Svec[:, :, t] \ ν[t, :]
            end
        end

        if innovModel == :dsp
            setOffset!(offset, ν, offsetMethod)
            if groupsize_common == 1
                update_dsp!(ν, S, P, H, H̃, ξ, ϕ, μ, σ²ₙ, priorSettings, mixture, Dᵩ,offset, α, β, updateσₙ, h_upper, polyaoffset)
            else
                update_dsp!(groupsize_common, ν, S, P, H, H̃, ξ, ϕ, μ, σ²ₙ, priorSettings, mixture, Dᵩ, offset, α, β, updateσₙ, h_upper, polyaoffset)
            end
        elseif innovModel == :homogaussuniv # homoscedastic case
            update_homoscedastic_uni!(ν, H, 4, exp(m₀[1]/2))
        else
            error("the chosen innovation model is not implemented yet.")
        end

        if i > nBurn
            θpost[:, :, i-nBurn] = θ
            #Hpost[:, :, i-nBurn] = H
            #ϕpost[:, i-nBurn] = ϕ
            #σ²ₙpost[:, i-nBurn] = σ²ₙ
            #μpost[:, i-nBurn] = μ
            if collectScaling
                #Svec_collect[:, :, :, i-nBurn] = Svec
            end
        end
    end

    #return θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure, Svec_collect
    return θpost, groupSizes, nFailure, nLaplaceFailure
end


