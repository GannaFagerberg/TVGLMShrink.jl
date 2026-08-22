
using TVGLMShrink
using Distributions, LaTeXStrings, Plots, LinearAlgebra, Measures, Random
using PDMats, LogExpFunctions
using SMCsamplers, DynamicGlobalLocalShrinkage
using Utils: quantile_multidim
using Utils: mvcolors as colors 
using JLD2

include("DistModel.jl")
# Scatter plot the state against the observations so see how linear it is
nSim = 1000
invlink(x) = exp(x)
p = 1
x = [1]
β = [1.0]
τ = 1.0
μ = β
Σ = τ^2*I(p)
stateDraws = rand(MvNormal(μ,Σ), nSim)
obsDraws = zeros(Int, nSim)
for i in 1:nSim
    obsDraws[i] = rand(Poisson(invlink(x ⋅ stateDraws[:,i])))
end

plt = []
for j in 1:p
    correlation = round(cor(stateDraws[j,:], obsDraws), digits = 3)
    push!(plt, scatter(stateDraws[j,:], obsDraws, color = colors[j], label = "",
        title = L"\mathrm{Corr}(\beta_{%$(j-1)}, y) = %$(correlation)", xlabel = "State $j",
        ylabel = "Observations", markersize = 5, markerstrokecolor = :black, markerstrokewidth = 0))
end                 
plot(plt..., layout = (2,2), size = (1000, 800))

# Exponentially distributed observations
nSim = 1000
invlink(x) = exp(x)
p = 1
x = [1]
β = [1.0]
τ = 1.0
μ = β
p=1
Σ = τ^2*I(p)
stateDraws = rand(MvNormal(μ,Σ), nSim)
obsDrawsExp = zeros(nSim)
for i in 1:nSim
    obsDrawsExp[i] = rand(Exponential(invlink(x ⋅ stateDraws[:,i])))
end 
pltExp = []
for j in 1:p
    correlation = round(cor(stateDraws[j,:], obsDrawsExp), digits = 3)
    push!(pltExp, scatter(stateDraws[j,:], obsDrawsExp, color = colors[j], label = "",
        title = L"\mathrm{Corr}(\beta_{%$(j-1)}, y) = %$(correlation)", xlabel = "State $j",
        ylabel = "Observations", markersize = 5, markerstrokecolor = :black, markerstrokewidth = 0))
end
plot(pltExp..., layout = (2,2), size = (1000, 800))

# Beta distributed observations
nSim = 10000
invlinkmean(x) = LogExpFunctions.logistic(x) # Inverse link function for Beta regression
invlinkprec(x) = exp(x) # Inverse link function for Beta regressionp = 1
x = [1]
β = [1.0, 1.0, -1.0]
τ = [0.1, 1, 2]
BetaPlt = []
p=1
for i in 1:3
    μ = [β[i]]
    Σ = (τ[i]^2)*I(p)
    stateDraws = rand(MvNormal(μ,Σ), nSim)
    nStates = size(stateDraws,1)
    obsDrawsBeta = zeros(nSim)
    for i in 1:nSim
        mean = 0.5#invlinkmean(x ⋅ stateDraws[:,i])
        precision = invlinkprec(x ⋅ stateDraws[:,i])
        obsDrawsBeta[i] = rand(Beta(mean*precision, (1-mean)*precision))
    end
    push!(BetaPlt, (
                obsDrawsBeta=obsDrawsBeta,
                stateDraws=stateDraws)
                    )


end 
plot(BetaPlt[3].obsDrawsBeta)
pltBeta = []
method = 3
for j in 1:nStates
    correlation = round(cor(BetaPlt[method].stateDraws[j,:], BetaPlt[method].obsDrawsBeta), digits = 3)
    push!(pltBeta, scatter(BetaPlt[method].stateDraws[j,:], BetaPlt[method].obsDrawsBeta, color = colors[j], label = "",
        title = L"\mathrm{Corr}(\beta_{%$(j-1)}, y) = %$(correlation)", xlabel = "State $j",
        ylabel = "Observations", markersize = 5, markerstrokecolor = :black, markerstrokewidth = 0))
end
betap = plot(pltBeta..., layout = (1,3), size = (1000, 800))        
savefig(betap, "/Users/niuyijie/Dropbox/TV_GLM_DSP/ClusterOUT/PoisSim/IPLF_Fail(corr).pdf")



######## IPLF fails for Poisson regression and Exponential regressionwith log link
Random.seed!(1234)
methodlabel = "IPLF"
nPerGroup=5
FisherInfo_none = []
algoSettings = (
    stateSamplingMethod=:ffbs_slr, # Algorithm to sample the state
    nParticles=100,           # Number of particles if using PGAS
    nIter=10000,              # Number of iterations in the Gibbs sampler
    nBurn=3000,               # Number of burn-in iterations
    nMaxIter=10,              # Maximum number of iterations for Laplace/IPLF
    nPrePGAS=500,             # Number of pre-PGAS iterations to initialize the particles
    offsetMethod=eps(),       # Offset for log-volatility
    h_upper=Inf,              # Upper bound for log-volatility
    polyaoffset=0.0,          # Offset for Polya-Gamma variables in the update of h_t
    scaling=:none,            # Scaling of state innov, can be :full, :diagonal or :none
    FisherInfo=FisherInfo_none, # Fisher info
    nCalibScale=1000,         # No. iter to calibrate the scaling matrix :fullfixed case
    fixed_scaling = false,     # Should the scaling matrix be fixed across Gibbs iter?
    verbose=true,             # Whether to print verbose output during sampling.
);
gr(legend=:topleft, grid=false, color=colors[2], lw=2, legendfontsize=12,
    xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14,
    titlefontsize=16, markerstrokecolor=:auto)

keep_t0 = false # Whether to keep the state at time t=0 in the output of the Gibbs sampler
results = []
interpMethod = :linear

## Poisson Regression
covSel = [[1,2]]
p = length(covSel[1])
q = 0
link = (LogLinLink(),)

## Set up the prior, model and algorithm settings
priorSettings = (
    ϕ₀=0.5, κ₀=0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀=3.0, ψ₀=1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=zeros(p+q), Σ₀=I(p+q),n₀ = 1,  # Prior for βₜ at time t=0
);

# Poisson regression
dist = "Poisson"

dataSettings = (y=PoisData[:,1], X=PoisData[:,2:3], covSel=covSel, nPerGroup=5);
observation(param, state, t) = product_distribution(Poisson.(linkinv.(param.link[1], param.Z[1][t] * state)))
condMean(param, state, t) = linkinv.(param.link[1], param.Z[1][t] * state)
condCov(param, state, t) = diagm(linkinv.(param.link[1], param.Z[1][t] * state))

modelSettings = (
    observation=observation,
    link=link, # link functions for mean and precision
    condMean=condMean,
    condCov=condCov,
    innovModel=:dsp,   # choices: :dsp, :homogaussuniv
    α=1 / 2,
    β=1 / 2,
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);

θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);     

plt = plot_param_path_poisreg(β)
PlotPostParamEvolution!(plt, quant_paramtime, methodlabel, groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:shaded, lw=1, c=colors[1])


push!(results, (
                name="$methodlabel $(dist), nPerGroup $(nPerGroup)",
                quant_paramtime=quant_paramtime,
                priorSettings=priorSettings,
                modelSettings=modelSettings,
                algoSettings=algoSettings,
                groupSizes=groupSizes,
                nFailure=nFailure[])
                    )


algoSettings = (; algoSettings..., stateSamplingMethod=:ffbs_laplace);
methodlabel = "Laplace"
θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);     

plt = plot_param_path_poisreg(β)
titles = vcat([L"\beta_{%$(j-1)}" for j in 1:p])
PlotPostParamEvolution!(plt, quant_paramtime, methodlabel, groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:shaded, lw=1, c=colors[1])

push!(results, (
                name="$methodlabel $(dist), nPerGroup $(nPerGroup)",
                quant_paramtime=quant_paramtime,
                priorSettings=priorSettings,
                modelSettings=modelSettings,
                algoSettings=algoSettings,
                groupSizes=groupSizes,
                nFailure=nFailure[])
                    )

# Exponential regression
dist = "Exponential"
dataSettings = (y=ExpData[:,1], X=ExpData[:,2:3], covSel=covSel, nPerGroup=5);
link = (LogLinLink(),)
observation(param, state, t) = product_distribution(Exponential.(linkinv.(param.link[1], param.Z[1][t] * state)))
condMean(param, state, t) = linkinv.(param.link[1], param.Z[1][t] * state)
condCov(param, state, t) = begin
    μ = linkinv.(param.link[1], param.Z[1][t] * state)
    diagm(μ .^ 2)
end
modelSettings = (
    observation=observation,
    link=link, # link functions for mean and precision
    condMean=condMean,
    condCov=condCov,
    innovModel=:dsp,   # choices: :dsp, :homogaussuniv
    α=1 / 2,
    β=1 / 2,
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);
algoSettings = (; algoSettings..., stateSamplingMethod=:ffbs_slr);
methodlabel = "IPLF"
θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);     

plt = plot_param_path_poisreg(β)
titles = vcat([L"\beta_{%$(j-1)}" for j in 1:p])
PlotPostParamEvolution!(plt, quant_paramtime, methodlabel, groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:shaded, lw=1, c=colors[1])
push!(results, (
                name="$methodlabel $(dist), nPerGroup $(nPerGroup)",
                quant_paramtime=quant_paramtime,
                priorSettings=priorSettings,
                modelSettings=modelSettings,
                algoSettings=algoSettings,
                groupSizes=groupSizes,
                nFailure=nFailure[])
                    )


algoSettings = (; algoSettings..., stateSamplingMethod=:ffbs_laplace);
methodlabel = "Laplace"
θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);     

plt = plot_param_path_poisreg(β)
titles = vcat([L"\beta_{%$(j-1)}" for j in 1:p])
PlotPostParamEvolution!(plt, quant_paramtime, methodlabel, groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:shaded, lw=1, c=colors[1])

push!(results, (
                name="$methodlabel $(dist), nPerGroup $(nPerGroup)",
                quant_paramtime=quant_paramtime,
                priorSettings=priorSettings,
                modelSettings=modelSettings,
                algoSettings=algoSettings,
                groupSizes=groupSizes,
                nFailure=nFailure[])
                    )


######## IPLF fails for Negative Binomial regression
dist = "Negative Binomial"

covSel = [[1,2],[1]]
p = length(covSel[1])
q = length(covSel[2])
# Link function for the mean and precision
link = (LogLinLink(),LogLinLink())

priorSettings = (
    ϕ₀=0.5, κ₀=0.3,             # Prior for ϕ ~ N(ϕ₀, κ₀²)
    m₀=-15.0, σ₀=3.0,           # Prior for μ ~ N(m₀, σ₀²)
    ν₀=3.0, ψ₀=1,               # Prior for σ²ₙ ~ scaled inverse χ²(ν₀, ψ₀)
    μ₀=zeros(p+q), Σ₀=I(p+q),n₀ = 1, # Prior for βₜ at time t=0
);


dataSettings = (y=NegbinomData[:,1], X=NegbinomData[:,2:3], covSel=covSel, nPerGroup=5);
observation(param, state, t) =
    @views product_distribution(
        NegBinomMean.(
            GLM.linkinv.(param.link[1], param.Z[1][t] * state[param.Zidx[1]]),
            GLM.linkinv.(param.link[1], param.Z[2][t] * state[param.Zidx[2]])
        )
    )
@views condMean(param, state, t) = GLM.linkinv.(param.link[1],
    param.Z[1][t] * state[param.Zidx[1]])
function condCov(param, state, t)
    @views begin
        μ = GLM.linkinv.(param.link[1], param.Z[1][t] * state[param.Zidx[1]])
        ψ = GLM.linkinv.(param.link[1], param.Z[2][t] * state[param.Zidx[2]])
    end
    return diagm(μ + ((μ .^2) ./ ψ))
end

modelSettings = (
    observation=observation,
    link=link, # link functions for mean and precision
    condMean=condMean,
    condCov=condCov,
    innovModel=:dsp,   # choices: :dsp, :homogaussuniv
    α=1 / 2,
    β=1 / 2,
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);
algoSettings = (; algoSettings..., stateSamplingMethod=:ffbs_slr);
methodlabel = "IPLF"
θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);     

plt = plot_param_path_betareg(β1,γ1)
titles = vcat([L"\beta_{%$(j-1)}" for j in 1:p])
PlotPostParamEvolution!(plt, quant_paramtime, methodlabel, groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:shaded, lw=1, c=colors[1])

push!(results, (
                name="$methodlabel $(dist), nPerGroup $(nPerGroup)",
                quant_paramtime=quant_paramtime,
                priorSettings=priorSettings,
                modelSettings=modelSettings,
                algoSettings=algoSettings,
                groupSizes=groupSizes,
                nFailure=nFailure[])
                    )
dataSettings = (y=NegbinomData[:,1], X=NegbinomData[:,2:3], covSel=covSel, nPerGroup=5);
algoSettings = (; algoSettings..., stateSamplingMethod=:ffbs_laplace);
methodlabel = "Laplace"
θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);     

plt = plot_param_path_betareg(β1,γ1)
titles = vcat([L"\beta_{%$(j-1)}" for j in 1:p])
PlotPostParamEvolution!(plt, quant_paramtime, methodlabel, groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=3, c=colors[3])

push!(results, (
                name="$methodlabel $(dist), nPerGroup $(nPerGroup)",
                quant_paramtime=quant_paramtime,
                priorSettings=priorSettings,
                modelSettings=modelSettings,
                algoSettings=algoSettings,
                groupSizes=groupSizes,
                nFailure=nFailure[])
                    )

# Beta regression
dist = "Beta"
link = (LogitLink(),LogLinLink())
dataSettings = (y=BetaData[:,1], X=BetaData[:,2:3], covSel=covSel, nPerGroup=5);


observation(param, state, t) =
    @views product_distribution(
        BetaMean.(
            GLM.linkinv.(param.link[1], param.Z[1][t] * state[param.Zidx[1]]),
            linkinv.(param.link[2], param.Z[2][t] * state[param.Zidx[2]])
        )
    )
@views condMean(param, state, t) = linkinv.(param.link[1],
    param.Z[1][t] * state[param.Zidx[1]])
function condCov(param, state, t)
    @views begin
        μ = linkinv.(param.link[1], param.Z[1][t] * state[param.Zidx[1]])
        ψ = linkinv.(param.link[2], param.Z[2][t] * state[param.Zidx[2]])
    end
    return diagm(μ .* (1 .- μ) ./ (1 .+ ψ))
end
modelSettings = (
    observation=observation,
    link=link, # link functions for mean and precision
    condMean=condMean,
    condCov=condCov,
    innovModel=:dsp,   # choices: :dsp, :homogaussuniv
    α=1 / 2,
    β=1 / 2,
    updateσₙ=false, # Update σ²ₙ in the Gibbs sampler, or set σₙ = 1
    nMixComp=10,    # nComp in mixture approximation of log χ²₁. Only 5 or 10 supported.
);
algoSettings = (; algoSettings..., stateSamplingMethod=:ffbs_slr);
methodlabel = "IPLF"
θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);     

plt = plot_param_path_betareg(β,γ)
PlotPostParamEvolution!(plt, quant_paramtime, methodlabel, groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:shaded, lw=1, c=colors[1])

push!(results, (
                name="$methodlabel $(dist), nPerGroup $(nPerGroup)",
                quant_paramtime=quant_paramtime,
                priorSettings=priorSettings,
                modelSettings=modelSettings,
                algoSettings=algoSettings,
                groupSizes=groupSizes,
                nFailure=nFailure[])
                    )


algoSettings = (; algoSettings..., stateSamplingMethod=:ffbs_laplace);
methodlabel = "Laplace"
θpost, Hpost, ϕpost, σ²ₙpost, μpost, groupSizes, nFailure = GibbsTVGLM(dataSettings,
    priorSettings, modelSettings, algoSettings);

prcFailure = 100 * nFailure[] / (algoSettings.nBurn + algoSettings.nIter);
println("$(algoSettings.stateSamplingMethod) failed at $(prcFailure)% 
    of the simulated trajectories")

# Parameter quantiles on the parameter time scale - this always includes t=0
quant_paramtime = quantile_multidim(θpost, [0.025, 0.5, 0.975], dims=3);     

plt = plot_param_path_betareg(β, γ)
titles = vcat([L"\beta_{%$(j-1)}" for j in 1:p])
PlotPostParamEvolution!(plt, quant_paramtime, methodlabel, groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:shaded, lw=1, c=colors[1])

push!(results, (
                name="$methodlabel $(dist), nPerGroup $(nPerGroup)",
                quant_paramtime=quant_paramtime,
                priorSettings=priorSettings,
                modelSettings=modelSettings,
                algoSettings=algoSettings,
                groupSizes=groupSizes,
                nFailure=nFailure[])
                    )


save_path = "/Users/niuyijie/Dropbox/TV_GLM_DSP/ClusterOUT/PoisSim/IPLF_Fail(none).jld2"
@save save_path results
using JLD2
results = load(save_path)["results"]

methodvec = ["IPLF", "Laplace"]
plt = plot_param_path_poisreg(β)
#titles = vcat([L"\beta_{%$(j-1)}" for j in 1:p])
PlotPostParamEvolution!(plt, results[1].quant_paramtime, methodvec[1], groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=3, c=colors[3])
PlotPostParamEvolution!(plt, results[2].quant_paramtime, methodvec[2], groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=3, c=colors[4])
plot!(plt[1], legend=:topright)

pltexp = plot_param_path_poisreg(β)
PlotPostParamEvolution!(pltexp , results[3].quant_paramtime, methodvec[1], groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=3, c=colors[3])
PlotPostParamEvolution!(pltexp , results[4].quant_paramtime, methodvec[2], groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=3, c=colors[4])
plot!(pltexp[1], legend=:topright)

pltwork = plot(plt, pltexp, layout=(1, 2), size = (1200, 800))

pltneg1 = plot_param_path_betareg(β1, γ1)
PlotPostParamEvolution!(pltneg1, results[5].quant_paramtime, methodvec[1], groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=3, c=colors[3])
plot!(pltneg1[1], legend=:topright)

pltneg2 = plot_param_path_betareg(β1, γ1)
PlotPostParamEvolution!(pltneg2, results[6].quant_paramtime, methodlabel, groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=3, c=colors[4])
plot!(pltneg2[1], legend=:topright)

pltbeta1 = plot_param_path_betareg(β, γ)
PlotPostParamEvolution!(pltbeta1, results[7].quant_paramtime, methodvec[1], groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=3, c=colors[3])
plot!(pltbeta1[1], legend=:topright)

pltbeta2 = plot_param_path_betareg(β1, γ1)
PlotPostParamEvolution!(pltbeta2, results[6].quant_paramtime, methodlabel, groupSizes;
   interpMethod=interpMethod, plot_t0=keep_t0, interval_style=:solid, lw=3, c=colors[4])
plot!(pltbeta2[1], legend=:topright)

pltfail = plot(pltneg1, pltneg2, pltbeta1,pltbeta2,layout=(2, 2), size = (2400, 2000))
savefig(pltwork, "/Users/niuyijie/Dropbox/TV_GLM_DSP/ClusterOUT/PoisSim/IPLF_Fail(work).pdf")
savefig(pltfail, "/Users/niuyijie/Dropbox/TV_GLM_DSP/ClusterOUT/PoisSim/IPLF_Fail(fail).pdf")
