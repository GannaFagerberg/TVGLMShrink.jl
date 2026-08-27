using Plots
using GLM

η = range(-10, 10, length=500)

μ_logit   = GLM.linkinv.(Ref(GLM.LogitLink()), η)
μ_cauchit = GLM.linkinv.(Ref(GLM.CauchitLink()), η)
μ_probit  = GLM.linkinv.(Ref(GLM.ProbitLink()), η)

plot(
    η,
    μ_logit,
    label="Logit",
    lw=2,
    xlabel="η",
    ylabel="μ",
    legend=:topleft
)

plot!(
    η,
    μ_cauchit,
    label="Cauchit",
    lw=2,
    color="green"
)

plot!(
    η,
    μ_probit,
    label="Probit",
    lw=2,
    color="red"
)

μ_cloglog = GLM.linkinv.(Ref(GLM.CloglogLink()), η)

plot!(
    η,
    μ_cloglog,
    label="Cloglog",
    lw=2,
    color="orange"
)