# Testing funtions in TVGLMModels.jl
@testset "TVGLMModels.jl" begin
    link = LogLinLink(3.0)
    @test linkfun(link, 2.0) ≈ log(2.0)
    @test linkinv(link, 2.0) ≈ exp(2.0)
end