# Testing funtions in TVGLMUtils.jl
@testset "TVGLMUtils.jl" begin
    @test exp_lin(2.0; x₀=3.0) ≈ exp(2.0)
    @test exp_lin_inv(exp_lin(4; x₀=3.0); x₀=3.0) ≈ 4
    @test exp_lin_inv(exp_lin(2; x₀=3.0); x₀=3.0) ≈ 2
end