using DISTS
using Test
using Images 

@testset "DISTS.jl" begin
    net_params, weights = DISTS.load_weights(joinpath(pwd(), "weights","net_param.mat"),joinpath(pwd(), "weights","alpha_beta.mat"))
    @test isa(net_params, Dict)
    @test isa(weights, Dict)
    resize_img = false
    use_gpu = false
    ref = Images.load(joinpath(pwd(), "images","r0.png"))
    dist = Images.load(joinpath(pwd(), "images","r1.png"))
    @test round(DISTS.DISTS_score(ref, dist, net_params, weights, resize_img, use_gpu; pretraining="DISTS"), digits=4)==0.3353
end
