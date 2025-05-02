
#%%imports
include(joinpath(@__DIR__, "../src_jl/LVisP.jl"))
using .LVisP

#%%tests
@testset "LVisP" begin
    @testset "helper functions" begin
        @test begin
            truth = [(1.0, 0.0pi), (1.0, pi/2), (1.0, 1.0pi), (1.0, 3pi/2), (2.0,pi/4)]
            res = LVisP.carth2polar.([1,0,-1,0,sqrt(2)],[0,1,0,-1,sqrt(2)])
            all(res .== truth)
        end
        @test begin
            truth = [(1.,0.),(0.,1.),(-1.,0.),(0.,-1.),(sqrt(2),sqrt(2))]
            res = LVisP.polar2carth.([1.0,1.0,1.0,1.0,2.0],[0.0pi,pi/2,1.0pi,3pi/2,pi/4])
            all(map((r,t) -> all(isapprox.(r, t, atol=1e-12)), res, truth))
        end
        @test begin
            truth = [-0.25, -0.125, 0.0, 0.125, 0.25]
            res = LVisP.minmaxscale(collect(-1:0.5:1), -0.5, 0.5; xmin_ref=-2, xmax_ref=2)
            all(res .== truth)
        end
        @test begin
            truth = [0, 45, 90, 315, 360, 405, 270, 315, 360]
            res = LVisP.correct_labelrotation.(0:45:360)
            all(res .== truth)
        end
    end
end
