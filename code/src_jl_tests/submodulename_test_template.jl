

#%%imports

include(joinpath(@__DIR__, "../src_jl/submodulename_test_template.jl"))
using .submodulename_test_template

#%%tests
@testset "submodulename_test_template" begin
    @testset "struct or function" begin
        @test begin
            #your test
        end
    end
end