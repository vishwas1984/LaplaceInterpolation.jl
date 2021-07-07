using Test, LaplaceInterpolation, SparseArrays, LinearAlgebra

printstyled("Running tests:\n", color=:blue)

tests = ["One_dimension", "Two_dimensions", "Three_dimensions"]

for t in tests
  @testset "$t LaplaceInterpolation" begin
    include("$t.jl")
  end
end


