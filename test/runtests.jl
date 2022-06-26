using Test
using BenchmarkTools

using LinearAlgebra
using Random
using SparseArrays
using FiniteDiff

using Mehrotra

@testset "random ncp"                   verbose=true begin include("random_ncp.jl") end
@testset "contact ncp"                  verbose=true begin include("contact_ncp.jl") end
# @testset "differentiability"            verbose=true begin include("differentiability.jl") end
