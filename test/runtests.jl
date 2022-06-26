using Test
using BenchmarkTools

using LinearAlgebra
using Random
using SparseArrays
using FiniteDiff

using Mehrotra

@testset "lcp"                          verbose=true begin include("lcp.jl") end
@testset "ncp"                          verbose=true begin include("ncp.jl") end
@testset "differentiability"            verbose=true begin include("differentiability.jl") end
