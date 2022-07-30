using Test
using BenchmarkTools

using LinearAlgebra
using Random
using FiniteDiff

using SymbolicNet

@testset "api"             verbose=true begin include("api.jl") end
