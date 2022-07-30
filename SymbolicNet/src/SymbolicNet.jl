module SymbolicNet

using Symbolics
using BenchmarkTools
using LinearAlgebra
using FiniteDiff
using Test
using Graphs
using GraphRecipes
using Plots
using StaticArrays

include("macro.jl")
include("net.jl")
include("evaluation.jl")

end
