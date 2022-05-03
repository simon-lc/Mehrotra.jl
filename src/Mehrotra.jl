module Mehrotra

using AMD
using BenchmarkTools
using Crayons
using Printf
using QDLDL
using Scratch
using LinearAlgebra
using StaticArrays
using Symbolics
using Rotations
using SparseArrays
using MeshCat
using GeometryBasics
using Plots


include("codegen.jl")
include("dimensions.jl")
include("indices.jl")
# include("linear_solver.jl")
include("methods.jl")
include("options.jl")
include("print.jl")
include("problem_data.jl")
include("solver_data.jl")
include("solver.jl")

include("cones/methods.jl")
include("cones/codegen.jl")
include("cones/cone.jl")
include("cones/nonnegative.jl")
include("cones/second_order.jl")

end # module
