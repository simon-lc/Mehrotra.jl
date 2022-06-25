module Mehrotra

using AMD
using BenchmarkTools
using Crayons
using Printf
using QDLDL
using Scratch
using LinearAlgebra
using Symbolics
using Rotations
using SparseArrays
using MeshCat
using GeometryBasics
using FiniteDiff

include("dimensions.jl")
include("indices.jl")
include("codegen.jl")
# include("linear_solver.jl")
include("methods.jl")
include("options.jl")
include("point.jl")
include("print.jl")
include("problem_data.jl")
include("solver_data.jl")
include("trace.jl")

include("cones/methods.jl")
include("cones/codegen.jl")
include("cones/cone.jl")
include("cones/nonnegative.jl")
include("cones/second_order.jl")
include("cones/cone_search.jl")

include("solver.jl")
include("initialize.jl")
include("evaluate.jl")
include("residual.jl")
include("search_direction.jl")
include("centering.jl")
include("solve.jl")
include("violation.jl")

export
    Dimensions,
    Indices

end # module
