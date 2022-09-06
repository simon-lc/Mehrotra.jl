# sCOPas
# pygmalion
# galatea
# daedalus
#
#
# Convex Polytope
# Hierarchical
# Convex decomposition
# Shape Decomposition
# Sim to Real
# Differentiable
# Real to Sim
# Sequential

using CUDA
CUDA.functional()

W = cu(rand(2, 5)) # a 2×5 CuArray
b = cu(rand(2))

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = cu(rand(5)), cu(rand(2)) # Dummy data
loss(x, y) # ~ 3
