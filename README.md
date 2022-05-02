
# Mehrotra.jl
A solver for cone-constrained feasibility problems. The main application for this solver is implicit integration of contact dynamics.

## Standard form
Problems of the following form:
```
find         x = [y, z, s]

subject to   f(y,z,s; p) = 0,
             z ∘ s = κ
             z, s in K = R+ x Q^1 x ... x Q^k
```
can be optimized for

- x = [y, z, s]: decision variables
- y: primal variables
- z: duals variables
- s: slack variables
- p: problem parameters
- κ: central-path parameter
- ∘: cone product
- K: Cartesian product of convex cones; nonnegative orthant R+ and second-order cones Q are currently implemented

## Solution gradients
The solver is differentiable, and gradients of the solution (including internal solver variables) with respect to the problem parameters are efficiently computed.

## Quick start
```julia
using Mehrotra
