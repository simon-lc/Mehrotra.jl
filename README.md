[![CI](https://github.com/simon-lc/Mehrotra.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/simon-lc/Mehrotra.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/simon-lc/Mehrotra.jl/branch/main/graph/badge.svg?token=XTJdkIODOX)](https://codecov.io/gh/simon-lc/Mehrotra.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://simon-lc.github.io/Mehrotra.jl/dev)

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
- z: dual variables
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
```

## Remaining tasks
- [ ] primals regualization 
- [ ] interface so that we can provide equality residual only and the solver builds methods internally ForwardDiff etc
- [ ] exploit sparsity
- [ ] @turbo loop vectorization
- [ ] in-place addition to a vector using symbolics
- [ ] custom pure-julia sparse solver
- [ ] sparse allocation-free linear solve
- [ ] improve upon compressed step
- [ ] think about a general framework to handle QP, LP, DynamicsLCP, DynamicsNCP
- [ ] warm-starting strategy (add user-provided slacks initialization method s = F(z) in general)
- [x] consistency logic for efficient dynamics query
- [ ] exploit structure of the symmetric problems
- [x] allocation-free implementation
- [ ] experiment with different relaxation scheduling strategies
- [x] interface to provide your own function evaluations and gradients
- [x] provide your own linear system solver
- [x] implement different benchmark problems 
- [x] add tests
- [ ] add documentation
- [ ] register package
- [ ] add quaternion support
- [x] differentiate solution

### Name

complementarity solver
conic
predictor corrector
mehrotra
primal dual 
interior point method
warm start
differentiable
smooth
LP
QP
LCP
NCP


