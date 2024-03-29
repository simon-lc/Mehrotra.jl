# Get Started

__[Mehrotra](https://github.com/simon-lc/Mehrotra.jl) is a differentiable solver for nonlinear complementarity problems with conic constraints__. The solver is written in pure Julia in order to be both performant and easy to use.

## Features
* __Differentiable__: Solutions are efficiently differentiable with respect to problem data provided to the solver
* __Second-Order-Cone Constraints__: Cone constraints are natively supported in the non-convex problem setting
* __Codegen for Derivatives__: User-provided functions (e.g., objective, constraints) are symbolically differentiated and fast code is autogenerated using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl)
* __Open Source__: Code is available on [GitHub](https://github.com/simon-lc/Mehrotra.jl) and distributed under the MIT Licence

## Installation
Mehrotra can be installed using the Julia package manager for Julia `v1.7` and higher. Inside the Julia REPL, type `]` to enter the Pkg REPL mode then run:

`pkg> add Mehrotra`

If you want to install the latest version from Github run:

`pkg> add Mehrotra#main`

## Citation
If this project is useful for your work please consider:
* [Citing](citing.md) the relevant paper
* Leaving a star on the [GitHub repository](https://github.com/simon-lc/Mehrotra.jl)

## Licence
Mehrotra is licensed under the MIT License. For more details click [here](https://github.com/simon-lc/Mehrotra.jl/blob/main/LICENSE.md).
