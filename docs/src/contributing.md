# Contributing

Contributions are always welcome!

* If you want to contribute features, bug fixes, etc, please take a look at our __Code Style Guide__ below
* Please report any issues and bugs that you encounter in [Issues](https://github.com/simon-lc/Mehrotra.jl/issues)
* As an open source project we are also interested in any projects and applications that use Mehrotra. Please let us know via email to: simonlc@stanford.edu

## Potentially Useful Contributions
Here are a list of current to-do's that would make awesome contributions:

- support for sparse [FiniteDiff](https://github.com/JuliaDiff/FiniteDiff.jl), [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) derivatives
- support for additional linear-system solvers like: [Pardiso](https://github.com/JuliaSparse/Pardiso.jl), [HSL](https://github.com/JuliaSmoothOptimizers/HSL.jl), or [Krylov](https://github.com/JuliaSmoothOptimizers/Krylov.jl)
- integration with [ChainRules](https://github.com/JuliaDiff/ChainRules.jl)
- precompilation
- Python + JAX/Torch wrapper

## Code Style Guide

The code in this repository follows the naming and style conventions of [Julia Base](https://docs.julialang.org/en/v1.0/manual/style-guide/#Style-Guide-1) with a few modifications. This style guide is heavily "inspired" by the guides of [John Myles White](https://github.com/johnmyleswhite/Style.jl), [JuMP](http://www.juliaopt.org/JuMP.jl/latest/style), and [COSMO](https://github.com/oxfordcontrol/COSMO.jl)

### Formatting
* Use one tab when indenting a new block (except `module`)

* Use spaces between operators, except for `^`, `'`, and `:`
* Use single space after commas and semicolons
* Don't use spaces around parentheses, or braces

**Bad**: `f(x,y) = [5*sin(x+y);y']` **Good**: `f(x, y) = [5 * sin(x + y); y']`
* Use spacing with keyword arguments

**Bad**: `foo(x::Float; y::Integer = 1)` **Good**: `foo(x::Float; y::Integer=1)`

* Don't parenthesize conditions

**Bad**: `if (a == b)` **Good**: `if a == b`
### Naming
* Modules and Type names use capitilization and camel case, e.g. `module LinearAlgebra`, `struct ConvexSets`.
* Functions are lowercase and use underscores to seperate words, e.g. `has_key(x)`, `is_valid(y)`.
* Normal variables are lowercase and use underscores like functions, e.g. `convex_set`
* Constants are uppercase, e.g. `const MY_CONSTANT`
* **Always** append `!` to names of functions that modify their arguments.
* Function arguments that are mutated come first. Otherwise follow the rules layed out in Julia Base [Argument ordering](https://docs.julialang.org/en/v1.0/manual/style-guide/#Write-functions-with-argument-ordering-similar-to-Julia-Base-1)
* Files are named like functions, e.g. `my_new_file.jl`

### Syntax
* Use `1.0` instead of `1.`
