using Symbolics

struct SequenceMethods{T,M}   
    methods::Vector{M}
    equality_jacobian_variables_cache::Vector{T}
    equality_jacobian_parameters_cache::Vector{T}
    equality_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
end

struct Methods111{E,EX,Eθ}
    equality_constraint::E                             # e
    equality_jacobian_variables::EX                    # ex
    equality_jacobian_parameters::Eθ                   # eθ
end

function symbolics_methods(out_of_place_equality::Function, dim::Dimensions, idx::Indices)
    function in_place_equality!(e, x, θ)
        e .= out_of_place_equality(x[idx.primals], x[idx.duals], x[idx.slacks], θ)
        return nothing
    end

    e, ex, eθ, ex_sparsity, eθ_sparsity = generate_gradients(in_place_equality!, dim)

    methods = [Methods111(e, ex, eθ)]
    sequence_methods = SequenceMethods(
        methods,
        zeros(length(ex_sparsity)),
        zeros(length(eθ_sparsity)),
        ex_sparsity,
        eθ_sparsity,
    )

    return sequence_methods
end

function symbolics_methods(in_place_equalities!::Vector{Function}, dim::Dimensions, idx::Indices)
    methods = Vector{Methods111}()
    Ex_sparsity = Vector{Tuple{Int,Int}}()
    Eθ_sparsity = Vector{Tuple{Int,Int}}()

    for in_place_equality! in in_place_equalities!
        e, ex, eθ, ex_sparsity, eθ_sparsity = generate_gradients(in_place_equality!, dim)
        push!(methods, Methods111(e, ex, eθ))
        union!(Ex_sparsity, ex_sparsity)
        union!(Eθ_sparsity, eθ_sparsity)
    end

    sequence_methods = SequenceMethods(
        methods,
        zeros(length(Ex_sparsity)),
        zeros(length(Eθ_sparsity)),
        Ex_sparsity,
        Eθ_sparsity,
    )

    return sequence_methods
end


"""
    func!(e, x, θ)
"""
function generate_gradients(func!::Function, dim::Dimensions;
        checkbounds=true,
        threads=false)

    parallel = threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()
    parallel_parameters = (threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()

    x = Symbolics.variables(:x, 1:dim.variables)
    θ = Symbolics.variables(:θ, 1:dim.parameters)
    e = Symbolics.variables(:e, 1:dim.equality)
    f = Symbolics.variables(:f, 1:dim.equality)
    f .= e

    dim.parameters > 0 ? func!(f, x, θ) : func!(f, x)

    fx = Symbolics.sparsejacobian(f, x)
    fθ = Symbolics.sparsejacobian(f, θ)

    fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
    fθ_sparsity = collect(zip([findnz(fθ)[1:2]...]...))

    f_expr = Symbolics.build_function(f, e, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fx_expr = Symbolics.build_function(fx.nzval, e, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fθ_expr = Symbolics.build_function(fθ.nzval, e, x, θ,
        parallel=parallel_parameters,
        checkbounds=checkbounds,
        expression=Val{false})[2]

    return f_expr, fx_expr, fθ_expr, fx_sparsity, fθ_sparsity
end


function sequence_evaluate!(problem::ProblemData{T},
        methods::Methods111{E,EX,EP},
        sequence_methods::SequenceMethods{T},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        ) where {T,E,EX,EP}

    x = solution.all
    θ = parameters

    # dimensions
    nθ = length(θ)

    # equality
    ne = length(problem.equality_constraint)

    if equality_constraint && ne > 0
        methods.equality_constraint(
            problem.equality_constraint, 
            problem.equality_constraint, 
            x, θ)
    end

    if (equality_jacobian_variables && ne > 0)
        methods.equality_jacobian_variables(
            sequence_methods.equality_jacobian_variables_cache, 
            sequence_methods.equality_jacobian_variables_cache, 
            x, θ)
        # problem.equality_jacobian_variables.nzval .= methods.equality_jacobian_variables_cache
    end

    if (equality_jacobian_parameters && ne > 0 && nθ > 0)
        methods.equality_jacobian_parameters(
            sequence_methods.equality_jacobian_parameters_cache, 
            sequence_methods.equality_jacobian_parameters_cache, 
            x, θ)
        # problem.equality_jacobian_parameters.nzval .= methods.equality_jacobian_parameters_cache
    end

    return
end




options=Options(
        verbose=false, 
        complementarity_tolerance=1e-4,
        # compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=false,
        warm_start=true,
        )

bodies, contacts = get_convex_drop(; 
    timestep=0.05, 
    gravity=-9.81, 
    mass=1.0, 
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    options=options,
    )

nodes = [bodies; contacts]
num_variables = sum(variable_dimension.(nodes))
num_primals = sum(primal_dimension.(nodes))
num_cone = sum(cone_dimension.(nodes))
num_parameters = sum(parameter_dimension.(nodes))
num_equality = num_primals + num_cone

dim = Dimensions(num_primals, num_cone, num_parameters);
idx = Indices(num_primals, num_cone, num_parameters);


local_mechanism_residual(primals, duals, slacks, parameters) = 
    mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)
sequence_methods0 = symbolics_methods(local_mechanism_residual, dim, idx);
methods0 = sequence_methods0.methods[1];


local_residuals = Vector{Function}()
# body
for body in bodies
    push!(local_residuals, (e, x, θ) -> residual!(e, x, θ, body))
end

# contact
for contact in contacts
    push!(local_residuals, (e, x, θ) -> residual!(e, x, θ, contact, bodies))
end

local_residuals
sequence_methods1 = symbolics_methods(local_residuals, dim, idx);


solution = Point(dim, idx)
parameters = rand(num_parameters)
problem = ProblemData(num_variables, num_parameters, num_equality, num_cone)


sequence_methods0.equality_jacobian_variables_cache .= 0.0
sequence_methods0.equality_jacobian_parameters_cache .= 0.0

sequence_methods1.equality_jacobian_variables_cache .= 0.0
sequence_methods1.equality_jacobian_parameters_cache .= 0.0


sequence_evaluate!(problem,
        sequence_methods0.methods[1],
        sequence_methods0,
        solution,
        parameters;
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=false,
)

@benchmark $sequence_evaluate!($problem,
        $(sequence_methods0.methods[1]),
        $sequence_methods0,
        $solution,
        $parameters;
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
    )

for m in sequence_methods1.methods
    sequence_evaluate!(problem,
            m,
            sequence_methods1,
            solution,
            parameters;
            equality_constraint=true,
            equality_jacobian_variables=true,
            equality_jacobian_parameters=false,
        )
end

@benchmark $sequence_evaluate!($problem,
        $(sequence_methods1.methods[1]),
        $sequence_methods1,
        $solution,
        $parameters;
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
    )



ex0 = sequence_methods0.equality_jacobian_variables_cache
eθ0 = sequence_methods0.equality_jacobian_parameters_cache

sequence_methods0.equality_jacobian_variables_sparsity .== sequence_methods1.equality_jacobian_variables_sparsity

ex1 = sequence_methods1.equality_jacobian_variables_cache
eθ1 = sequence_methods1.equality_jacobian_parameters_cache

norm(ex0 - ex1)
norm(ex0)
norm(ex1)


norm(eθ0 - eθ1)
norm(eθ0)
norm(eθ1)



