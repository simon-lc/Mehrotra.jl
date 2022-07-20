################################################################################
# methods
################################################################################
function generate_gradients(func::Function, num_equality::Int, num_variables::Int,
        num_parameters::Int;
        checkbounds=true,
        threads=false)

    f = Symbolics.variables(:f, 1:num_equality)
    e = Symbolics.variables(:e, 1:num_equality)
    x = Symbolics.variables(:x, 1:num_variables)
    θ = Symbolics.variables(:θ, 1:num_parameters)

    f .= e
    func(f, x, θ)

    fx = Symbolics.sparsejacobian(f, x)
    fθ = Symbolics.sparsejacobian(f, θ)

    fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
    fθ_sparsity = collect(zip([findnz(fθ)[1:2]...]...))

    f_expr = Symbolics.build_function(f, e, x, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fx_expr = Symbolics.build_function(fx.nzval, x, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fθ_expr = Symbolics.build_function(fθ.nzval, x, θ,
        parallel=((threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    return f_expr, fx_expr, fθ_expr, fx_sparsity, fθ_sparsity
end

abstract type NodeMethods182{T,E,EX,Eθ} end

struct DynamicsMethods182{T} <: AbstractProblemMethods{T,E,EX,EP}
    methods::Vector{NodeMethods182}
    α::T
end

struct BodyMethods182{T,E,EX,Eθ} <: NodeMethods182{T,E,EX,Eθ}
    equality_constraint::E
    equality_jacobian_variables::EX
    equality_jacobian_parameters::Eθ
    equality_jacobian_variables_cache::Vector{T}
    equality_jacobian_parameters_cache::Vector{T}
    equality_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
end

function BodyMethods182(body::Body182, dimensions::MechanismDimensions182)
    r!(e, x, θ) = body_residual!(e, x, θ, body)
    f, fx, fθ, fx_sparsity, fθ_sparsity = generate_gradients(r!, dimensions.equality,
        dimensions.variables, dimensions.parameters)
    return BodyMethods182(
        f,
        fx,
        fθ,
        zeros(length(fx_sparsity)),
        zeros(length(fθ_sparsity)),
        fx_sparsity,
        fθ_sparsity,
        )
end

struct ContactMethods182{T,E,EX,Eθ,C,S} <: NodeMethods182{T,E,EX,Eθ}
    contact_solver::C
    subvariables::Vector{T}
    subparameters::Vector{T}

    set_subparameters!::S
    equality_constraint::E
    equality_jacobian_variables::EX
    equality_jacobian_parameters::Eθ
    equality_jacobian_variables_cache::Vector{T}
    equality_jacobian_parameters_cache::Vector{T}
    equality_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
end

function ContactMethods182(contact::PolyPoly182, pbody::Body182, cbody::Body182,
        dimensions::MechanismDimensions182;
        checkbounds=true,
        threads=false)


    contact_solver = ContactSolver(
        contact.A_parent_collider,
        contact.b_parent_collider,
        contact.A_child_collider,
        contact.b_child_collider,
        )

    num_equality = dimensions.equality
    num_variables = dimensions.variables
    num_parameters = dimensions.parameters
    num_subvariables = contact_solver.num_subvariables
    num_subparameters = contact_solver.num_subparameters
    subvariables = zeros(num_subvariables)
    subparameters = zeros(num_subparameters)

    # set_subparameters!
    x = Symbolics.variables(:x, 1:num_variables)
    θ = Symbolics.variables(:θ, 1:num_parameters)
    v25_parent = unpack_body_variables(x[pbody.index.x])
    v25_child = unpack_body_variables(x[cbody.index.x])

    x2_parent, _, _, timestep_parent = unpack_body_parameters(θ[pbody.index.θ], pbody)
    x2_child, _, _, timestep_child = unpack_body_parameters(θ[cbody.index.θ], cbody)
    x3_parent = x2_parent .+ timestep_parent[1] * v25_parent
    x3_child = x2_child .+ timestep_child[1] * v25_child
    @show x2_parent, timestep_parent, x3_parent, v25_parent

    Ap, bp, Ac, bc = unpack_contact_parameters(θ[contact.index.θ], contact)

    # θl = fct(x, θ)
    θl = [x3_parent; x3_child; vec(Ap); bp; vec(Ac); bc]
    θl2 = [x2_parent; x2_child; vec(Ap); bp; vec(Ac); bc]

    set_subparameters! = Symbolics.build_function(θl, x, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    # evaluation
    f = Symbolics.variables(:f, 1:num_equality)
    e = Symbolics.variables(:e, 1:num_equality)
    xl = Symbolics.variables(:xl, 1:num_subvariables)
    ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child = unpack_contact_subvariables(xl, contact)


    f .= e
    contact_residual!(f, x, xl, θ, contact, pbody, cbody)

    # for this one we are missing only third order tensors
    fx = Symbolics.sparsejacobian(f, x)
    fx[contact.index.e, [pbody.index.x; cbody.index.x]] .+= -sparse(N)
    # for this one we are missing ∂ϕ/∂θ and third order tensors
    fθ = Symbolics.sparsejacobian(f, θ)

    fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
    fθ_sparsity = collect(zip([findnz(fθ)[1:2]...]...))

    f_expr = Symbolics.build_function(f, e, x, xl, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fx_expr = Symbolics.build_function(fx.nzval, x, xl, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fθ_expr = Symbolics.build_function(fθ.nzval, x, xl, θ,
        parallel=((threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    return ContactMethods182(
        contact_solver,
        subvariables,
        subparameters,
        set_subparameters!,
        f_expr,
        fx_expr,
        fθ_expr,
        zeros(length(fx_sparsity)),
        zeros(length(fθ_sparsity)),
        fx_sparsity,
        fθ_sparsity,
    )
end

function mechanism_methods(bodies::Vector, contacts::Vector, dimensions::MechanismDimensions182)
    methods = Vector{NodeMethods182}()

    # body
    for body in bodies
        push!(methods, BodyMethods182(body, dimensions))
    end

    # contact
    for contact in contacts
        # TODO here we need to avoid hardcoding body1 and body2 as paretn and child
        push!(methods, ContactMethods182(contact, bodies[1], bodies[2], dimensions))
    end

    return DynamicsMethods182(methods, 1.0)
end

################################################################################
# evaluate
################################################################################

# function evaluate!(e::Vector{T}, ex::Matrix{T}, eθ::Matrix{T},
#         x::Vector{T}, θ::Vector{T}, methods::Vector{NodeMethods182}) where T
#     e .= 0.0
#     ex .= 0.0
#     eθ .= 0.0
#     for m in methods
#         evaluate!(e, ex, eθ, x, θ, m)
#     end
# end
#
# function evaluate!(e::Vector{T}, ex::Matrix{T}, eθ::Matrix{T},
#         x::Vector{T}, θ::Vector{T}, methods::BodyMethods182{T,E,EX,Eθ}) where {T,E,EX,Eθ}
#
#     methods.equality_constraint(e, e, x, θ)
#     methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, θ)
#     methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, θ)
#
#     for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
#         ex[idx...] += methods.equality_jacobian_variables_cache[i]
#     end
#     for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
#         eθ[idx...] += methods.equality_jacobian_parameters_cache[i]
#     end
# end
#
# function evaluate!(e::Vector{T}, ex::Matrix{T}, eθ::Matrix{T},
#         x::Vector{T}, θ::Vector{T}, methods::ContactMethods182{T,S}) where {T,S}
#
#     contact_solver = methods.contact_solver
#     xl = methods.subvariables
#     θl = methods.subparameters
#
#     # update xl = [ϕ, pa, pb, N, ∂pa, ∂pb]
#     methods.set_subparameters!(θl, x, θ)
#     update_subvariables!(xl, θl, contact_solver)
#
#     # modify e, ex, eθ in-place using symbolics methods taking x, θ, xl as inputs
#     methods.equality_constraint(e, e, x, xl, θ)
#     methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, xl, θ)
#     methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, xl, θ)
#
#     for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
#         ex[idx...] += methods.equality_jacobian_variables_cache[i]
#     end
#     for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
#         eθ[idx...] += methods.equality_jacobian_parameters_cache[i]
#     end
# end

function evaluate!(
        problem::ProblemData{T},
        methods::DynamicsMethods182{T},
        cone_methods::ConeMethods{T,B,BX,P,PX,PXI},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        cone_constraint=false,
        cone_jacobian=false,
        cone_jacobian_inverse=false,
        ) where {T,B,BX,P,PX,PXI}

    # TODO this method allocates memory, need fix

    # reset
    problem.equality_constraint .= 0.0
    problem.equality_jacobian_variables .= 0.0
    problem.equality_jacobian_parameters .= 0.0

    # apply all methods
    for method in methods.methods
        evaluate!(problem, method, solution, parameters;
            equality_constraint=equality_constraint,
            equality_jacobian_variables=equality_jacobian_variables,
            equality_jacobian_parameters=equality_jacobian_parameters)
    end

    # evaluate candidate cone product constraint, cone target and jacobian
    cone!(problem, cone_methods, solution,
        cone_constraint=cone_constraint,
        cone_jacobian=cone_jacobian,
        cone_jacobian_inverse=cone_jacobian_inverse,
    )

    return nothing
end

function evaluate!(problem::ProblemData{T},
        methods::BodyMethods182{T,E,EX,Eθ},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        ) where {T,E,EX,Eθ}

    x = solution.all
    θ = parameters

    # dimensions
    nθ = length(θ)

    # equality
    ne = length(problem.equality_constraint)

    (equality_constraint && ne > 0) && methods.equality_constraint(
        problem.equality_constraint, problem.equality_constraint, x, θ)

    if (equality_jacobian_variables && ne > 0)
        methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
            problem.equality_jacobian_variables[idx...] += methods.equality_jacobian_variables_cache[i]
        end
    end

    if (equality_jacobian_parameters && ne > 0 && nθ > 0)
        methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
            problem.equality_jacobian_parameters[idx...] += methods.equality_jacobian_parameters_cache[i]
        end
    end
    return
end

function evaluate!(problem::ProblemData{T},
        # methods::ContactMethods182{T,E,EX,Eθ},
        methods::ContactMethods182{T,S},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        # ) where {T,E,EX,Eθ}
        ) where {T,S}

    x = solution.all
    θ = parameters

    # dimensions
    nθ = length(θ)

    # equality
    ne = length(problem.equality_constraint)

    # update xl = [ϕ, pa, pb, N, ∂pa, ∂pb]
    contact_solver = methods.contact_solver
    xl = methods.subvariables
    θl = methods.subparameters
    methods.set_subparameters!(θl, x, θ)
    update_subvariables!(xl, θl, contact_solver)
    # @show x[1:3]
    # @show x[4:6]
    # @show θl[1:3]
    # @show θl[4:6]
    # @show xl[1]

    # update equality constraint and its jacobiens
    (equality_constraint && ne > 0) && methods.equality_constraint(
        problem.equality_constraint, problem.equality_constraint, x, xl, θ)

    if (equality_jacobian_variables && ne > 0)
        methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, xl, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
            problem.equality_jacobian_variables[idx...] += methods.equality_jacobian_variables_cache[i]
        end
    end

    if (equality_jacobian_parameters && ne > 0 && nθ > 0)
        methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, xl, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
            problem.equality_jacobian_parameters[idx...] += methods.equality_jacobian_parameters_cache[i]
        end
    end
    return
end
