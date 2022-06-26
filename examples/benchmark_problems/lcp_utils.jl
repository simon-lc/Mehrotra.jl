function unpack_lcp_non_negative_cone_parameters(parameters, num_primals, num_cone)
    off = 0
    A = reshape(parameters[off .+ (1:num_primals^2)], num_primals, num_primals)
    off += num_primals^2
    b = reshape(parameters[off .+ (1:num_primals)], num_primals)
    off += num_primals
    C = reshape(parameters[off .+ (1:num_cone^2)], num_cone, num_cone)
    off += num_cone^2
    d = reshape(parameters[off .+ (1:num_cone)], num_cone)
    off += num_cone
    return A, b, C, d
end

function unpack_lcp_second_order_cone_parameters(parameters, num_primals, num_cone)
    off = 0
    A = reshape(parameters[off .+ (1:num_primals^2)], num_primals, num_primals)
    off += num_primals^2
    b = reshape(parameters[off .+ (1:num_primals)], num_primals)
    off += num_primals
    C = reshape(parameters[off .+ (1:num_cone^2)], num_cone, num_cone)
    off += num_cone^2
    d = reshape(parameters[off .+ (1:num_cone)], num_cone)
    off += num_cone
    return A, b, C, d
end

function lcp_residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    num_primals = length(primals)
    num_cone = length(duals)
    A, b, C, d = unpack_lcp_second_order_cone_parameters(parameters, num_primals, num_cone)

    res = [
        A * y + b;
        C * z + d - s;
        # z .* s .- κ[1];
        ]
    return res
end


function random_lcp(; num_primals::Int=2, num_cone::Int=3,
        cone_type::Symbol=:non_negative_cone,
        options::Mehrotra.Options228=Mehrotra.Options228(),
        seed::Int=1,
        )
    Random.seed!(seed)

    num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

    if cone_type == :non_negative_cone
        idx_nn = collect(1:num_cone)
        idx_soc = [collect(1:0)]
    elseif cone_type == :second_order_cone
        idx_nn = collect(1:0)
        if num_cone % 2 == 0
            idx_soc = [collect(2(i-1) .+ (1:2)) for i=1:Int(num_cone/2)]
        else
            idx_soc = [[collect(1:3)]; [collect(3 + 2(i-1) .+ (1:2)) for i=1:Int((num_cone-3)/2)]]
        end
    end

    As = rand(num_primals, num_primals)
    A = As' * As
    b = rand(num_primals)
    Cs = rand(num_cone, num_cone)
    C = Cs * Cs'
    d = rand(num_cone)
    parameters = [vec(A); b; vec(C); d]

    return solver = Mehrotra.Solver(lcp_residual, num_primals, num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        options=options,
        )
end
