# initialize
function initialize_nonnegative!(x, idx_nn;
    initial=1.0)
    for i in idx_nn
        x[i] = initial
    end
    return
end

# barrier
nonnegative_barrier(x) = sum(log.(x))
nonnegative_barrier_gradient(x) = 1.0 ./ x

# product
nonnegative_product(a, b) = a .* b

function nonnegative_product_jacobian(a, b)
    Array(Diagonal(b))
end

function nonnegative_product_jacobian_inverse(a, b)
    Array(Diagonal(1.0 ./ b))
end

# target
nonnegative_target(x) = ones(length(x))

# violation
function nonnegative_violation(x̂, x, τ, idx)
    for i in idx
        x̂[i] <= (1.0 - τ) * x[i] && (return true)
    end
    return false
end
