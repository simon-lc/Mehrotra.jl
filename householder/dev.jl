using SparseArrays
using LinearAlgebra
using AMD
using Plots

A = sprand(100, 100, .02)
A = A * A'
rank(A)
p_amd = amd(A)
Ao = A[p_amd, p_amd]

plot(Gray.(1e3abs.(Matrix(A))))
plot(Gray.(1e3abs.(Matrix(Ao))))
plot(Gray.(1e3abs.(Matrix(qr(Ao).R))))
plot(Gray.(1e3abs.(Matrix(qr(Ao).Q))))

qr(Ao).R

idx = mech.solver.indices

Jf = mech.solver.data.jacobian_variables_dense
D = Jf[idx.duals, idx.slacks]
S = Jf[idx.slacks, idx.duals]
Z = Jf[idx.slacks, idx.slacks]
Jc = Jf[idx.equality, idx.equality]
Jc[idx.duals, idx.duals] .-= D*Z\S

J = sparse(Jc)
plot(Gray.(1e3abs.(Matrix(Jd))))
p_amd = symamd(J)
Jo = J[p_amd, p_amd]
plot(Gray.(1e3abs.(Matrix(J))))
plot(Gray.(1e3abs.(Matrix(Jo))))
plot(Gray.(1e3abs.(Matrix(qr(Jo).R))))
plot(Gray.(1e3abs.(Matrix(qr(Jo).Q))))
plot(Gray.(1e3abs.(Matrix(lu(Jo).L))))
plot(Gray.(1e3abs.(Matrix(lu(Jo).U))))

using BenchmarkTools
@benchmark $qr($J)
@benchmark $lu($J)
mech.solver.dimensions.primals
mech.solver.dimensions.cone

n = 5
A = rand(n,n)
Ac = deepcopy(A)
function myqr(A::AbstractMatrix{T}) where T
    n, m = size(A)
    τ = zeros(T,n)
    U = zeros(T,n,n)
    R = zeros(T,n,n)
    
    for i = 1:n
        χ1 = A[i,i]
        x2 = A[i+1:end,i]
        x = A[i:end,i]
        sign = (χ1 >= 0) ? 1 : -1
        ρ = -sign * norm(x)
        ν1 = χ1 + sign * norm(x)
        u2 = x2 ./ ν1
        τ[i] = (1 + u2'*u2) /2
        U[i+1:end,i] .= u2
        
        A[i,i] = ρ
        
        A[i+1:end,i] .= 0
        
        a12 = A[i,i+1:end]
        w12 = (a12 + A[i+1:end, i+1:end]'*u2) ./ τ[i]
        A[i,i+1:end] .-= w12
        
        A[i+1:end,i+1:end] .-= u2*w12'
    end
    return A, τ, U, R
end
Ac, τ, U, R = myqr(Ac)

plot(Gray.(1e3abs.(A)))
plot(Gray.(1e3abs.(Ac)))
plot(Gray.(1e3abs.(qr(Ac).R - Ac)))