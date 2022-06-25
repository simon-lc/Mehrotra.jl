mutable struct Polyhedron{T,N,D}
    A::Matrix{T}
    b::Vector{T}
    δ::T
    n::Int
    d::Int
end

function Polyhedron(A::Matrix{T}, b::Vector{T}; δ=1.0) where T
    n, d = size(A)
    @assert length(b) == n
    for i = 1:n
        A[i,:] .= normalize(A[i,:])
    end
    return Polyhedron{T,n,d}(A, b, δ, n, d)
end
