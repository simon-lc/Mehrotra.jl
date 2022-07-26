abstract type Shape1140{T} end

struct PolytopeShape1140{T}
    A::Matrix{T}
    b::Vector{T}
end

function PolytopeShape1140(A, b)
    T = eltype(b[1])
    return PolytopeShape1140{T}(A, b)
end

struct SphereShape1140{T}
    radius::Vector{T}
    position_offset::Vector{T}
end

function SphereShape1140(radius::T, position_offset=zeros(T,2)) where T
    return SphereShape1140{T}([radius], position_offset)
end
