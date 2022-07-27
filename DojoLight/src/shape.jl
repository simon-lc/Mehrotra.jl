abstract type Shape1160{T} end

struct PolytopeShape1160{T}
    A::Matrix{T}
    b::Vector{T}
end

function PolytopeShape1160(A, b)
    T = eltype(b[1])
    return PolytopeShape1160{T}(A, b)
end

struct SphereShape1160{T}
    radius::Vector{T}
    position_offset::Vector{T}
end

function SphereShape1160(radius::T, position_offset=zeros(T,2)) where T
    return SphereShape1160{T}([radius], position_offset)
end
