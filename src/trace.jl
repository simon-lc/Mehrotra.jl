mutable struct Trace{T}
    iterations::Int
end

function Trace()
    return Trace{Float64}(0)
end
