struct Consistency
    solved::Vector{Bool}
    differentiated::Dict{Symbol,Bool}
end

function Consistency(; keywords::Vector{Symbol}=[:all])
    differentiated = Dict{Symbol,Bool}()
    for keyword in keywords
        differentiated[keyword] = false
    end
    return Consistency([false], differentiated)
end

function set_bool!(differentiated::Dict{Symbol,Bool}, boolean::Bool)
    for key in eachindex(differentiated)
        differentiated[key] = boolean
    end
    return nothing
end


# consistency = Consistency113([false], Dict(:all => false))
# # @benchmark $(consistency.solved) = true
# # @benchmark $(consistency.differentiated[:all]) = true
# # consistency.solved .= true
# # consistency.differentiated[:all] = true
# set_bool!(consistency.differentiated, true)
#
# Main.@code_warntype set_bool!(consistency.differentiated, true)
# @benchmark $set_bool!($(consistency.differentiated), true)
#
# consistency.solved
# consistency.differentiated
#
#
# # solution should be recomputed and gradients too
# solver.consistency.solved .= false
# set_bool!(solver.consistency.differentiated, false)
