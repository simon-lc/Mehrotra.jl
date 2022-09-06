using Flux
using CUDA
using BenchmarkTools

CUDA.functional()

n = 20
m = 5
W = rand(n, m) # a 2×5 CuArray
b = rand(n)

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(m), rand(n) # Dummy data
loss(x, y) # ~ 3


W = cu(rand(n, m)) # a 2×5 CuArray
b = cu(rand(n))

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = cu(rand(m)), cu(rand(n)) # Dummy data
loss(x, y) # ~ 3


dloss(x, y) = gradient(loss, x, y)
dloss(x, y)



Tf = Float32
nβ = 100000
nh

β0 = -π + atan(e0[2], e0[1]) .+ Vector(range(+0.3π, -0.3π, length=nβ))
e0
d0 = trans_point_cloud(e0, β0, ρ0*100, θinit, polytope_dimensions)
v0 = hcat([[cos(β0[i]), sin(β0[i])] for i=1:nβ]...)
α0 = [d0[:,i]'*v0[:,i] - e0'*v0[:,i] for i=1:nβ]
AA = [Matrix(reshape(unpack_halfspaces(θinit, polytope_dimensions, i)[1], (nh,2))) for i = 1:np]
bb = [Vector(unpack_halfspaces(θinit, polytope_dimensions, i)[2]) for i = 1:np]
oo = [Vector(unpack_halfspaces(θinit, polytope_dimensions, i)[3]) for i = 1:np]
bb = [bb[i] + AA[i] * oo[i] - AA[i] * e0 for i=1:np]

α0 = convert.(Tf, α0)
v0 = convert.(Tf, v0)
AA = [convert.(Tf, Ai) for Ai in AA]
bb = [convert.(Tf, bi) for bi in bb]

cu_α0 = cu(α0)
cu_v0 = cu(v0)
cu_AA = cu.(AA)
cu_bb = cu.(bb)


function rendering_loss(cond::AbstractVector, sdf::AbstractVector, sdfv::AbstractMatrix,
    Av::AbstractMatrix, αβ::AbstractVector, αhβ::AbstractMatrix,
    α_ref::AbstractVector{T}, v::AbstractMatrix{T},
    A::AbstractMatrix{T}, b::AbstractVector{T}) where T

    # sdf nβ
    # sdfv nh x nβ
    # Av nh x nβ
    # αβ nβ
    # αhβ nh x nβ
    # α_ref nβ
    # v 2
    # A nh x 2
    # b nh

    αβ .= +Inf
    Av .= A * v
    # αhβ .= Av
    # for i = 1:nh
    #     αhβ[i,:] ./= b[i]
    # end
    αhβ .= Av ./ b
    for i = 1:nh
        # for j = 1:nh
        #     sdfv[j,:] .= αhβ[i,:] .* Av[j,:] .- b[i]
        # end
        @views sdfv .=  αhβ[i,:]' .* Av .- b[i]

        sdf .= maximum(sdfv, dims=1)[1,:]
        @views cond .= (sdf .< 1e-5) .&& (αhβ[i,:] .< αβ)
        @views αβ .= αβ .* .!cond + αhβ[i,:] .* cond
    end
    return sum((α_ref .- αβ).^2) / nβ#, αβ
end


function rendering_loss(α_ref::AbstractVector, v::AbstractMatrix,
    A::AbstractMatrix, b::AbstractVector)

    # sdf nβ
    # sdfv nh x nβ
    # Av nh x nβ
    # αβ nβ
    # αhβ nh x nβ
    # α_ref nβ
    # v 2
    # A nh x 2
    # b nh

    αβ = +Inf * ones()
    Av = A * v
    αhβ = Av ./ b
    # for i = 1:nh
    #     @views sdfv .=  αhβ[i,:]' .* Av .- b[i]
    #
    #     sdf .= maximum(sdfv, dims=1)[1,:]
    #     @views cond .= (sdf .< 1e-5) .&& (αhβ[i,:] .< αβ)
    #     @views αβ .= αβ .* .!cond + αhβ[i,:] .* cond
    # end
    αβ = sum(αhβ, dims=1)[1,:]
    return sum((α_ref .- αβ).^2) / nβ
end


A = [1 2 ;  3 4 ]
sum(A, dims=1)
cnd = zeros(Bool, nβ)
sdf0 = zeros(Tf, nβ)
sdfv = zeros(Tf, nh, nβ)
Av = zeros(Tf, nh, nβ)
αβ = zeros(Tf, nβ)
αhβ = zeros(Tf, nh, nβ)

cu_cnd = cu(cnd)
cu_sdf0 = cu(sdf0)
cu_sdfv = cu(sdfv)
cu_Av = cu(Av)
cu_αβ = cu(αβ)
cu_αhβ = cu(αhβ)

@elapsed rendering_loss(
    cnd,
    sdf0,
    sdfv,
    Av,
    αβ,
    αhβ,
    α0,
    v0,
    AA[1],
    bb[1],
    )

@elapsed rendering_loss(
    α0,
    v0,
    AA[1],
    bb[1],
    )

@elapsed rendering_loss(
    cu_cnd,
    cu_sdf0,
    cu_sdfv,
    cu_Av,
    cu_αβ,
    cu_αhβ,
    cu_α0,
    cu_v0,
    cu_AA[1],
    cu_bb[1],
    )

gradient(rendering_loss())

dAloss() = gradient(rendering_loss,
    # cnd,
    # sdf0,
    # sdfv,
    # Av,
    # αβ,
    # αhβ,
    α0,
    v0,
    AA[1],
    bb[1])[3:4]

α0
v0
AA[1]
dAloss()
@benchmark dAloss()


@benchmark ForwardDiff.gradient(b -> rendering_loss(
    α0,
    v0,
    AA[1],
    b,
    ),
    bb[1])
@benchmark ForwardDiff.gradient(A -> rendering_loss(
    α0,
    v0,
    A,
    bb[1],
    ),
    AA[1])


gs = gradient(() -> rendering_loss(A, b), Flux.params(AA[1], bb[1]))


# @benchmark rendering_loss(
#     cnd,
#     sdf0,
#     sdfv,
#     Av,
#     αβ,
#     αhβ,
#     α0,
#     v0,
#     AA[1],
#     bb[1],
#     )

# @benchmark rendering_loss(
#     cu_cnd,
#     cu_sdf0,
#     cu_sdfv,
#     cu_Av,
#     cu_αβ,
#     cu_αhβ,
#     cu_α0,
#     cu_v0,
#     cu_AA[1],
#     cu_bb[1],
#     )
#

# Main.@profiler [rendering_loss(
#     cu_cnd,
#     cu_sdf0,
#     cu_sdfv,
#     cu_Av,
#     cu_αβ,
#     cu_αhβ,
#     cu_α0,
#     cu_v0,
#     cu_AA[1],
#     cu_bb[1],
#     ) for i = 1:1000]
#
