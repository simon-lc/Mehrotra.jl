using LinearAlgebra

function scn(a::Float64; digits=1, exp_digits=1)
	isnan(a) && return " NaN" * " "^(digits + exp_digits)
	@assert digits >= 0
    # a = m x 10^e
    if a == 0
        e = 0
        m = 0.0
    elseif a == Inf
		return " Inf"
	elseif a == -Inf
		return "-Inf"
	else
        e = Int(floor(log(abs(a))/log(10)))
        m = a*exp(-e*log(10))
    end

    m = round(m, digits=digits)
	if m == 10.0
		m = 1.0
		e += 1
	end
    if digits == 0
        m = Int(floor(m))
		strm = string(m)
	else
		strm = string(m)
		is_neg = m < 0.
		strm = strm*"0"^max(0, 2+digits+is_neg-length(strm))
    end
    sgn = a >= 0 ? " " : ""
    sgne = e >= 0 ? "+" : "-"

	stre = string(abs(e))
	stre = "0"^max(0, exp_digits - length(stre)) * stre
    return "$sgn$(strm)e$sgne$(stre)"
end

nx = 3
ny = 4
nw = nx + 2ny
nθ = nx + ny

# A_ = rand(nx,nx)
# A = A_ * A_'
# B = 0.1*rand(nx,ny)
# C = 0.1*rand(ny,nx)
# D = Diagonal(ones(ny))
θ = rand(nx + ny)

function unpack_data(θ)
	return θ[1:nx], θ[nx .+ (1:ny)]
end

function pack_data(b, d)
	return [b; d]
end

function residual(x, y, z, κ, θ)
	b, d = unpack_data(θ)
	rx = A*x + B*y + b.^2
    ry = C*x + D*z + d.^2
    rz = y .* z .- κ
    return [rx; ry; rz]
end

function bilinear_residual(x, y, z, κ)
    return y .* z .- κ
end

function unpack_vars(w)
    return w[1:nx], w[nx .+ (1:ny)], w[nx + ny .+ (1:ny)]
end

function pack_vars(x, y, z)
    return [x; y; z]
end

function easy_solver(x, y, z, θ; rtol=1e-4, btol=1e-4)
    w = [x; y; z]
    κ = 1e-0
	iter = 0
	println("∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘")

    for i = 1:10
		(norm(residual(unpack_vars(w)..., btol, θ), Inf) < rtol) && break
		println("-------------------------------------------------------")
        for j = 1:20
			iter += 1
            r = residual(unpack_vars(w)..., κ, θ)
			(norm(r, Inf) <= rtol) && break
			H = FiniteDiff.finite_difference_jacobian(w -> residual(unpack_vars(w)..., κ, θ), w)
            Δ = - H \ r
            α = 1.0
			for k = 1:100
				all((w + α*Δ)[nx .+ (1:2ny)] .> 0.0) && break
				α /= 2
			end
            for k = 1:100
                (norm(residual(unpack_vars(w + α*Δ)..., κ, θ)) < norm(r, Inf)) && break
                α /= 2
            end
            w = w + α * Δ
            println(
                "i ", i,
                " j ", j,
				" α ", scn(α),
				" κ ", scn(κ),
                " r ", scn(norm(residual(unpack_vars(w)..., κ, θ), Inf), digits=0, exp_digits=2),
                " br ", scn(norm(bilinear_residual(unpack_vars(w)..., κ), Inf), digits=0, exp_digits=2),
                " Δ ", scn(norm(Δ, Inf), digits=0, exp_digits=2),
                )
        end
        κ = max(btol, κ/10)
    end
    return unpack_vars(w)..., iter
end

function sensitivity(w, θ)
	H = FiniteDiff.finite_difference_jacobian(w -> residual(unpack_vars(w)..., 0.0, θ), w)
	S = FiniteDiff.finite_difference_jacobian(θ -> residual(unpack_vars(w)..., 0.0, θ), θ)
	return -H \ S
end
S = FiniteDiff.finite_difference_jacobian(θ -> residual(unpack_vars(w)..., 0.0, θ), θ)

x = zeros(nx)
y = ones(ny)
z = ones(ny)
θ = 0.2 * ones(nx + ny)
xs0, ys0, zs0, iter = easy_solver(x, y, z, θ, rtol=1e-5, btol=1e-5)
iter
xs0
ys0
zs0


x = zeros(nx)
y = ones(ny)
z = ones(ny)
θ = 0.2 * ones(nx + ny)
Δθ = 10*[0.03 * ones(nx); -0.01 * ones(ny)]
xs, ys, zs, iter = easy_solver(x, y, z, θ + Δθ)
iter



x = zeros(nx)
y = ones(ny)
z = ones(ny)
ws0 = pack_vars(xs0, ys0, zs0)
dwdθ = sensitivity(ws0, θ)
Δws = dwdθ * Δθ
ws0 + Δws
xs1, ys1, zs1 = unpack_vars(ws0 + Δws)
xs, ys, zs, iter = easy_solver(xs1, ys1, zs1, θ + Δθ)
iter
xs, ys, zs, iter = easy_solver(x, y, z, θ + Δθ)
iter
xs, ys, zs, iter = easy_solver(xs0, ys0, zs0, θ + Δθ)
iter
xs
ys
zs




scatter(log.(abs.(residual(x, y, z, 0.0, θ + Δθ))), color=:black, markersize=8.0, label="init")
scatter!(log.(abs.(residual(xs, ys, zs, 0.0, θ + Δθ))), color=:red, markersize=8.0, label="solution")
scatter!(log.(abs.(residual(xs0, ys0, zs0, 0.0, θ + Δθ))), color=:blue, markersize=8.0, label="old_solution")
scatter!(log.(abs.(residual(xs1, ys1, zs1, 0.0, θ + Δθ))), color=:green, markersize=8.0, label="sensitivity_solution")
