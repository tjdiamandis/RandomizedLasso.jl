export NystromApprox, RandomizedNystromPreconditioner

struct NystromApprox{T}
    U::Matrix{T}
    Λ::Diagonal{T, Vector{T}}
end

# Constructs Â_nys in factored form
# Â_nys = (AΩ)(ΩᵀAΩ)^†(AΩ)^ᵀ = UΛUᵀ
# [Martinsson & Tropp, Algorithm 16]
function NystromApprox(A::Matrix{T}, k::Int, r::Int; check=false) where {T <: Real}
    check && check_psd(A)
    n = size(A, 1)

    Ω = randn(n, r)
    Ω .= Array(qr(Ω).Q)
    Y = A * Ω
    ν = sqrt(n)*eps(norm(Y))
    @. Y += ν * Ω
    B = Y / cholesky(Symmetric(Ω' * Y)).U
    U, Σ, _ = svd(B)
    Λ = Diagonal(max.(0, Σ.^2 .- ν)[1:k])

    return NystromApprox(U[:, 1:k], Λ)
end

LinearAlgebra.eigvals(Anys::NystromApprox) = Anys.Λ.diag

function LinearAlgebra.mul!(y, Anys::NystromApprox, x; cache=zeros(size(Anys.U, 2)))
    length(y) != length(x) || length(y) != size(Anys.U, 1) && error(DimensionMismatch())
    r = size(Anys.U, 2)
    @views mul!(cache[1:r], Anys.U', x)
    @views cache[1:r] .*= Anys.Λ.diag
    @views mul!(y, Anys.U, cache[1:r])
    return nothing
end

function LinearAlgebra.:*(Anys::NystromApprox, x::AbstractVector)
    y = Anys.U*x
    y .*= Anys.Λ
    return Anys.U'*y
end

# Doubles rank until the approximation is sufficiently good
function adaptive_nystrom_approx(A::Matrix{T}, r0::Int; tol=1e-6, check=false, q=10) where {T <: Real}
    check && check_psd(A)
    n = size(A, 1)
    cache = (
        v0=zeros(n),
        v=zeros(n),
        Anys_mul=zeros(n)
    )
    r = r0
    Enorm = Inf
    Anys = nothing
    while Enorm > tol && r < n
        k = Int(round(.9*r))
        Anys = NystromApprox(A, k, r; check=false)
        Enorm = estimate_norm_E(A, Anys; q=q, cache=cache)
        r = 2r
    end
    return Anys
end


# By prop 5.3, have that κ(P^{-1/2} * A * P^{-1/2}) ≤ (λᵣ + μ + ||E||)/μ
function estimate_norm_E(A, Anys; q=10, cache=nothing)
    n = size(A, 1)
    if !isnothing(cache)
        v0, v = cache.v0, cache.v
    else
        v0, v = zeros(n), zeros(n)
        cache = (Anys_mul=zeros(n),)
    end
    
    v0 .= randn(n)
    normalize!(v0)

    Ehat = Inf
    for _ in 1:q
        mul!(v, Anys, v0; cache=cache.Anys_mul)
        mul!(v, A, v0, 1.0, -1.0)
        Ehat = dot(v0, v)
        normalize!(v)
        v0 .= v
    end
    return Ehat
end

# theoretically, choose r = 2⌈1.5deff(µ)⌉ + 1.
function deff(A::AbstractMatrix, μ; check=false)
    check && check_psd(A)
    λ = eigvals(A)
    return sum(x->x/(x+μ), λ)
end

function check_psd(A)
    n = size(A, 1)
    psd_tol = sqrt(n)*eps(norm(A))
    !isposdef(A + psd_tol*I) && error(ArgumentError("A must be PSD"))
end


struct RandomizedNystromPreconditioner{T <: Real}
    A_nys::NystromApprox{T}
    λ::MVector{1, T}
    μ::MVector{1, T}
    cache::Vector{T}
    function RandomizedNystromPreconditioner(A_nys::NystromApprox{T}, μ::T) where {T <: Real}
        return new{T}(A_nys, SA[A_nys.Λ.diag[end]], SA[μ], zeros(size(A_nys.U, 2)))
    end
end
Base.eltype(P::RandomizedNystromPreconditioner{T}) where {T} = T

function LinearAlgebra.:\(P::RandomizedNystromPreconditioner{T}, x::Vector{T}) where {T <: Real} 
    λ = P.λ[1]
    μ = P.μ[1]
    # P⁻x = x - U*((λ + μ)*(Λ + μI)^-1 + I)*Uᵀ*x
    mul!(P.A_nys.cache, P.A_nys.U', x)
    @. P.A_nys.cache *= (P.λ + P.μ) * 1 / (Λ.diag + μ) + 1
    return x - U*P.A_nys.cache
end

function LinearAlgebra.ldiv!(y::Vector{T}, P::RandomizedNystromPreconditioner{T}, x::Vector{T}) where {T <: Real}
    λ = P.λ[1]
    μ = P.μ[1]
    length(y) != length(x) && error(DimensionMismatch())
    mul!(P.cache, P.A_nys.U', x)
    @. P.cache *= (P.λ + P.μ) * 1 / (P.A_nys.Λ.diag + P.μ) - 1
    mul!(y, P.A_nys.U, P.cache)
    @. y = x + y
    return nothing
end

function LinearAlgebra.ldiv!(P::RandomizedNystromPreconditioner{T}, x::Vector{T}) where {T <: Real} 
    λ = P.λ[1]
    μ = P.μ[1]
    mul!(P.cache, P.A_nys.U', x)
    @. P.cache *= (P.λ + P.μ) * 1 / (P.A_nys.Λ.diag + P.μ) + 1
    x .-= P.A_nys.U*P.cache
    return nothing
end

# Used for Krylov method
struct RandomizedNystromPreconditionerInverse{T <: Real}
    A_nys::NystromApprox{T}
    λ::MVector{1, T}
    μ::MVector{1, T}
    cache::Vector{T}
    function RandomizedNystromPreconditionerInverse(A_nys::NystromApprox{T}, μ::T) where {T <: Real}
        return new{T}(A_nys, SA[A_nys.Λ.diag[end]], SA[μ], zeros(size(A_nys.U, 2)))
    end
end
Base.eltype(P::RandomizedNystromPreconditionerInverse{T}) where {T} = T

function LinearAlgebra.mul!(y, P::RandomizedNystromPreconditionerInverse, x)
    length(y) != length(x) && error(DimensionMismatch())
    
    λ = P.λ[1]
    μ = P.μ[1]
    
    mul!(P.cache, P.A_nys.U', x)
    @. P.cache *= (P.λ + P.μ) / (P.A_nys.Λ.diag + P.μ) - 1
    mul!(y, P.A_nys.U, P.cache)
    @. y = x + y
end