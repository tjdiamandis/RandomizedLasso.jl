
export NystromApprox, RandomizedNystromPreconditioner

struct NystromApprox{T}
    U::Matrix{T}
    Λ::Diagonal{T, Vector{T}}
end

# Constructs Â_nys in factored form
# Â_nys = (AΩ)(ΩᵀAΩ)^†(AΩ)^ᵀ = UΛ̂Uᵀ
# [Frangella et al., Algorithm 2.1]
function NystromApprox(A::Matrix{T}, r::Int) where {T <: Real}
    !issymmetric(A) && !isposdef(A + 1e-12I) && error(ArgumentError("A must be PSD"))

    n = size(A, 1)
    Ω = randn(n, r)
    Ω .= Array(qr(Ω).Q)
    Y = A * Ω
    ν = eps(norm(Y))
    @. Y += ν * Ω
    B = Y / cholesky(Symmetric(Ω' * Y)).U
    U, Σ, _ = svd(B)
    Λ = Diagonal(max.(0, Σ.^2 .- ν))
    return NystromApprox(U, Λ)
end


mutable struct RandomizedNystromPreconditioner{T <: Real}
    A_nys::NystromApprox{T}
    λ::T
    μ::T
    cache::Vector{T}
    function RandomizedNystromPreconditioner(A_nys::NystromApprox{T}, μ::T) where {T <: Real}
        return new{T}(A_nys, A_nys.Λ.diag[end], μ, zeros(size(A_nys.U, 2)))
    end
end

function LinearAlgebra.:\(P::RandomizedNystromPreconditioner{T}, x::Vector{T}) where {T <: Real} 
    # P⁻x = x - U*((λ + μ)*(Λ + μI)^-1 + I)*Uᵀ*x
    mul!(P.A_nys.cache, P.A_nys.U', x)
    @. P.A_nys.cache *= (P.λ + P.μ) * 1 / (Λ.diag + μ) + 1
    return x - U*P.A_nys.cache
end

function LinearAlgebra.ldiv!(y::Vector{T}, P::RandomizedNystromPreconditioner{T}, x::Vector{T}) where {T <: Real}
    length(y) != length(x) && error(DimensionMismatch())
    mul!(P.cache, P.A_nys.U', x)
    @. P.cache *= (P.λ + P.μ) * 1 / (P.A_nys.Λ.diag + P.μ) + 1
    mul!(y, P.A_nys.U, P.cache)
    @. y = x - y
    return nothing
end

function LinearAlgebra.ldiv!(P::RandomizedNystromPreconditioner{T}, x::Vector{T}) where {T <: Real} 
    mul!(P.cache, P.A_nys.U', x)
    @. P.cache *= (P.λ + P.μ) * 1 / (P.A_nys.Λ.diag + P.μ) + 1
    x .-= P.A_nys.U*P.cache
    return nothing
end
