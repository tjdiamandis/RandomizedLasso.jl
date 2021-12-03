
export NystromApprox, RandomizedNystromPreconditioner

struct NystromApprox{T}
    U::Matrix{T}
    Λ::Diagonal{T, Vector{T}}
end

# Constructs Â_nys in factored form
# Â_nys = (AΩ)(ΩᵀAΩ)^†(AΩ)^ᵀ = UΛUᵀ
# [Martinsson & Tropp, Algorithm 16]
function NystromApprox(A::Matrix{T}, k::Int, r::Int; check=false) where {T <: Real}
    n = size(A, 1)
    if check
        #TODO: reevaluate this tolerance chocie
        psd_tol = sqrt(n)*eps(norm(A))
        !isposdef(A + psd_tol*I) && error(ArgumentError("A must be PSD"))
    end

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
    mul!(cache, Anys.U', x)
    cache .= Anys.Λ .* cache
    mul!(y, Anys.U, cache)
    return nothing
end


struct RandomizedNystromPreconditioner{T <: Real}
    A_nys::NystromApprox{T}
    λ::SVector{1, T}
    μ::SVector{1, T}
    cache::Vector{T}
    function RandomizedNystromPreconditioner(A_nys::NystromApprox{T}, μ::T) where {T <: Real}
        return new{T}(A_nys, SA[A_nys.Λ.diag[end]], SA[μ], zeros(size(A_nys.U, 2)))
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
