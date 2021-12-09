using Random
using LinearAlgebra, Krylov
using Plots
using RandomizedLasso
const RL = RandomizedLasso

## Example with random data
n, r = 1000, 500
μ = 1e-3
A = randn(n, r)
A = A*A' + μ*I
λA = sort(eigvals(A), by=x->-x)
plt_eigvals = plot(
    λA,
    dpi=300,
    lw=3,
    title="Eigenvalues of A",
    legend=false
)
savefig(plt_eigvals, joinpath(@__DIR__, "figs/eigvals.pdf"))
deff_A = RL.deff(A, μ)

rs = 10:10:1000
ks = Int.(round.(0.9 .* rs .- 1))
tail = zeros(length(ks))
for (ind, k) in enumerate(ks)
    tail[ind] = sum(λA[k+1:end])
end

bound = max.(1e-8, λA[10:10:end] + ks ./ (rs .- ks .- 1) .* tail)

errors = Float64[]
sizehint!(errors, length(rs))
for (r, k) in zip(rs, ks)
    Anys = RL.NystromApprox(A, k, r)
    E = maximum(svdvals(A - Anys.U*Anys.Λ*Anys.U'))
    push!(errors, E)
    r % 100 == 0 && @info "Finished with r = $r"
end

plt_sketch_error = plot(ks[1:70], 
    [errors[1:70], bound[1:70]], 
    yaxis=:log,
    dpi=300,
    lw=3,
    label=["Error" "Expectation Bound"],
    ylabel="||A - Anys||",
    xlabel="rank k",
    title="Error vs. r with k = 0.9r - 1"
)
savefig(plt_sketch_error, joinpath(@__DIR__, "figs/sketch_error.pdf"))