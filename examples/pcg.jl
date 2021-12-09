using Random
using LinearAlgebra, Krylov
using Plots
using RandomizedLasso
const RL = RandomizedLasso


## Compare "best" vs random preconditioner on random example
# Data
n, r = 1000, 500
A = randn(n, r)
A = A*A'
μ = 1e-2
xtrue = randn(n)
b = A*xtrue
D, V = eigen(A)

function true_preconditioner(k, D, V, μ)
    return (D[k+1] + μ)*V*Diagonal(1.0 ./ (D .+ μ))*V' + (I - V*V')
end
function cg_iters(A, b, P)
    _, stats = cg(A, b; history=true, M = P)
    !stats.solved && @warn "Did not correctly solve CG!!!"
    return length(stats.residuals)
end

nys_iters = zeros(length(ks))
true_iters = zeros(length(ks))
nopc_iters = ones(length(ks)) * cg_iters(A, b, I)
for (ind, (k, r)) in enumerate(zip(ks, rs))
    Anys = RL.NystromApprox(A, k, r)
    P = RL.RandomizedNystromPreconditionerInverse(Anys, μ)
    nys_iters[ind] = cg_iters(A, b, P)
    true_iters[ind] = cg_iters(A, b, true_preconditioner(k, D, V, μ))
    r % 100 == 0 && @info "Finished with r = $r"
end

plt_sketch_error = plot(ks[1:60], 
    [nys_iters[1:60], true_iters[1:60], nopc_iters[1:60]], 
    dpi=300,
    lw=3,
    label=["Nystrom Preconditioner" "True Preconditioner" "No Preconditioner"],
    ylabel="CG Iters",
    xlabel="Rank k",
    title="Convergence vs Preconditioner Rank",
    legend=:left
)
savefig(plt_sketch_error, joinpath(@__DIR__, "figs/cg_iters.pdf"))


## Real Dataset
using CSV, DataFrames
file = CSV.read("/Users/theodiamandis/Downloads/file7b5323e77330.csv", DataFrame)
M = Matrix(file)
b = M[:, 1]
A = M[:, 2:end]
m, n = size(A)
b = 1/m * A'*b
A = 1/m * A' * A

μ = 1e-3
_, stats = cg(A+μ*I, b; history=true)
npc_res = stats.residuals
nys_res = Vector{Float64}[]
rs = [10, 50, 100, 250, 500, 1000]
for r in rs
    k = Int(round(.9r))
    Anys = RL.NystromApprox(A, k, r)
    P = RL.RandomizedNystromPreconditionerInverse(Anys, μ)
    _, stats = cg(A+μ*I, b; history=true, M=P)
    push!(nys_res, stats.residuals)
    @info "Finished with r = $r"
end

plt_cg_real = plot(
    npc_res,
    dpi=300,
    lw=2,
    label="No Preconditioner",
    ylabel="residual",
    xlabel="iteration",
    title="Convergence of CG",
    legend=:topright,
    yaxis=:log
)
for (ind, y) in enumerate(nys_res)
    plot!(plt_cg_real, y, label="Nystrom, r = $(rs[ind])", lw=2)
end
savefig(plt_cg_real, joinpath(@__DIR__, "figs/cg_real_res.pdf"))
RL.deff(A, μ)