using Random
using LinearAlgebra
using Plots
using CSV, DataFrames
using RandomizedLasso
const RL = RandomizedLasso


## Load data
file = CSV.read("/Users/theodiamandis/Downloads/file7b5323e77330.csv", DataFrame)
M = Matrix(file)
b = M[:, 1]
A = M[:, 2:end]
m, n = size(A)
λmax = maximum(1/m*A'*b)
λ = 0.01*λmax

## Solve ADMM
prob = RL.LassoADMMProblem(A, b, λ)

RL.reset!(prob)
res_npc = RL.solve!(
    prob; 
    print_iter=10, 
    precondition=false, 
    max_iters=200, 
    tol=1e-5,
    logging=true
)

RL.reset!(prob)
res = RL.solve!(
    prob; 
    print_iter=10, 
    precondition=true, 
    max_iters=200, 
    tol=1e-5,
    logging=true
)

## Plotting
log_npc = res_npc.log
log = res.log


function make_plot(y1, y2, yaxis_label; yaxis=:log)
    plt = plot(
        log_npc.iter_time,
        y1,
        dpi=300,
        lw=3,
        label="No Preconditioner",
        ylabel=yaxis_label,
        xlabel="Wall clock time (s)",
        title="$yaxis_label Convergence",
        legend=:topright,
        yaxis=yaxis
    )
    plot!(plt, log.iter_time, y2, label="Nystrom", lw=3)
    return plt
end

plt_rp = make_plot(log_npc.rp, log.rp, "Primal Residual")
plt_rd = make_plot(log_npc.rd, log.rd, "Dual Residual")
plt_dual_gap = make_plot(log_npc.dual_gap, log.dual_gap, "Duality Gap")
plt_obj_val =make_plot(log_npc.obj_val, log.obj_val, "Objective Value"; yaxis=:none)
savefig(plt_rp, joinpath(@__DIR__, "figs/rp.pdf"))
savefig(plt_rd, joinpath(@__DIR__, "figs/rd.pdf"))
savefig(plt_dual_gap, joinpath(@__DIR__, "figs/dg.pdf"))
savefig(plt_obj_val, joinpath(@__DIR__, "figs/obj.pdf"))