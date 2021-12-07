
# Assume that m ≫ n, so we want to throw out A
struct LassoADMMProblemData{T}
    ATA::AbstractMatrix{T}          # n x n matrix
    ATb::AbstractVector{T}          # n vector
    bTb::T                          # scalar
    function LassoADMMProblemData(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T <: Real}
        return new{T}(A'*A, A'*b, sum(x->x^2, b))
    end
end

struct LassoADMMProblem{T}
    data::LassoADMMProblemData{T}
    λ::MVector{1,T}
    Ax::AbstractMatrix{T}       # matrix in x update
    bx::AbstractVector{T}       # vector in x update
    obj_val::MVector{1,T}       # objective (uses zk)
    sq_error::MVector{1,T}      # 0.5*||Ax - b||² (uses zk)
    dual_gap::MVector{1,T}      # duality gap
    xk::AbstractVector{T}       # primal var
    zk::AbstractVector{T}       # primal var
    uk::AbstractVector{T}       # dual var (ADMM)
    ρ::MVector{1,T}
    α::MVector{1,T}
    function LassoADMMProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, λ::T; ρ=1e-6, α=1.5) where {T <: Real}
        n = size(A, 2)
        data = LassoADMMProblemData(A, b)
        return new{T}(
            data,
            SA[λ],
            similar(data.ATA),
            similar(data.ATb),
            SA[zero(T)],
            SA[zero(T)],
            SA[zero(T)],
            zeros(n),
            zeros(n),
            zeros(n),
            SA[ρ],
            SA[α]
        )
    end
end
#λmax = maximum(ATb) => x⋆ = 0


struct LassoADMMLog{T <: AbstractFloat}
    dual_gap::Union{AbstractVector{T}, Nothing}
    obj_val::Union{AbstractVector{T}, Nothing}
    iter_time::Union{AbstractVector{T}, Nothing}
    setup_time::T
    precond_time::T
    solve_time::T
end

struct LassoADMMResult{T}
    obj_val::T                 # primal objective
    sq_error::T                # 0.5*||Ax - b||²
    x::AbstractVector{T}       # primal soln
    dual_gap::T                       # duality gap
    # vT::AbstractVector{T}    # dual certificate
    log::LassoADMMLog
end

# --- substeps ---
# P is a preconditioner
function update_x!(prob::LassoADMMProblem{T}, solver::S, P) where {T <: Real, S <: KrylovSolver}
    # TODO: perhaps should decrease the ϵ here?
    # TODO: If A is fat, want to use matrix inversion lemma:
    #   (I + ρAᵀA)⁻¹ = I - ρAᵀ(I + ρAAᵀ)⁻¹A
    #   NOTE: should also adjust preconditioner to use AAᵀ
    @. prob.bx = prob.data.ATb + prob.ρ[1] * (prob.zk - prob.uk)
    cg!(solver, prob.Ax, prob.bx; M=P)
    !issolved(solver) && error("CG failed")
    prob.xk .= solver.x
end

function soft_threshold(x::T, y::T, κ::T) where {T <: Real}
    return max(0, x + y - κ) - max(0, -x - y - κ)
end

function update_z!(prob::LassoADMMProblem{T}; relax=true, xhat=nothing) where {T <: Real}
    if relax
        prob.zk .= soft_threshold!.(xhat, prob.uk, prob.λ[1] / prob.ρ[1])
    else    
        prob.zk .= soft_threshold!.(prob.xk, prob.uk, prob.λ[1] / prob.ρ[1])
    end
end

function update_u!(prob::LassoADMMProblem{T}; relax=true, xhat=nothing) where {T <: Real}
    if relax
        @. prob.uk += xhat - prob.zk
    else
        @. prob.uk += prob.xk - prob.zk
    end
end

# updated_obj = true if 
#   1) prob.sq_error has been updated
#   2) cache.v has been updated
function dual_gap(prob::LassoADMMProblem{T}, cache; updated_obj=false) where {T}
    x = prob.zk
    v = cache.v
    if !updated_obj
        mul!(v, prob.data.ATA, x)
    end
    u = prob.data.ATb
    w = prob.data.bTb
    @. cache.v_u = v - u
    z = norm(cache.v_u, Inf)
    uTx = dot(u, x)

    if !updated_obj
        prob.sq_error = 0.5*(dot(x, v) - 2uTx + w)
        prob.obj_val[1] = prob.sq_error[1] + prob.λ[1] * norm(x, 1)
    end

    prob.dual_gap[1] = prob.λ[1]^2/z^2*prob.sq_error[1] + prob.obj_val[1] + 1/z * (uTx - w)

    return prob.dual_gap[1]
end

function sq_error(prob::LassoADMMProblem{T}, cache) where {T}
    x = prob.zk
    v = cache.v
    mul!(v, prob.data.ATA, x)

    u = prob.data.ATb
    uTx = dot(u, x)
    w = prob.data.bTb

    prob.sq_error[1] = 0.5*(dot(x, v) - 2uTx + w)
    prob.obj_val[1] = prob.sq_error[1] + prob.λ[1] * norm(x, 1)
    return prob.sq_error[1]
end


# --- main solver ---
function solve!(
    prob::LassoADMMProblem{T}; 
    relax::Bool=true,
    logging::Bool=false,
    precondition::Bool=true,
    tol=1e-6, 
    max_iters::Int=100, 
    print_iter::Int=25,
) where {T <: Real}
    setup_time_start = time_ns()
    @printf("Starting setup...")

    # --- parameters ---
    n = size(prob.data.ATA, 1)
    t = 1
    η = Inf
    r0 = n ÷ 100

    # --- enable multithreaded BLAS ---
    n_threads = Sys.CPU_THREADS
    BLAS.set_num_threads(n_threads ÷ 2)

    # TODO: scale problem data??
    

    # --- init cache and memory allocations ---
    ρ = prob.ρ[1]
    λ = prob.λ[1]
    α = prob.α[1]
    solver = CgSolver(n, n, typeof(prob.xk))
    if relax
        xhat = copy(prob.xk)                             # for relaxation
    end
    cache = (
        v=zeros(n),
        v_u=zeros(n)
    )
    prob.Ax .= prob.data.ATA
    prob.Ax[diagind(prob.Ax)] .+= ρ

    # --- Precondition ---
    if precondition
        @printf("\n\tPreconditioning...")
        precond_time_start = time_ns()
        ATA_nys = adaptive_nystrom_approx(prob.data.ATA, r0)
        P = RandomizedNystromPreconditionerInverse(ATA_nys, ρ)
        precond_time = (time_ns() - precond_time_start) / 1e9
        @printf("\n\tPreconditioned in %6.3fs", precond_time)
    else
        P = I
        precond_time = 0.0
    end

    # --- Logging ---
    if logging
        dual_gap_log = zeros(max_iters)
        obj_val_log = zeros(max_iters)
        iter_time_log = zeros(max_iters)
    else
        dual_gap_log = nothing
        obj_val_log = nothing
        iter_time_log = nothing
    end

    setup_time = (time_ns() - setup_time_start) / 1e9
    @printf("\nSetup in %6.3fs\n", setup_time)

    # --- Print Headers ---
    headers = ["Iteration", "Objective", "RMSE", "Dual Gap", "Time"]
    print_header(headers)


    # --------------------------------------------------------------------------
    # --------------------- ITERATIONS -----------------------------------------
    # --------------------------------------------------------------------------
    solve_time_start = time_ns()
    while t <= max_iters && η > tol

        # --- Update Iterates ---
        update_x!(prob, solver, P)
        if relax
            @. xhat = α * prob.xk + (1-α) * prob.zk
        end
        update_z!(prob; relax=relax, xhat=xhat)
        update_u!(prob; relax=relax, xhat=xhat)

        # --- Eval Termination Criterion
        obj_val = sq_error(prob, cache) + prob.λ[1] .* norm(prob.zk, 1)
        η = dual_gap(prob, cache; updated_obj=true)

        # --- Logging ---
        time_sec = (time_ns() - solve_time_start) / 1e9
        if logging
            dual_gap_log[t] = η
            obj_val_log[t] = obj_val
            iter_time_log[t] = time_sec
        end
        
        # --- Printing ---
        if t == 1 || t % print_iter == 0
            print_iter_func((
                string(t),
                prob.obj_val[1],
                prob.sq_error[1],
                prob.dual_gap[1],
                time_sec
            ))
        end

        t += 1
    end

    solve_time = (time_ns() - solve_time_start) / 1e9
    @printf("\nSolved in %6.3fs\n", solve_time)
    print_footer()


    # --- Construct Logs ---
    log = LassoADMMLog(
        dual_gap_log, obj_val_log, iter_time_log,
        setup_time, precond_time, solve_time
    )


    # --- Construct Solution ---
    res = LassoADMMResult(
        prob.obj_val[1],
        prob.sq_error[1],
        prob.xk,
        prob.dual_gap[1],
        log
    )

    return res, log

end