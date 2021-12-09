# RandomizedLasso

[![Build Status](https://github.com/tjdiamandis/RandomizedLasso.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tjdiamandis/RandomizedLasso.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/tjdiamandis/RandomizedLasso.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/tjdiamandis/RandomizedLasso.jl)

An L1 regression solver using ADMM and [Randomized Nystrom Preconditioning](https://arxiv.org/pdf/2110.02820).

**Currently a work-in-progress**

## Use
This package solves L1 regression problems of the form
```math
min     1/2m * ||Ax - b||² + λ||x||₁
```
where `A` is an `m x n` matrix, `b` is an `m` vector, and `λ` is a scalar parameter.

Problems are constructed as follows.
```julia
prob = LassoADMMProblem(A, b, λ)
```

Then these problems can be solved by calling the `solve!` function:
```julia
result = solve!(prob)
```
There are several optional keyword arguments:
- `relax::Bool=true`
    - Toggles the use of relaxation (see CITE)
- `logging::Bool=false`
    - If true, logs the objective value, duality gap, RMSE, and primal & dual residuals at each iteration.
- `precondition::Bool=true`
    - Toggles use of the preconditioner for conjugate gradients
- `tol=1e-6 `
    - Stopping tolerance (duality gap)
- `max_iters::Int=100 `
- `print_iter::Int=25` 


### Regularization path
If you are solving this problem with several `λ`'s, there is no need to reconstruct the problem. Instead, update `λ`:
```julia
update_λ!(prob, λ_new)
```
The problem retains the solution from the previous run, so it will warmstart the next run of the optimization procedure.


### Results
The solver will return a `LassoADMMResult` with the following fields:
- `obj_val`
- `sq_error`
    - Currently `1/2m * ||Ax - b||²`, but may change to RMSE
- `x`
- `dual_gap`
- `log`
    - The log always contains the problem setup time, preconditioning time, and solve time. If `logging=true`, it also contains information about the metrics at each iterate. The full structure is below:
```julia
struct LassoADMMLog{T <: AbstractFloat}
    dual_gap::Union{AbstractVector{T}, Nothing}
    obj_val::Union{AbstractVector{T}, Nothing}
    iter_time::Union{AbstractVector{T}, Nothing}
    rp::Union{AbstractVector{T}, Nothing}
    rd::Union{AbstractVector{T}, Nothing}
    setup_time::T
    precond_time::T
    solve_time::T
end
```


## References
TODO

## TODOs
[ ] Smarter scaling of problem data
[ ] Function for regularization path
[ ] Better parallelization