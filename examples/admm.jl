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
RL.solve!(prob; print_iter=10, precondition=false, max_iters=200, tol=1e-4)

RL.reset!(prob)
RL.solve!(prob; print_iter=10, precondition=true, max_iters=200, tol=1e-4)