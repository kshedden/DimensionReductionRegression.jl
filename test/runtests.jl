using DimensionReductionRegression
using Test
using Random
using LinearAlgebra
using Statistics
using StatsBase
using RCall
using StableRNGs
using Distributions

function canonical_angles(A, B)
    A, _, _ = svd(A)
    B, _, _ = svd(B)
    _, s, _ = svd(A' * B)
    @assert maximum(abs, s) < 1 + 1e-10
    s = clamp.(s, -1, 1)
    return acos.(s)
end

function gendat_linear(n, p, rng)

    X = randn(rng, n, p)
    for j in 1:p
        X[:, j] .-= mean(X[:, j])
    end
    y = X[:, 1] - X[:, 2] + 0.5*randn(rng, n)
    ii = sortperm(y)
    X = X[ii, :]
    y = y[ii]

    return X, y
end

function gendat_quadratic(n, p, rng; xsd=ones(p))

    X = randn(rng, n, p)
    X = X * Diagonal(xsd)
    for j in 1:p
        X[:, j] .-= mean(X[:, j])
    end
    lp1 = X[:, 1] - X[:, 2]
    lp2 = X[:, 3] + X[:,4]
    y = lp1 .* (1 .+ lp2) + 0.5*randn(rng, n)
    ii = sortperm(y)
    X = X[ii, :]
    y = y[ii]

    return X, y
end

include("Aqua.jl")
include("cume.jl")
include("opg.jl")
include("sir.jl")
include("phd.jl")
include("save.jl")
include("core.jl")
