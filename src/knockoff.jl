using LinearAlgebra, Statistics, StatsBase, Random

"""
    pvaagg_call(pvals; alpha)

Return the estimated dimension using p-values `pvals` such that the probability 
of overstating the dimension is bounded by `alpha`.  `pvals` is a m x d array
of p-values from m independent knockoff augmentations, considering dimensions
up to d.
"""
function pvagg_call(pvals::Matrix{T}; alpha = 0.05)::Integer where {T<:Real}

    pv = copy(pvals)
    for i = 1:size(pv, 1)
        for j = 1:size(pv, 2)
            if pv[i, j] > alpha
                pv[i, j+1:end] .= 1
                break
            end
        end
    end

    apv = zeros(size(pvals, 2))
    for j = 1:size(pvals, 2)
        u = [x for x in pv[:, j] if x < 1]
        sort!(u)
        u = [v * length(u) / j for (j, v) in enumerate(u)]
        apv[j] = length(u) > 0 ? minimum(u) : 1
    end

    if apv[1] > alpha
        return 0
    else
        for j in eachindex(apv)
            if (apv[j] <= alpha) && ((j == length(apv)) || (apv[j+1] > alpha))
                return j
            end
        end
    end
    return length(apv)
end

"""
    pvagg(pvals, dim)

Returns a p-value for the null hypothesis that the dimension is equal to
`dim`, with the alternative being that the dimension is greater than 
`dim`.
"""
function pvagg(pvals::Matrix{T}, dim::Integer) where {T<:Real}

    maxdim = size(pvals, 2)

    if pvagg_call(pvals; alpha = 0.0) > dim
        return 0.0
    end

    # Check the endpoints
    a1, a2 = 0.0, 1.0
    d1 = pvagg_call(pvals; alpha = a1)
    if d1 > dim
        println("Warning, estimated dimension at alpha = 0 is $(d1) > $(dim) = dim")
    end
    d2 = pvagg_call(pvals; alpha = a2)
    if (d2 <= dim) && (dim < maxdim)
        println("Warning, estimated dimension at alpha = 1 is $(d2) <= $(dim) = dim")
    end

    while a2 - a1 > 0.001
        aa = (a1 + a2) / 2
        if pvagg_call(pvals; alpha = aa) > dim
            a2 = aa
        else
            a1 = aa
        end
    end

    return (a1 + a2) / 2
end

mutable struct KnockoffResults

    """
    `Pvalues` is the p x r array of p-values, where p is the number of variables and r is
    the maximum possible dimension.
    """
    Pvalues::Vector{Float64}

    RawPvalues::Matrix{Float64}

    """
    `Stat` is the p x r array of test statistics, which follow beta distributions under 
    the null.
    """
    Stat::Matrix{Float64}

    """
    `alpha` is used to control the probability that the estimated dimension
    exceeds the actual dimension.
    """
    alpha::Float64

    """
    `DimensionEstimate` is the estimated dimension, with the probability
    that the actual dimension is greater than this value being controlled
    at `alpha`.
    """
    DimensionEstimate::Integer

end

"""
    knockoff_test(y, x, drm; nslice=20, ndir=2)

Use a knockoff approach to estimate the dimension of the effective dimension reduction 
(EDR) space.

`drm` is a function implementing a dimension reduction method, e.g. 
`(y, x)->sir(y, x; nslice=10, ndim=3)`.`
"""
function knockoff_test(y, x, drm; alpha = 0.05, maxdim = 3, nrep = 1000)
    n, p = size(x)
    pval = zeros(nrep, maxdim)
    stat = zeros(nrep, maxdim)
    xa = zeros(n, 2 * p)
    xa[:, 1:p] = x

    # Stabilization replications
    for j = 1:nrep

        # Append the knockoff columns
        ii = randperm(n)
        xa[:, p+1:end] = x[ii, :]

        ss = drm(y, xa)
        for k = 1:maxdim
            sq1 = sum(abs2, ss.dirs[1:p, k])
            sq2 = sum(abs2, ss.dirs[p+1:end, k])
            b = sq1 / (sq1 + sq2)
            pval[j, k] = 1 - cdf(Beta((p - k + 1) / 2, p / 2), b)
            stat[j, k] = b
        end
    end

    de = pvagg_call(pval; alpha = alpha)
    pv = [pvagg(pval, j) for j = 1:maxdim]

    return KnockoffResults(pv, pval, stat, alpha, de)
end
