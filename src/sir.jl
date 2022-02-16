using StatsBase, LinearAlgebra, Printf, Distributions

abstract type AbstractSlicedInverseRegression end

"""
    SlicedInverseRegression

A multi-index regression model fit using sliced inverse
regression (SIR).
"""
struct SlicedInverseRegression <: AbstractSlicedInverseRegression

    "`y`: the response variable, sorted"
    y::Vector{Float64}

    "`X`: the explanatory variables, sorted to align with `y`"
    X::Matrix{Float64}

    "`Xw`: the whitened explanatory variables"
    Xw::Matrix{Float64}

    "`sm`: the slice means (column-wise)"
    sm::Matrix{Float64}

    "`fw`: the proportion of the original data in each slice"
    fw::Vector{Float64}

    "`dirs`: a basis for the estimated effective dimension reduction (EDR) space"
    dirs::Matrix{Float64}

    "`eigs`: the eigenvalues of the weighted covariance of slice means"
    eigs::Array{Float64}

    "`nslice`: the number of slices"
    nslice::Int

    "`slice_assignments`: the slice indicator for each observation, aligns
    with data supplied by user"
    slice_assignments::Vector{Int}

    "`n`: the sample size"
    n::Int

end

function coef(r::SlicedInverseRegression)
    return r.dirs
end

# Return the slice boundaries for approximately 'nslice' slices.
function slicer(y::AbstractVector, nslice::Integer)

    # Minimum slice size
    m = div(length(y), nslice)

    bds = Int[]
    i = 1 # Beginning of current slice
    while i < length(y)
        push!(bds, i)

        # Nominal last position in slice
        j = i + m - 1

        # Scan forward to the end of a run of ties
        while j < length(y) && y[j+1] == y[j]
            j += 1
        end
        i = j + 1
    end

    # If the last slice is too small, merge it with the previous one.
    if bds[end] > length(y) - m / 2
        bds = bds[1:end-1]
    end

    # Append a sentinel
    bds = push!(bds, length(y) + 1)

    return bds
end

# Calculate means of blocks of consecutive rows of x.  The number of
# blocks is nslice
function _slice_means(
    y::AbstractVector,
    x::Matrix{T},
    nslice::Integer,
) where {T<:AbstractFloat}

    n, p = size(x)
    bd = slicer(y, nslice)
    h = length(bd) - 1 # Number of slices

    # Slice means and sample sizes
    sm = zeros(Float64, h, p)
    ns = zeros(Int64, h)

    for i = 1:h
        sm[i, :] = mean(x[bd[i]:bd[i+1]-1, :], dims = 1)
        ns[i] = bd[i+1] - bd[i]
    end

    return (sm, ns, bd)
end

# Center the columns of the array in-place
function center!(X::Matrix{T}) where {T<:AbstractFloat}
    for j = 1:size(X, 2)
        X[:, j] .-= mean(X[:, j])
    end
    return X
end

# Whiten the array X, which has already been centered.
function whiten(X::Matrix{T}) where {T<:AbstractFloat}
    n = size(X, 1)
    c = X' * X / n
    r = cholesky(c)
    Xw = X / r.U
    return tuple(Xw, r.U)
end

"""
    sir_test(s)

Returns p-values and Chi-squared statistics for the null hypotheses
that only the largest k eigenvalues are non-null.
"""
function sir_test(s::SlicedInverseRegression; maxdim::Int = -1)

    p = length(s.eigs)
    maxdim = maxdim < 0 ? min(p - 1, s.nslice - 2) : maxdim
    cs = zeros(maxdim + 1)
    pv = zeros(maxdim + 1)
    df = zeros(Int, maxdim + 1)

    for k = 0:maxdim
        cs[k+1] = s.n * sum(s.eigs[k+1:end])
        df[k+1] = (p - k) * (s.nslice - k - 1)
        pv[k+1] = 1 - cdf(Chisq(df[k+1]), cs[k+1])
    end

    return (Pvalues = pv, ChiSquare = cs, Degf = df)
end

# Convert the array of slice boundaries to an array of slice indicators.
function expand_slice_bounds(bd, n)
    z = zeros(Int, n)
    for i = 1:length(bd)-1
        z[bd[i]:bd[i+1]-1] .= i
    end
    return z
end

function fit(
    ::Type{M},
    y::Vector{S},
    X::Matrix{T};
    nslice::Integer = 20,
    ndir::Integer = 2,
)::SlicedInverseRegression where {S,T<:Real,M<:AbstractSlicedInverseRegression}

    # Dimensions of the problem
    n, p = size(X)

    # Sort the rows according to the values of y.  This also copies
    # X and y.
    ii = sortperm(y)
    X = X[ii, :]
    y = y[ii]

    # Transform to orthogonal coordinates
    center!(X)
    Xw, cxu = whiten(X)

    # Estimate E[X | Y]
    sm, ns, bd = _slice_means(y, Xw, nslice)
    bx = expand_slice_bounds(bd, length(y))
    fw = Array{Float64}(ns)
    fw ./= sum(fw)

    # Reorder the slice indicators so they reflect the original
    # order of the data as supplied.
    jj = sortperm(ii)
    bx = bx[jj]

    # Get the SIR directions
    cx = StatsBase.cov(sm, fweights(fw); corrected = false)
    eg = eigen(cx)
    eigs = eg.values[end:-1:1]
    dirs = eg.vectors[:, end:-1:1]
    dirs = dirs[:, 1:ndir]

    # Map back to the original coordinates
    dirs = cxu \ dirs

    return SlicedInverseRegression(y, X, Xw, sm, fw, dirs, eigs, nslice, bx, n)
end

"""
    sir(y, x; nslice=20, ndir=2)

Use Sliced Inverse Regression (SIR) to estimate the effective dimension reduction (EDR) space.
"""
function sir(y, X; nslice = 20, ndir = 2)
    return fit(SlicedInverseRegression, y, X; nslice, ndir)
end
