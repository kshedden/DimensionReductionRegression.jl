using StatsBase, LinearAlgebra, Printf, Distributions

"""
    SIRResults

The result of sliced inverse regression.
"""
struct SIRResults

    "`dirs`: a basis for the estimated effective dimension reduction (EDR) space"
    dirs::Array{Float64,2}

    "`eigs`: the eigenvalues"
    eigs::Array{Float64}

    "`nslice`: the number of slices"
    nslice::Int

    "`n`: the sample size"
    n::Int

end

# Return the slice boundaries for approximately 'nslice' slices.
function slicer(y::AbstractVector, nslice::Integer)

    # Minimum slice size
    m = div(length(y), nslice)

    bds = []
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
    if bds[end] == length(y)
        bds = bds[1:end-1]
    end

    bds = push!(bds, length(y) + 1)

    return bds
end

# Calculate means of blocks of consecutive rows of x.  The number of
# blocks is nslice
function _slice_means(
    y::AbstractVector,
    x::Array{T,2},
    nslice::Integer,
)::Tuple{Array{Float64,2},Array{Int64}} where {T<:AbstractFloat}

    n, p = size(x)
    bd = slicer(y, nslice)
    h = length(bd) - 1 # Number of slices

    sm = zeros(Float64, h, p)
    ns = zeros(Int64, h)

    for i = 1:h
        sm[i, :] = mean(x[bd[i]:bd[i+1]-1, :], dims = 1)
        ns[i] = bd[i+1] - bd[i]
    end

    return sm, ns
end

# Center the columns of the array in-place
function _center!(x::Matrix{T}) where {T<:AbstractFloat}
    for j = 1:size(x, 2)
        x[:, j] .-= mean(x[:, j])
    end
end

# Whiten the array in-place
function _whiten!(x::Matrix{T})::Matrix{T} where {T<:AbstractFloat}
    n = size(x)[1]
    c = x' * x / n
    r = cholesky(c)
    x .= x / r.U
    return r.U
end

"""
    sir_test(s)

Returns p-values and Chi-squared statistics for the null hypotheses
that only the largest k eigenvalues are non-null.
"""
function sir_test(s::SIRResults; maxdim::Int = -1)

    p = length(s.eigs)
    maxdim = maxdim < 0 ? p - 1 : maxdim
    cs = zeros(maxdim + 1)
    pv = zeros(maxdim + 1)
    df = zeros(Int, maxdim + 1)

    for k = 0:maxdim
        cs[k+1] = s.n * sum(s.eigs[k+1:end])
        df[k+1] = (p - k) * (s.nslice - k - 1)
        pv[k+1] = 1 - cdf(Chisq(df[k+1]), cs[k+1])
    end

    return tuple(pv, cs, df)
end


"""
    sir(y, x; nslice=20, ndir=2)

Use Sliced Inverse Regression (SIR) to estimate the effective dimension reduction (EDR) space.
"""
function sir(
    y::Vector{S},
    x::Matrix{T};
    nslice::Integer = 20,
    ndir::Integer = 2,
)::SIRResults where {S,T<:Real}

    # Dimensions of the problem
    n, p = size(x)

    # Sort the rows according to the values of y.  This also copies
    # x and y.
    ii = sortperm(y)
    x = x[ii, :]
    y = y[ii]

    # Transform to orthogonal coordinates
    _center!(x)
    cxu = _whiten!(x)

    # Estimate E[X | Y]
    sm, ns = _slice_means(y, x, nslice)
    w = Array{Float64}(ns)
    w ./= sum(w)

    # Get the SIR directions
    cx = StatsBase.cov(sm, fweights(w); corrected = false)
    eg = eigen(cx)
    eigs = eg.values[end:-1:1]
    dirs = eg.vectors[:, end:-1:1]
    dirs = dirs[:, 1:ndir]

    # Map back to the original coordinates
    dirs = cxu \ dirs

    return SIRResults(dirs, eigs, nslice, n)
end
