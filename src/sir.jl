using StatsBase, LinearAlgebra, Printf, Distributions

"""
    SIRResults

The result of sliced inverse regression.
"""
struct SIRResults

    "`dirs`: a basis for the estimated effective dimension reduction (EDR) space"
    dirs::Array{Float64, 2}

    "`eigs`: the eigenvalues"
    eigs::Array{Float64}

    "`nslice`: the number of slices"
    nslice::Int

    "`n`: the sample size"
    n::Int

end

# Calculate means of blocks of consecutive rows of x.  The number of
# blocks is nslice
function _slice_means(x::Array{T, 2}, nslice::Integer)::Tuple{Array{Float64, 2}, Array{Int64}} where{T<:AbstractFloat}

    n, p = size(x)

    sm = zeros(Float64, nslice, p)
    ns = zeros(Int64, nslice)

    # Slice size is m+1 (first r slices) or m (remaining slices)
    m = div(n, nslice)
    r = rem(n, nslice)

    ii = 1
    for i in 1:nslice

	# The size of the current slice
        s = i <= r ? m + 1 : m

        for j in 1:p
            sm[i, j] = mean(x[ii:ii+s-1, j])
        end

        ns[i] = s
        ii += s

    end

    return sm, ns

end

# Center the columns of the array in-place
function _center!(x::Array{T, 2}) where {T<:AbstractFloat}
    for j in 1:size(x, 2)
        x[:, j] .= x[:, j] .- mean(x[:, j])
    end
end

# Whiten the array in-place
function _whiten!(x::Array{T, 2})::Array{Float64, 2} where {T<:AbstractFloat}
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
function sir_test(s::SIRResults)

    p = length(s.eigs)
    cs = zeros(p)
    pv = zeros(p)
    df = zeros(Int, p)

    for k in 0:p-1

       cs[k+1] = s.n * (p - k) * mean(s.eigs[k+1:end])
       df[k+1]  = (p - k) * (s.nslice - k - 1)
       pv[k+1] = 1 - cdf(Chisq(df[k+1]), cs[k + 1])

    end

    return tuple(pv, cs, df)

end


"""
    sir(y, x; nslice=20, ndir=2)

Use Sliced Inverse Regression (SIR) to estimate the effective dimension reduction (EDR) space.
"""
function sir(y::Array{S}, x::Array{T, 2}; nslice::Integer=20,
             ndir::Integer=2)::SIRResults where {S<:Real, T<:AbstractFloat}

    # Dimensions of the problem
    n, p = size(x)

    # Transform to orthogonal coordinates
    x = copy(x)
    _center!(x)
    cxu = _whiten!(x)

    # Sort the rows according to the values of y
    ii = sortperm(y)
    x .= x[ii, :]

    # Estimate E[X | Y]
    sm, ns = _slice_means(x, nslice)
    w = Array{Float64}(ns)
    w .= w / sum(w)

    # Get the SIR directions
    cx = StatsBase.cov(sm, fweights(w); corrected=false)
    eg = eigen(cx)
    eigs = eg.values[end:-1:1]
    dirs = eg.vectors[:, end:-1:1]
    dirs = dirs[:, 1:ndir]

    # Map back to the original coordinates
    dirs .= cxu \ dirs

    return SIRResults(dirs, eigs, nslice, n)

end
