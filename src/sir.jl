using StatsBase, LinearAlgebra, Printf

abstract type DimensionReduction end

"""
    DimensionReductionEigen

The result of a dimension reduction regression analysis that involves
an eigendecomposition.
"""
struct DimensionReductionEigen <: DimensionReduction

    "`dirs`: a basis for the estimated effective dimension reduction (EDR) space"
    dirs::Array{Float64, 2}

    "`eigs`: the eigenvalues"
    eigs::Array{Float64}

end


# Calculate means of blocks of consecutive rows of x.  The number of
# blocks is n_slice
function _slice_means(x::Array{T, 2}, n_slice::Integer)::Tuple{Array{Float64, 2}, Array{Int64}} where{T<:AbstractFloat}

    n = size(x)[1]
    p = size(x)[2]

    sm = zeros(Float64, n_slice, p)
    ns = zeros(Int64, n_slice)

    # Slice size is m+1 (first r slices) or m (remaining slices)
    m = div(n, n_slice)
    r = rem(n, n_slice)

    ii = 1
    for i in 1:n_slice

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
    p = size(x)[2]
    for i in 1:p
        x[:, i] .= x[:, i] .- mean(x[:, i])
    end
end

# Whiten the array in-place
function _whiten!(x::Array{T, 2})::Array{Float64, 2} where {T<:AbstractFloat}
    n = size(x)[1]
    c = transpose(x) * x / n
    r = cholesky(c)
    x .= x/r.U
    r.U
end

"""
    sir(y, x; nslice=20, ndir=2)

Use Sliced Inverse Regression (SIR) to estimate the effective dimension reduction (EDR) space.
"""
function sir(y::Array{S}, x::Array{T, 2}; nslice::Integer=20,
             ndir::Integer=2)::DimensionReductionEigen where {S<:Real, T<:AbstractFloat}

    # Dimensions of the problem
    n = size(x)[1]
    p = size(x)[2]

    # Transform to orthogonal coordinates
    x = copy(x)
    _center!(x)
    cxu = _whiten!(x)

    # Sort the rows according to the values of y
    ii = sortperm(y)
    x .= x[ii, :]

    # Estimate E[X | Y]
    sm, ns = _slice_means(x, 20)
    w = convert(Array{Float64}, ns)
    w .= w / sum(w)

    # Get the SIR directions
    ex = StatsBase.cov(sm, fweights(w); corrected=false)
    eg = eigen(ex)
    eigs = eg.values[end:-1:1]
    dirs = eg.vectors[:, end:-1:1]
    dirs = dirs[:, 1:ndir]

    # Map back to the original coordinates
    dirs .= cxu\dirs

    return DimensionReductionEigen(dirs, eigs)

end
