using StatsBase, LinearAlgebra, Printf, Distributions

abstract type DimensionReductionModel <: RegressionModel end

"""
	SlicedInverseRegression

A multi-index regression model fit using sliced inverse
regression (SIR).
"""
mutable struct SlicedInverseRegression <: DimensionReductionModel

    "`y`: the response variable, sorted"
    y::AbstractVector

    "`X`: the explanatory variables, sorted to align with `y`"
    X::AbstractMatrix

    "`Xw`: the whitened explanatory variables"
    Xw::AbstractMatrix

    "`sm`: the slice means (each column contains one slice mean)"
    sm::AbstractMatrix

    "`cx`: the covariance of the slice means"
    cx::AbstractMatrix

    "`fw`: the proportion of the original data in each slice"
    fw::AbstractVector

    "`dirs`: a basis for the estimated effective dimension reduction (EDR) space"
    dirs::AbstractMatrix

    "`eigs`: the eigenvalues of the weighted covariance of slice means"
    eigs::AbstractVector

    "`trans`: map data coordinates to orthogonalized coordinates"
    trans::AbstractMatrix

    "`nslice`: the number of slices"
    nslice::Int

    "`slice_assignments`: the slice indicator for each observation, aligns
    with data supplied by user"
    slice_assignments::AbstractVector

    "`n`: the sample size"
    n::Int
end

function SlicedInverseRegression(
    y::AbstractVector,
    X::AbstractMatrix,
    nslice::Int;
    slicer = slicer,
)
	@assert issorted(y)
    @assert length(y) == size(X, 1)
    n, p = size(X)

    # Transform to orthogonal coordinates
    center!(X)
    Xw, trans = whiten(X)

    sm = zeros(0, 0)
    fw = zeros(0)
    dirs = zeros(0, 0)
    eigs = zeros(0)

    bd = slicer(y, nslice)

    # Actual number of slices, may differ from nslice
    h = length(bd) - 1

    # Estimate E[X | Y]
    sm = slice_means(y, Xw, bd)
    sa = expand_slice_bounds(bd, length(y))

	# Slice frequencies
	ns = diff(bd)
    fw = Float64.(ns)
    fw ./= sum(fw)

    return SlicedInverseRegression(
        y,
        X,
        Xw,
        sm,
        zeros(0, 0),
        fw,
        dirs,
        eigs,
        trans,
        h,
        sa,
        n,
    )
end

function coef(r::SlicedInverseRegression)
    return r.dirs
end

# Find slice bounds, placing each distinct value of y into its own slice.
# This function assumes that y is sorted.  This matches the slice1 function
# in the R dr package.
function slice1(y, u)
    bds = Int[]
    for j in eachindex(u)
        ii = searchsortedfirst(y, u[j])
        push!(bds, ii)
    end
    push!(bds, length(y) + 1)
    return bds
end

# The main slicing function, matches slice2 in the R dr package.
function slice2(y, u, nslice)

    myfind = function (x, v)
        ii = findfirst(x .<= v)
        return ifelse(isnothing(ii), length(v), ii)
    end

    # Cumulative counts of distinct values
    bds = slice1(y, u)
    cty = cumsum(diff(bds))

    n = length(y)
    m = floor(n / nslice) # nominal number of obs per slice
    bds = Int[]
    jj, j = 0, 0
    while jj < n - 2
        jj += m
        j += 1
        s = myfind(jj, cty)
        jj = cty[s]
        push!(bds, s)
    end
    return vcat(1, 1 .+ cty[bds])
end

# Return the slice boundaries for approximately 'nslice' slices.
function slicer(y::AbstractVector, nslice::Integer)
    u = sort(unique(y))
    if length(u) > nslice
        return slice2(y, u, nslice)
    else
        return slice1(y, u)
    end
end

# Calculate means of blocks of consecutive rows of x.  The number of
# blocks is nslice
function slice_means(y::AbstractVector, X::AbstractMatrix, bd::AbstractVector)

    n, p = size(X)
	h = length(bd) - 1

    # Slice means and sample sizes
    sm = zeros(Float64, h, p)

    for i = 1:h
        sm[i, :] = mean(X[bd[i]:bd[i+1]-1, :], dims = 1)
    end

    return sm
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
function dimension_test(s::SlicedInverseRegression; maxdim::Int = -1)

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

function fit!(si::SlicedInverseRegression; ndir::Integer = 2)

    # Get the SIR directions
    cx = StatsBase.cov(si.sm, fweights(si.fw); corrected = false)
    eg = eigen(cx)
    si.eigs = eg.values[end:-1:1]
    dirs = eg.vectors[:, end:-1:1]
    si.dirs = dirs[:, 1:ndir]
    si.cx = cx

    # Map back to the original coordinates
    si.dirs = si.trans \ si.dirs

	# Scale to unit length
	for j in 1:size(si.dirs, 2)
		si.dirs[:, j] ./= norm(si.dirs[:, j])
	end
end

"""
	sir(y, x; nslice=20, ndir=2)

Use Sliced Inverse Regression (SIR) to estimate the effective dimension reduction (EDR) space.

'y' must be sorted before calling 'fit'.
"""
function fit(::Type{SlicedInverseRegression}, X, y; nslice = max(8, size(X, 2) + 3), ndir = 2)
	if !issorted(y)
		error("y must be sorted")
	end
    sm = SlicedInverseRegression(y, X, nslice)
    fit!(sm; ndir = ndir)
    return sm
end
