
"""
	SlicedAverageVarianceEstimation

A multi-index regression model fit using sliced average variance
estimation (SAVE).
"""
mutable struct SlicedAverageVarianceEstimation <: DimensionReductionModel

    "`y`: the response variable, sorted"
    y::AbstractVector

    "`X`: the explanatory variables, sorted to align with `y`"
    X::AbstractMatrix

    "`Xw`: the whitened explanatory variables"
    Xw::AbstractMatrix

    "`M`: the save kernel matrix"
	M::AbstractMatrix

    "`fw`: the proportion of the original data in each slice"
    fw::AbstractVector

    "`dirs`: a basis for the estimated effective dimension reduction (EDR) space"
    dirs::AbstractMatrix

    "`eigs`: the eigenvalues of the weighted covariance of slice means"
    eigs::AbstractVector

    "`trans`: map data coordinates to orthogonalized coordinates"
    trans::AbstractMatrix
end

function SlicedAverageVarianceEstimation(
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

    bd = slicer(y, nslice)

    dirs = zeros(0, 0)
    eigs = zeros(0)

	# Slice frequencies
	ns = diff(bd)
    fw = Float64.(ns)
    fw ./= sum(fw)

	M = save_kernel(y, Xw, bd, fw)

    # Actual number of slices, may differ from nslice
    h = length(bd) - 1

	return SlicedAverageVarianceEstimation(y, X, Xw, M, fw, dirs, eigs, trans)
end

function save_kernel(y::AbstractVector, X::AbstractMatrix, bd::AbstractVector, fw::AbstractVector)

    n, p = size(X)
    h = length(bd) - 1

    M = zeros(p, p)
	c = zeros(p, p)
    for i = 1:h
        c .= I(p) - cov(X[bd[i]:bd[i+1]-1, :])
        M .+= fw[i] * c * c
    end

    return M
end

function fit!(save::SlicedAverageVarianceEstimation; ndir=2)

	eg = eigen(save.M)

    save.eigs = eg.values[end:-1:1]
    dirs = eg.vectors[:, end:-1:1]
    save.dirs = dirs[:, 1:ndir]

    # Map back to the original coordinates
    save.dirs = save.trans \ save.dirs

	# Scale to unit length
	for j in 1:size(save.dirs, 2)
		save.dirs[:, j] ./= norm(save.dirs[:, j])
	end
end

"""
	fit(SlicedAverageVarianceEstimation, X, y; nslice, ndir)

Use Sliced Average Variance Estimation (SAVE) to estimate the effective dimension reduction (EDR) space.

'y' must be sorted before calling 'fit'.
"""
function fit(::Type{SlicedAverageVarianceEstimation}, X, y; nslice = max(8, size(X, 2) + 3), ndir = 2)
	if !issorted(y)
		error("y must be sorted")
	end
    save = SlicedAverageVarianceEstimation(y, X, nslice)
    fit!(save; ndir = ndir)
    return save
end

function coef(r::SlicedAverageVarianceEstimation)
    return r.dirs
end
