
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

    "`nslice`: the number of slices"
    nslice::Int

    "`M`: the save kernel matrix"
	M::AbstractMatrix

	"`A`: the within-slice covariance matrices"
	A::Vector{AbstractMatrix}

    "`fw`: the proportion of the original data in each slice"
    fw::AbstractVector

    "`dirs`: a basis for the estimated effective dimension reduction (EDR) space"
    dirs::AbstractMatrix

    "`eigv`: the eigenvectors of the weighted covariance of slice means"
    eigv::AbstractMatrix

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
    y = copy(y)
    X = center(X)
    Xw, trans = whiten(X)

    bd = slicer(y, nslice)
	nslice = length(bd) - 1

	# Storage to be filled in during fit.
    dirs = zeros(0, 0)
    eigs = zeros(0)
    eigv = zeros(0, 0)

	# Slice frequencies
	ns = diff(bd)
    fw = Float64.(ns)
    fw ./= sum(fw)

	A, M = save_kernel(y, Xw, bd, fw)

    # Actual number of slices, may differ from nslice
    h = length(bd) - 1

	return SlicedAverageVarianceEstimation(y, X, Xw, nslice, M, A, fw, dirs, eigv, eigs, trans)
end

function save_kernel(y::AbstractVector, X::AbstractMatrix, bd::AbstractVector, fw::AbstractVector)

    n, p = size(X)
    h = length(bd) - 1

	# Number of observations per slice
	nw = n * fw

	A = Vector{AbstractMatrix}()
    M = zeros(p, p)
    for i = 1:h
        c = I(p) - cov(X[bd[i]:bd[i+1]-1, :], corrected=false)
        push!(A, sqrt(nw[i]) * c)
        M .+= nw[i] * c * c
    end

	M ./= n
	for j in eachindex(A)
		A[j] ./= sqrt(n)
	end

    return A, M
end

function fit!(save::SlicedAverageVarianceEstimation; ndir=2)

	eg = eigen(Symmetric(save.M))

    save.eigs = eg.values[end:-1:1]
    save.eigv = eg.vectors[:, end:-1:1]#[:, 1:ndir]
    save.dirs = eg.vectors[:, end:-1:1][:, 1:ndir]

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

function dimension_test(save::SlicedAverageVarianceEstimation)

	(; X, A, dirs, eigs, eigv) = save
	ndirs = size(dirs, 2)
	h = save.nslice
	p = length(eigs)
	n = length(save.y)

	# Test statistic based on normal and general theory
	nstat = zeros(ndirs)
	gstat = zeros(ndirs)

	# Degrees of freedom based on normal and general theory
	ndf = zeros(ndirs)
	gdf = zeros(ndirs)

	# P-values based on normal and general theory
	npv = zeros(ndirs)
	gpv = zeros(ndirs)

    qrx = qr(X)
	Z = sqrt(n) * Matrix(qrx.Q)

	# We need to resolve the sign ambiguity here.
	R = qrx.R
	for i in 1:size(R, 1)
		if R[i, i] < 0
			Z[:, i] *= -1
		end
	end

	for i in 0:ndirs-1

		E = eigv[:, i+1:end]
		H = Z * E
		ZH = zeros(n, (p-i)*(p-i))

		# Normal theory test statistics
		for j in 1:h
			nstat[i+1] += sum((E' * A[j] * E) .^ 2) * n / 2
		end

		# Normal theory degrees of freedom and p-value
		ndf[i + 1] = (h - 1) * (p - i) * (p - i + 1) / 2
		npv[i + 1] = 1 - cdf(Chisq(ndf[i+1]), nstat[i+1])

		# General theory test
		for j in 1:n
			ZH[j, :] = vec(H[j, :] * H[j, :]')
		end
		S = cov(ZH) / 2
        gdf[i + 1] = (h - 1) * sum(diag(S))^2 / sum(S.^2)
        gstat[i + 1] = nstat[i + 1] * sum(diag(S)) / sum(S.^2)
        gpv[i + 1] = 1 - cdf(Chisq(gdf[i + 1]), gstat[i + 1])
	end

    return (NormalPvals = npv, NormalStat = nstat, NormalDF = ndf, GeneralPvals = gpv, GeneralStat = gstat, GeneralDF = gdf)
end
