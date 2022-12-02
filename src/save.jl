
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

	bd::AbstractVector

	"`slice_assignments`: the slice indicator for each observation, aligns
	with data supplied by user"
	slice_assignments::AbstractVector
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
	sa = expand_slice_bounds(bd, length(y))
	nslice = length(bd) - 1

	# Storage to be filled in during fit.
	dirs = zeros(0, 0)
	eigs = zeros(0)
	eigv = zeros(0, 0)

	# Slice frequencies
	ns = diff(bd)
	fw = Float64.(ns)
	fw ./= sum(fw)

	A, M = save_kernel(Xw, bd, fw)

	# Actual number of slices, may differ from nslice
	h = length(bd) - 1

	return SlicedAverageVarianceEstimation(
		y,
		X,
		Xw,
		nslice,
		M,
		A,
		fw,
		dirs,
		eigv,
		eigs,
		trans,
		bd,
		sa,
	)
end

function save_kernel(X::AbstractMatrix, bd::AbstractVector, fw::AbstractVector)
	n, p = size(X)
	h = length(bd) - 1

	# Number of observations per slice
	nw = n * fw

	A = Vector{AbstractMatrix}()
	M = zeros(p, p)
	for i = 1:h
		c = I(p) - cov(X[bd[i]:bd[i+1]-1, :], corrected = false)
		push!(A, sqrt(nw[i]) * c)
		M .+= nw[i] * c * c
	end

	M ./= n
	for j in eachindex(A)
		A[j] ./= sqrt(n)
	end

	return A, M
end

function fit!(save::SlicedAverageVarianceEstimation; ndir = 2)

	eg = eigen(Symmetric(save.M))

	save.eigs = eg.values[end:-1:1]
	save.eigv = eg.vectors[:, end:-1:1]#[:, 1:ndir]
	save.dirs = eg.vectors[:, end:-1:1][:, 1:ndir]

	# Map back to the original coordinates
	save.dirs = save.trans \ save.dirs

	# Scale to unit length
	for j = 1:size(save.dirs, 2)
		save.dirs[:, j] ./= norm(save.dirs[:, j])
	end
end

"""
	fit(SlicedAverageVarianceEstimation, X, y; nslice, ndir)

Use Sliced Average Variance Estimation (SAVE) to estimate the effective dimension reduction (EDR) space.

'y' must be sorted before calling 'fit'.
"""
function fit(
	::Type{SlicedAverageVarianceEstimation},
	X,
	y;
	nslice = max(8, size(X, 2) + 3),
	ndir = 2,
)
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

	(; X, Xw, A, dirs, eigs, eigv) = save
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

	for i = 0:ndirs-1

		E = eigv[:, i+1:end]
		H = Xw * E
		ZH = zeros(n, (p - i) * (p - i))

		# Normal theory test statistics
		for j = 1:h
			nstat[i+1] += sum((E' * A[j] * E) .^ 2) * n / 2
		end

		# Normal theory degrees of freedom and p-value
		ndf[i+1] = (h - 1) * (p - i) * (p - i + 1) / 2
		npv[i+1] = 1 - cdf(Chisq(ndf[i+1]), nstat[i+1])

		# General theory test
		for j = 1:n
			ZH[j, :] = vec(H[j, :] * H[j, :]')
		end
		S = cov(ZH) / 2
		gdf[i+1] = (h - 1) * sum(diag(S))^2 / sum(S .^ 2)
		gstat[i+1] = nstat[i+1] * sum(diag(S)) / sum(S .^ 2)
		gpv[i+1] = 1 - cdf(Chisq(gdf[i+1]), gstat[i+1])
	end

	return (
		NormalPvals = npv,
		NormalStat = nstat,
		NormalDegf = ndf,
		GeneralPvals = gpv,
		GeneralStat = gstat,
		GeneralDegf = gdf,
	)
end

function slice_covs(X, bd)

	p = size(X, 2)
	h = length(bd) - 1
	sc = zeros(p, p, h)

	for i = 1:h
		sc[:, :, i] = cov(X[bd[i]:bd[i+1]-1, :], corrected = false)
	end

	return sc
end

"""
	coordinate_test(sir::SlicedAverageVarianceEstimation, Hyp, ndir)

Reference:
Yu, Zhu, Wen. On model-free conditional coordinate tests for regressions.
Journal of Multivariate Analysis 109 (2012), 61-67.
https://web.mst.edu/~wenx/papers/zhouzhuwen.pdf
"""
function coordinate_test(save::SlicedAverageVarianceEstimation, Hyp, ndir)

	(; y, X, M, eigs, eigv, trans, fw, bd, slice_assignments, trans, nslice) = save

	r = size(Hyp, 2)
	n, p = size(X)

	@assert size(Hyp, 1) == p

	# cov(X)
	Sigma = trans' * trans
	Sigma_sqrti = ssqrti(Symmetric(Sigma))

	# Calculate the test statistic
	P = eigv[:, 1:ndir] * eigv[:, 1:ndir]'
	J = Symmetric(Hyp' * (Sigma \ Hyp))
	K = ssqrti(J)
	H = Sigma_sqrti * Hyp * K
	T1 = n * tr(H' * P * H)

	U = slice_means(y, X, bd) * Diagonal(fw)
	eg = eigen(Symmetric(Sigma))
	c = eg.values
	C = Diagonal(c)
	P1 = eg.vectors
	C1 = getC1(c)
	F = eigv[:, 1:ndir] * Diagonal(1 ./ eigs[1:ndir]) * eigv[:, 1:ndir]'

	V = slice_covs(X, bd)
	for j in eachindex(fw)
		V[:, :, j] .*= fw[j]
	end

	Lsir = U * Diagonal(1 ./ fw) * U'
	Lsave = 2 * Lsir - Sigma
	g1 = zeros(p, p)
	g2 = zeros(p, p)
	g3 = zeros(p, p)
	g4 = zeros(p, p)
	for j = 1:nslice
		v = V[:, :, j]
		u = U[:, j]
		f = fw[j]
		g1 .= v * (Sigma \ v) / f
		g2 .= v * (Sigma \ u) * u' / f^2
		g3 .= u * (u' * (Sigma \ v)) / f^2
		g4 .= (u' * (Sigma \ u)) * u * u' / f^3
		Lsave .+= g1 - g2 - g3 + g4
	end

	PW = H * H'
	QW = I(p) - PW
	Mc = QW * M * QW
	eg = eigen(Symmetric(Mc))
	xi = eg.vectors[:, end:-1:1]
	Pc = xi[:, 1:ndir] * xi[:, 1:ndir]'
	eg = eigen(Symmetric(P - Pc))
	T2 = n * sum(abs2, P - Pc)

	Mstarstarsave = zeros(p, p)
	Lstarstarsir = zeros(p, p)
	Lstarstarsave = zeros(p, p)
	Gammastarstar = zeros(p, p)
	Ustarstar = zeros(p, nslice)
	Vstarstar = zeros(p, p, nslice)
	SigmaStar = zeros(p, p)
	A = zeros(r, p)
	B = zeros(p, p)
	R = zeros(p, p)
	Sri = ssqrti(Symmetric(Sigma))
	Omega1 = zeros(p * r, p * r)
	Omega2 = zeros(p * p, p * p)
	for i = 1:n

		Ustarstar .= X[i, :] * fw'
		Ustarstar[:, slice_assignments[i]] .+= X[i, :]

		for j = 1:nslice
			Vstarstar[:, :, j] = -fw[j] * Sigma
		end
		Vstarstar[:, :, slice_assignments[i]] .+= X[i, :] * X[i, :]'

		Lstarstarsir .= Ustarstar * Diagonal(1 ./ fw) * U'
		SigmaStar .= X[i, :] * X[i, :]' - Sigma

		Gammastarstar .= 0
		for j = 1:nslice
			f = fw[j]
			u = U[:, j]
			uss = Ustarstar[:, j]
			v = V[:, :, j]
			vss = Vstarstar[:, :, j]
			pstar = (slice_assignments[i] == j ? 1 : 0) - f
			g1 = -pstar * Sigma + vss * (Sigma \ v) / f + Sigma * (SigmaStar \ v) + vss
			g2 = vss * (Sigma \ u) * u' / f^2 + Sigma * (Sigma \ u) * u' / f + uss * u' / f
			g3 = (uss * u' * (Sigma \ v)) / f^2
			g4 = (u' * (Sigma \ u)) * uss * u' / f^3
			Gammastarstar .+= g1 - g2 - g3 + g4
		end

		# R = SigStar^{-1/2}
		R .= P1 * (C1 .* (P1' * SigmaStar * P1)) * P1'

		Lstarstarsave = 2 * Lstarstarsir - SigmaStar + Gammastarstar

		Mstarstarsave .= R * Lsave * Sri + Sri * Lstarstarsave * Sri
		A .= K * Hyp' * (R * P + Sri * Mstarstarsave * F)
		Omega1 .+= vec(A) * vec(A)'

		# Which of these is correct?
		#B .= (PW * Mstarstar + Sri * Hyp * (J \ Hyp') * R * Sri * M * Sri) * F
		B .= (PW * Mstarstarsave + Sri * Hyp * (J \ Hyp') * R * M) * F
		B .+= B'
		Omega2 .+= vec(B) * vec(B)'
	end

	Omega1 ./= n
	Omega2 ./= n

	stat1, degf1, pval1 = bx_pvalues(Omega1, T1)
	stat2, degf2, pval2 = bx_pvalues(Omega2, T2)

	return (
		Stat1 = T1,
		Pval1 = pval1,
		Degf1 = degf1,
		Stat2 = T2,
		Pval2 = pval2,
		Degf2 = degf2,
	)
end
