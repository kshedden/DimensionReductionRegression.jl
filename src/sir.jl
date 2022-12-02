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

	"`M`: the covariance of the decorrelated slice means"
	M::AbstractMatrix

	"`fw`: the proportion of the original data in each slice"
	fw::AbstractVector

	"`dirs`: a basis for the estimated effective dimension reduction (EDR) space"
	dirs::AbstractMatrix

	"`eigs`: the eigenvalues of the weighted covariance of slice means"
	eigs::AbstractVector

	"`eigv`: the eigenvectors of the weighted covariance of slice means"
	eigv::AbstractMatrix

	"`trans`: map data coordinates to orthogonalized coordinates"
	trans::AbstractMatrix

	bd::AbstractVector

	"`slice_assignments`: the slice indicator for each observation, aligns
	with data supplied by user"
	slice_assignments::AbstractVector

	"`nslice`: the number of slices"
	nslice::Int

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
	y = copy(y)
	X = center(X)
	Xw, trans = whiten(X)

	sm = zeros(0, 0)
	fw = zeros(0)
	dirs = zeros(0, 0)
	eigs = zeros(0)
	eigv = zeros(0, 0)

	bd = slicer(y, nslice)
	sa = expand_slice_bounds(bd, length(y))

	# Actual number of slices, may differ from nslice
	h = length(bd) - 1

	# Estimate E[X | Y]
	sm = slice_means(y, Xw, bd)

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
		eigv,
		trans,
		bd,
		sa,
		h,
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
# TODO y is not needed
function slice_means(y::AbstractVector, X::AbstractMatrix, bd::AbstractVector)

	n, p = size(X)
	h = length(bd) - 1

	# Slice means and sample sizes
	sm = zeros(Float64, p, h)

	for i = 1:h
		sm[:, i] = mean(X[bd[i]:bd[i+1]-1, :], dims = 1)
	end

	return sm
end

# Center the columns of the matrix X.
function center(X)
	X = copy(X)
	for j = 1:size(X, 2)
		X[:, j] .-= mean(X[:, j])
	end
	return X
end

# Whiten the array X, which has already been centered.
# When sym=true, the data are whitened using a symmetric
# square root.  sym should always be set to true as this
# is assumed by the coordinate tests.
function whiten(X; sym = true)
	n = size(X, 1)
	if sym
		qrx = qr(X)
		R = Matrix(qrx.R)
		S = R' * R / n
		T = ssqrti(Symmetric(S))
		W = X * T
		return W, inv(T)
	else
		# This is what the R dr package uses.
		qrx = qr(X)
		W, T = Matrix(qrx.Q), Matrix(qrx.R)
		k = sqrt(n)
		W *= k
		T /= k
		return W, T
	end
end

"""
	dimension_test(sir)

Returns p-values and Chi-squared statistics for the null hypotheses
that only the largest k eigenvalues are non-null.
"""
function dimension_test(sir::SlicedInverseRegression; maxdim::Int = -1)

	p = length(sir.eigs)
	maxdim = maxdim < 0 ? min(p - 1, sir.nslice - 2) : maxdim
	cs = zeros(maxdim + 1)
	pv = zeros(maxdim + 1)
	df = zeros(Int, maxdim + 1)

	for k = 0:maxdim
		cs[k+1] = sir.n * sum(sir.eigs[k+1:end])
		df[k+1] = (p - k) * (sir.nslice - k - 1)
		pv[k+1] = 1 - cdf(Chisq(df[k+1]), cs[k+1])
	end

	return (Pvals = pv, Stat = cs, Degf = df)
end

function ssqrt(A::Symmetric)
	eg = eigen(A)
	F = eg.vectors
	Q = sqrt.(eg.values)
	return F * Diagonal(Q) * F'
end

# Returns the inverse of the symmetric square root of A. 
function ssqrti(A::Symmetric)
	eg = eigen(A)
	F = eg.vectors
	Q = sqrt.(eg.values)
	return F * Diagonal(1 ./ Q) * F'
end

function getC1(c)
	p = length(c)
	C1 = zeros(p, p)
	for i = 1:p
		for j = 1:p
			if i == j
				C1[i, j] = -0.5 * c[i]^-1.5
			else
				C1[i, j] = (c[i]^-0.5 - c[j]^-0.5) / (c[i] - c[j])
			end
		end
	end
	return C1
end

# Use the Bentler-Xie approach to obtain approximate
# Chi^2 statistics from a weighted sum of Chi^2(1).
function bx_pvalues(Omega, T)
	eg = eigen(Symmetric(Omega))
	eigs = eg.values
	degf = sum(eigs)^2 / sum(eigs .^ 2)
	T *= degf / sum(eigs)
	pval = 1 - cdf(Chisq(degf), T)
	return T, degf, pval
end

function sim_pvalues(Omega, T; nrep = 10000)
	eg = eigen(Symmetric(Omega))
	eigs = eg.values
	p = length(eigs)

	pval = 0.0
	for i = 1:nrep
		x = rand(Chisq(1), p)
		if sum(x * eigs) > T
			pval += 1
		end
	end
	pval /= nrep
	return T, 0, pval
end

function ct_pvalues(Omega, T, pmethod)
	if pmethod == "bx"
		return bx_pvalues(Omega, T)
	elseif pmethod == "sim"
		return sim_pvalues(Omega, T)
	else
		error("Unknown p-value method")
	end
end

"""
	coordinate_test(sir::SlicedInverseRegression, Hyp, ndir)

Reference:
Yu, Zhu, Wen. On model-free conditional coordinate tests for regressions.
Journal of Multivariate Analysis 109 (2012), 61-67.
https://web.mst.edu/~wenx/papers/zhouzhuwen.pdf
"""
function coordinate_test(sir::SlicedInverseRegression, Hyp, ndir; pmethod = "bx")

	(; y, X, M, eigs, eigv, trans, fw, bd, slice_assignments, nslice) = sir

	r = size(Hyp, 2)
	n, p = size(X)

	@assert size(Hyp, 1) == p

	# cov(X)
	Sigma = trans' * trans
	Sigma_sqrti = ssqrti(Symmetric(Sigma))

	# Calculate the test statistic
	P = sir.eigv[:, 1:ndir] * sir.eigv[:, 1:ndir]'
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

	L = U * Diagonal(1 ./ fw) * U'
	PW = H * H'
	QW = I(p) - PW
	Mc = QW * M * QW
	eg = eigen(Symmetric(Mc))
	xi = eg.vectors[:, end:-1:1]
	Pc = xi[:, 1:ndir] * xi[:ndir]'
	eeigen(Symmetric(P - Pc))
	T2 = n * sum(abs2, P - Pc)

	Mstarstar = zeros(p, p)
	Lstarstar = zeros(p, p)
	Ustarstar = zeros(p, nslice)
	SigStar = zeros(p, p)
	A = zeros(r, p)
	B = zeros(p, p)
	R = zeros(p, p)
	Sri = ssqrti(Symmetric(Sigma))
	Omega1 = zeros(p * r, p * r)
	Omega2 = zeros(p * p, p * p)
	for i = 1:n

		Ustarstar .= X[i, :] * fw'
		Ustarstar[:, slice_assignments[i]] .+= X[i, :]

		Lstarstar .= Ustarstar * Diagonal(1 ./ fw) * U'

		# R = SigStar^{-1/2}
		SigStar .= X[i, :] * X[i, :]' - Sigma
		R .= P1 * (C1 .* (P1' * SigStar * P1)) * P1'

		Mstarstar .= R * L * Sri + Sri * Lstarstar * Sri
		A .= K * Hyp' * (R * P + Sri * Mstarstar * F)
		Omega1 .+= vec(A) * vec(A)'

		# Which of these is correct?
		#B .= (PW * Mstarstar + Sri * Hyp * (J \ Hyp') * R * Sri * M * Sri) * F
		B .= (PW * Mstarstar + Sri * Hyp * (J \ Hyp') * R * M) * F
		B .+= B'
		Omega2 .+= vec(B) * vec(B)'
	end

	Omega1 ./= n
	Omega2 ./= n

	stat1, degf1, pval1 = ct_pvalues(Omega1, T1, pmethod)
	stat2, degf2, pval2 = ct_pvalues(Omega2, T2, pmethod)

	return (
		Stat1 = T1,
		Pval1 = pval1,
		Degf1 = degf1,
		Stat2 = T2,
		Pval2 = pval2,
		Degf2 = degf2,
	)
end

# Convert the array of slice boundaries to an array of slice indicators.
function expand_slice_bounds(bd, n)
	z = zeros(Int, n)
	for i = 1:length(bd)-1
		z[bd[i]:bd[i+1]-1] .= i
	end
	return z
end

function fit!(sir::SlicedInverseRegression; ndir::Integer = 2)

	# Get the SIR directions
	sir.M = StatsBase.cov(copy(sir.sm'), fweights(sir.fw); corrected = false)
	eg = eigen(sir.M)

	# Raw eigenvalues and eigenvectors
	sir.eigs = eg.values[end:-1:1]
	sir.eigv = eg.vectors[:, end:-1:1]

	# Map back to the original coordinates
	dirs = eg.vectors[:, end:-1:1]
	sir.dirs = dirs[:, 1:ndir]
	sir.dirs = sir.trans \ sir.dirs

	# Scale to unit length
	for j = 1:size(sir.dirs, 2)
		sir.dirs[:, j] ./= norm(sir.dirs[:, j])
	end
end

"""
	sir(y, x; nslice=20, ndir=2)

Use Sliced Inverse Regression (SIR) to estimate the effective dimension reduction (EDR) space.

'y' must be sorted before calling 'fit'.
"""
function fit(
	::Type{SlicedInverseRegression},
	X,
	y;
	nslice = max(8, size(X, 2) + 3),
	ndir = 2,
)
	if !issorted(y)
		error("y must be sorted")
	end
	sm = SlicedInverseRegression(y, X, nslice)
	fit!(sm; ndir = ndir)
	return sm
end
