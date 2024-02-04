
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

    "`Z`: the whitened explanatory variables"
    Z::AbstractMatrix

	"`Xmean`: the means of the columns of X"
	Xmean::AbstractVector

    "`nslice`: the number of slices"
    nslice::Int

    "`M`: the save kernel matrix"
    M::AbstractMatrix

    "`A`: sqrt(f_j)*(I - S_j) for each slice j, where S_j is the within-slice covariance matrix"
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

    "`bd`: slice bounds"
    bd::AbstractVector

    "`slice_assignments`: the slice indicator for each observation, aligns
    with data supplied by user"
    slice_assignments::AbstractVector
end

function whitened_predictors(m::SlicedAverageVarianceEstimation)
    return m.Z
end

function modelmatrix(m::SlicedAverageVarianceEstimation)
    return m.X
end

function nobs(m::SlicedAverageVarianceEstimation)
    return length(m.y)
end

function nvar(m::SlicedAverageVarianceEstimation)
    return size(m.X, 2)
end

function response(m::SlicedAverageVarianceEstimation)
    return m.y
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
    X, mn = center(X)
    Z, trans = whiten(X)

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

    A, M = save_kernel(Z, bd, fw)

    # Actual number of slices, may differ from nslice
    h = length(bd) - 1

    return SlicedAverageVarianceEstimation(
        y,
        X,
        Z,
        mn,
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
        push!(A, sqrt(fw[i]) * c)
        M .+= nw[i] * c * c
    end

    M ./= n

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
    X::AbstractMatrix,
    y::AbstractVector;
    nslice = max(8, size(X, 2) + 3),
    ndir = min(5, size(X, 2)),
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

struct SAVEDimensionTest <: HypothesisTest
    nstat::Vector{Float64}
    gstat::Vector{Float64}
    ndof::Vector{Float64}
    gdof::Vector{Float64}
end

function teststat(dt::SAVEDimensionTest; method=:normal)
    (; nstat, gstat, ndof, gdof) = dt
    if method == :normal
        return dt.nstat
    elseif method == :general
        return dt.gstat
    else
        throw(ArgumentError("Unkown method='$(method)'"))
    end
end

function dof(dt::SAVEDimensionTest; method=:normal)
    (; nstat, gstat, ndof, gdof) = dt
    if method == :normal
        return dt.ndof
    elseif method == :general
        return dt.gdof
    else
        throw(ArgumentError("Unkown method='$(method)'"))
    end
end

function pvalue(dt::SAVEDimensionTest; method=:normal)
    (; nstat, gstat, ndof, gdof) = dt
    if method == :normal
        return 1 .- cdf.(Chisq.(ndof), nstat)
    elseif method == :general
        return 1 .- cdf.(Chisq.(gdof), gstat)
    else
        throw(ArgumentError("Unkown method='$(method)'"))
    end
end

function dimension_test(save::SlicedAverageVarianceEstimation; maxdim::Int = nvar(save), method=:chisq, args...)

    if !(method in [:chisq, :diva])
        @error("Unknown dimension test method '$(method)'")
    end

    if method == :diva
        return _dimension_test_diva(save; maxdim=maxdim, args...)
    end

    (; X, Z, A, dirs, eigs, eigv) = save
    h = save.nslice
    p = nvar(save)
    maxdim = maxdim < 0 ? min(p - 1, save.nslice - 2) : maxdim
    maxdim = min(maxdim, min(p - 1, save.nslice - 2))
    n = nobs(save)

    # Test statistic based on normal and general theory
    nstat = zeros(maxdim + 1)
    gstat = zeros(maxdim + 1)

    # Degrees of freedom based on normal and general theory
    ndof = zeros(maxdim + 1)
    gdof = zeros(maxdim + 1)

    for i = 0:maxdim

        E = eigv[:, i+1:end]
        H = Z * E
        ZH = zeros(n, (p - i) * (p - i))

        # Normal theory test statistics
        for j = 1:h
            nstat[i+1] += sum((E' * A[j] * E) .^ 2) * n / 2
        end

        # Normal theory degrees of freedom
        ndof[i+1] = (h - 1) * (p - i) * (p - i + 1) / 2

        # General theory test
        for j = 1:n
            ZH[j, :] = vec(H[j, :] * H[j, :]')
        end
        S = cov(ZH) / 2
        gdof[i+1] = (h - 1) * sum(diag(S))^2 / sum(S .^ 2)
        gstat[i+1] = nstat[i+1] * sum(diag(S)) / sum(S .^ 2)
    end

    return SAVEDimensionTest(nstat, gstat, ndof, gdof)
end

function slice_covs(X, bd)

    p = size(X, 2)
    h = length(bd) - 1
    sc = zeros(p, p, h)

    for i = 1:h
        sc[:, :, i] .= cov(X[bd[i]:bd[i+1]-1, :], corrected = false)
    end

    return sc
end

struct SAVECoordinateTest <: HypothesisTest
    stat_n::Float64
    dof_n::Float64
    pval_n::Float64
    stat_g::Float64
    dof_g::Float64
    pval_g::Float64
end

function pvalue(ct::SAVECoordinateTest; method=:normal)
    if method == :normal
        return ct.pval_n
    elseif method == :general
        return ct.pval_g
    else
        throw(ArgumentError("Unknown method=$(method) (options are normal and general)"))
    end
end

# Marginal chi^2 coordinate test
# https://academic.oup.com/biomet/article/94/2/285/223789#2524699
function _coord_test_chisq(save::SlicedAverageVarianceEstimation, Hyp::AbstractMatrix; pmethod=:bx)

    (; y, X, Z, Xmean, M, eigs, eigv, trans, fw, A, bd, slice_assignments, trans, nslice) = save

    n = nobs(save)
    p = nvar(save)
    h = length(bd) - 1 # Number of slices

    # Ensure that the hypothesis is non-redundant and make it orthogonal.
    Hyp, s, _ = svd(Hyp)
    Hyp = Hyp[:, s .> 1e-10]
    q = size(Hyp, 2)

    # cov(X) and its inverted symmetric square root
    Sigma = Symmetric(cov(X))
    Sri = ssqrti(Sigma)

    # Under the null hypotheis, col(Hz) is orthogonal to the SDR space.
    # The formula in the paper is wrong (??).
    B = Sri * Hyp
    Hz = B * ssqrti(Symmetric(B' * B))

    # Compute the test statistic
    T = 0.0
    for a in A
        B = Hz' * a * Hz
        T += tr(B*B)
    end
    T *= n

    # Calibrate the test statistic
    g = zeros(h)
    V = Z * Hz
    QN = zeros(n, q^2)
    QG = zeros(n, h*q^2)
    u = zeros(q, q)
    ii = 1
    for j in 1:h
        g .= -fw
        g[j] += 1
        g ./= sqrt.(fw)

        for i in bd[j]:bd[j+1]-1
            u = V[i, :] * V[i,:]'
            u .-= I(q)
            QN[ii, :] = kron(V[i, :], V[i, :])
            QG[ii, :] = kron(g, vec(u))
            ii += 1
        end
    end

    CN = cov(QN)
    startn = q^2 - Int(q * (q + 1) / 2) + 1
    CG = cov(QG)
    startg = h*q^2 - Int(h * q * (q + 1) / 2) + 1

    statn, dofn, pvaln = ct_pvalues(CN, T, h - 1, pmethod; start=startn)
    statg, dofg, pvalg = ct_pvalues(CG, T, 1, pmethod; start=startg)
    return SAVECoordinateTest(statn, dofn, pvaln, statg, dofg, pvalg)
end

#function _coord_test_vonmises(save::SlicedAverageVarianceEstimation, Hyp::AbstractMatrix, ndir::Int; pmethod=:bx)

#    (; y, X, Xmean, M, eigs, eigv, trans, fw, bd, slice_assignments, trans, nslice) = save

#    r = size(Hyp, 2)
#    n, p = size(X)

#    @assert size(Hyp, 1) == p

    # cov(X)
#    Sigma = Symmetric(cov(X))

    # cov(X)^{-1/2}
#    Sri = ssqrti(Symmetric(Sigma))

    # Calculate the test statistic
#    P = eigv[:, 1:ndir] * eigv[:, 1:ndir]'
#    J = Symmetric(Hyp' * (Sigma \ Hyp))
#    K = ssqrti(J)

	# The first test statistic
#    H = Sri * Hyp * K
#    T1 = n * tr(H' * P * H)

    # The second test statistic
#    PW = H * H'
#    QW = I(p) - PW
#    Mc = QW * M * QW
#    eg = eigen(Symmetric(Mc))
#    xi = eg.vectors[:, end-ndir+1:end]
#    Pc = xi * xi'
#    T2 = n * sum(abs2, P - Pc)

#    U = slice_means(X, bd) * Diagonal(fw)
#    eg = eigen(Sigma)
#    c = eg.values
#    P1 = eg.vectors
#    C1 = getC1(c)
#    F = eigv[:, 1:ndir] * Diagonal(1 ./ eigs[1:ndir]) * eigv[:, 1:ndir]'

#    V = slice_covs(X, bd)
#    for j in eachindex(fw)
#        V[:, :, j] .*= fw[j]
#    end

	# Lambda_sir and Lambda_save
#    Lsir = U * Diagonal(1 ./ fw) * U'
#    Lsave = 2 * Lsir - Sigma
 #   for j = 1:nslice
 #       v = V[:, :, j]
 #       u = U[:, j]
 #       f = fw[j]
 #       Lsave .+= v * (Sigma \ v) / f
 #       Lsave .-= v * (Sigma \ u) * u' / f^2
 #       Lsave .-= u * (u' * (Sigma \ v)) / f^2
 #       Lsave .+= (u' * (Sigma \ u)) * u * u' / f^3
 #   end

 #   Mstarstarsave = zeros(p, p)
 #   Lstarstarsir = zeros(p, p)
 ##   Lstarstarsave = zeros(p, p)
 #   Gammastarstar = zeros(p, p)
 #   Ustarstar = zeros(p, nslice)
 #   Vstarstar = zeros(p, p, nslice)
 #   SigmaStar = zeros(p, p)
 #   A = zeros(r, p)
 #   B = zeros(p, p)
 #   R = zeros(p, p)
 #   Omega1 = zeros(p * r, p * r)
 #   Omega2 = zeros(p * p, p * p)
 #   for i = 1:n

 #       Ustarstar .= X[i, :] * fw'
 #       Ustarstar[:, slice_assignments[i]] .+= X[i, :]

 #       for j = 1:nslice
 #           Vstarstar[:, :, j] .= -fw[j] * Sigma#
#			Vstarstar[:, :, j] .-= Ustarstar[:, j#] * Xmean'
 #       end
 #       Vstarstar[:, :, slice_assignments[i]] .+= X[i, :] * X[i, :]'

 #       Lstarstarsir .= Ustarstar * Diagonal(1 ./ fw) * U'
 #       SigmaStar .= X[i, :] * X[i, :]' - Sigma

#        Gammastarstar .= 0
#        for j = 1:nslice
#            f = fw[j]
#            u = U[:, j]
#            uss = Ustarstar[:, j]
#            v = V[:, :, j]
#            vss = Vstarstar[:, :, j]
#            pstar = (slice_assignments[i] == j ? 1 : 0) - f
#            Gammastarstar .+= -pstar * Sigma + vss * (Sigma \ v) / f + Sigma * (SigmaStar \ v) + vss
#            Gammastarstar .-= vss * (Sigma \ u) * u' / f^2 + Sigma * (SigmaStar \ u) * u' / f + uss * u' / f
#            Gammastarstar .= (uss * u' * (Sigma \ v)) / f^2
#            Gammastarstar .+= (u' * (Sigma \ u)) * uss * u' / f^3
#        end

        # R = SigStar^{-1/2}
#        R .= P1 * (C1 .* (P1' * SigmaStar * P1)) * P1'

#        Lstarstarsave .= 2 * Lstarstarsir - SigmaStar + Gammastarstar

#        Mstarstarsave .= R * Lsave * Sri + Sri * Lstarstarsave * Sri
#        A .= K * Hyp' * (R * P + Sri * Mstarstarsave * F)
#        Omega1 .+= vec(A) * vec(A)'

#        B .= (PW * Mstarstarsave + Sri * Hyp * (J \ Hyp') * R * M) * F
#        B .+= B'
#        Omega2 .+= vec(B) * vec(B)'
#    end

#    Omega1 ./= n
#    Omega2 ./= n

#    stat1, degf1, pval1 = ct_pvalues(Omega1, T1, pmethod)
#    stat2, degf2, pval2 = ct_pvalues(Omega2, T2, pmethod)

#    return SAVECoordinateTest(stat1, pval1, degf1, stat2, pval2, degf2)
#end

# SAVE-specific coordinate testing
function _coord_test(save::SlicedAverageVarianceEstimation, Hyp;
                    pmethod::Symbol = :bx, method::Symbol=:chisq)

    if method == :chisq
        return _coord_test_chisq(save, Hyp; pmethod=pmethod)
    else
        error("Unknown coordinate test method='$(method)'")
    end
end
