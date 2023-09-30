
"""
    PrincipalHessianDirections

Fit a regression model using principal Hessian directions
"""
mutable struct PrincipalHessianDirections <: DimensionReductionModel

    "`y`: the response variable, sorted"
    y::AbstractVector

    "`X`: the explanatory variables, sorted to align with `y`"
    X::AbstractMatrix

    "`Z`: the whitened explanatory variables, sorted to align with `y`"
    Z::AbstractMatrix

	"`Xmean`: the means of the columns of X"
	Xmean::AbstractVector

    "`M`: the kernel matrix"
    M::AbstractMatrix

    "`dirs`: a basis for the estimated effective dimension reduction (EDR) space"
    dirs::AbstractMatrix

    "`eigs`: the eigenvalues"
    eigs::AbstractVector

    "`method`: one of 'y', 'r', or 'q'."
    method::String
end

function modelmatrix(m::PrincipalHessianDirections)
    return m.X
end

function nobs(m::PrincipalHessianDirections)
    return length(m.y)
end

function nvar(m::PrincipalHessianDirections)
    return size(m.X, 2)
end

function response(m::PrincipalHessianDirections)
    return m.y
end

function whitened_predictors(m::PrincipalHessianDirections)
    return m.Z
end

function _resid(y, X, method)

    y = y .- mean(y)

    if method == "y"
        return y
    elseif method == "r"
        qrx = qr(X)
        q = Matrix(qrx.Q)
        y .-= q * (q' * y)
        return y
    elseif method == "q"
        error("q-method PHD is not implemented yet")
    end

    return y
end


"""
    fit(PrincipalHessianDirections, X, y; method="y", ndir=2)

Use Principal Hessian Directions (PHD) to estimate the effective dimension reduction (EDR) space.
"""
function fit(
    ::Type{PrincipalHessianDirections},
    X::AbstractMatrix,
    y::AbstractVector;
    method::String = "y",
    ndir::Integer = 2,
)

    if !(method in ["y", "r", "q"])
        error("Method must be one of 'y', 'r', or 'q'")
    end

    # Dimensions of the problem
    n, p = size(X)

    X, mn = center(X)
    Z, trans = whiten(X)

    y = copy(y)
    y = _resid(y, X, method)

    M = Z' * Diagonal(y) * Z
    M ./= n

    eg = eigen(M)
    ii = sortperm(-abs.(eg.values))
    eigs = eg.values[ii]
    dirs = eg.vectors[:, ii[1:ndir]]

    # Map back to the original coordinates
    dirs = trans \ dirs

    # Scale to unit length
    for j = 1:size(dirs, 2)
        dirs[:, j] ./= norm(dirs[:, j])
    end

    return PrincipalHessianDirections(y, X, Z, mn, M, dirs, eigs, method)
end

"""
    dimension_test(s)

Returns p-values and Chi-squared statistics for the null hypotheses
that only the largest k eigenvalues are non-null.
"""
function dimension_test(phd::PrincipalHessianDirections; maxdim::Int = nvar(phd), method=:chisq, args...)

    if !(method in [:chisq, :diva])
        @error("Unknown dimension test method '$(method)'")
    end

    if method == :diva
        return _dimension_test_diva(phd; maxdim=maxdim, args...)
    end

    p = length(phd.eigs)
    stat = zeros(p)
    dof = zeros(Int, p)
    vy = var(phd.y)

    for k = 0:p-1
        stat[k+1] = nobs(phd) * (p - k) * mean(abs2, phd.eigs[k+1:end]) / (2 * vy)
        dof[k+1] = div((p - k + 1) * (p - k), 2)
    end

    return DimensionTest(stat, dof)
end

function coef(r::PrincipalHessianDirections)
    return r.dirs
end
