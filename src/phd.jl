
"""
    PrincipalHessianDirections

Fit a regression model using principal Hessian directions
"""
mutable struct PrincipalHessianDirections <: DimensionReductionModel

    "`y`: the response variable, sorted"
    y::AbstractVector

    "`X`: the explanatory variables, sorted to align with `y`"
    X::AbstractMatrix

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

    "`n`: the sample size"
    n::Int
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
    Xw, trans = whiten(X)

    y = copy(y)
    y = _resid(y, X, method)

    M = Xw' * Diagonal(y) * Xw
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

    return PrincipalHessianDirections(y, X, mn, M, dirs, eigs, method, length(y))
end

"""
    dimension_test(s)

Returns p-values and Chi-squared statistics for the null hypotheses
that only the largest k eigenvalues are non-null.
"""
function dimension_test(s::PrincipalHessianDirections)

    p = length(s.eigs)
    stat = zeros(p)
    pv = zeros(p)
    df = zeros(Int, p)
    vy = var(s.y)

    for k = 0:p-1
        stat[k+1] = s.n * (p - k) * mean(abs2, s.eigs[k+1:end]) / (2 * vy)
        df[k+1] = div((p - k + 1) * (p - k), 2)
        pv[k+1] = 1 - cdf(Chisq(df[k+1]), stat[k+1])
    end

    return (Pvals = pv, Stat = stat, Degf = df)
end

function coef(r::PrincipalHessianDirections)
    return r.dirs
end
