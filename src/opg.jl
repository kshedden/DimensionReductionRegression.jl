using GLM
using SupportPoints

mutable struct OPG <: DimensionReductionModel

    # The response values
    y::AbstractVector

    # The covariates
    X::AbstractMatrix

    # The whitened covariates
    Z::AbstractMatrix

    # For computational speed, calculate local regressions around
    # each row of ZC.  By default, ZC=Z but a smaller number of
    # points determined using the support point algorithm can
    # be specified.
    ZC::AbstractMatrix

    trans::AbstractMatrix

    dirs::AbstractMatrix

    family::Distribution
end

function OPG(X, y, family; n_centers=-1, maxiter_spt=20)

    if length(y) != size(X, 1)
        error(@sprintf("length of y should equal leading dimension of X"))
    end

    y = copy(y)
    X, _ = center(X)
    Z, trans = whiten(X)

    ZC = if n_centers == -1
        Z
    else
        spt = supportpoints(Z', n_centers; maxit=maxiter_spt)
        copy(spt')
    end

    return OPG(y, X, Z, ZC, trans, zeros(0, 0), family)
end

function set_weights!(w, z, Z, bw)
    for i in eachindex(w)
        w[i] = sum(abs2, Z[i, :] - z)
    end
    w .= exp.(-w ./ (2 * bw^2))
end

function fit!(opg::OPG; bw=1.0, ndir=2, verbose=false)

    (; y, Z, ZC, trans, family) = opg

    n, p = size(Z)
    m = size(ZC, 1)
    w = zeros(n)
    Q = zeros(m, p)

    XX = zeros(n, p + 1)
    XX[:, 1] .= 1
    XX[:, 2:end] .= Z

    for i in 1:m
        # Localize the regression around ZC[i, :]
        set_weights!(w, ZC[i, :], Z, bw)
        md = fit(GLM.GeneralizedLinearModel, XX, y, family; wts=w)
        Q[i, :] = coef(md)[2:end]

        if verbose && (i % 100 == 0)
            println("$(i)/$(m)")
        end
    end

    _, _, v = svd(Q)
    opg.dirs = trans \ v[:, 1:ndir]

end

function fit(::Type{OPG}, X::AbstractMatrix, y::AbstractVector; family=Normal(),
             bw=sqrt(size(X, 2)), ndir=2, n_centers::Int=-1, verbose::Bool=false)

    opg = OPG(X, y, family; n_centers=n_centers)
    fit!(opg; bw=bw, ndir=ndir, verbose=verbose)
    return opg
end
