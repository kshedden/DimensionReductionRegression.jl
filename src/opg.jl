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

    eigs::AbstractVector

    family::Distribution

    models::Vector
end

function OPG(X, y, family; n_centers=-1, verbosity=0)

    if length(y) != size(X, 1)
        error(@sprintf("length of y should equal leading dimension of X"))
    end

    y = copy(y)
    X, _ = center(X)
    Z, trans = whiten(X)

    ZC = if n_centers == -1
        Z
    else
        spt = supportpoints(copy(Z'), n_centers; verbosity=verbosity)
        copy(spt')
    end

    return OPG(y, X, Z, ZC, trans, zeros(0, 0), zeros(0), family, [])
end

function set_weights!(w, z, Z, bw)
    for i in eachindex(w)
        w[i] = sum(abs2, Z[i, :] - z)
    end
    w .= exp.(-w ./ (2 * bw^2))
end

function fit!(opg::OPG; bw=1.0, ndir=2, verbosity=0)

    (; y, Z, ZC, trans, family, models) = opg

    n, p = size(Z)
    m = size(ZC, 1)
    w = zeros(n)
    Q = zeros(m, p)

    XX = zeros(n, p + 1)
    XX[:, 1] .= 1
    XX[:, 2:end] .= Z

    # Loop over the anchor points
    for i in 1:m
        # Localize the regression around ZC[i, :]
        set_weights!(w, ZC[i, :], Z, bw)
        md = fit(GLM.GeneralizedLinearModel, XX, y, family; wts=w)
        push!(models, md)
        Q[i, :] = coef(md)[2:end]

        if verbosity > 0 && (i % 100 == 0)
            println("$(i)/$(m)")
        end
    end

    _, s, v = svd(Q)
    opg.dirs = trans \ v[:, 1:ndir]
    opg.eigs = s[1:ndir]
end

function fit(::Type{OPG}, X::AbstractMatrix, y::AbstractVector; family=Normal(),
             bw=sqrt(size(X, 2)), ndir=2, n_centers::Int=-1, verbosity::Int=0)

    opg = OPG(X, y, family; n_centers=n_centers, verbosity=verbosity)
    fit!(opg; bw=bw, ndir=ndir, verbosity=verbosity)
    return opg
end
