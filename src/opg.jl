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

    bw::Float64

    models::Vector
end

function OPG(X, y, family; n_centers=-1, verbosity=0, bw=-1)

    if length(y) != size(X, 1)
        error(@sprintf("length of y should equal leading dimension of X"))
    end

    y = copy(y)
    X, _ = center(X)
    Z, trans = whiten(X)

    ZC = if n_centers == -1
        Z
    else
        spt = supportpoints(copy(Z'), n_centers; verbosity=verbosity, maxit_mm=5, maxit_grad=0)
        copy(spt')
    end

    return OPG(y, X, Z, ZC, trans, zeros(0, 0), zeros(0), family, bw, [])
end

function set_weights!(w, z, Z, bw)
    for i in eachindex(w)
        w[i] = sum(abs2, Z[i, :] - z)
    end
    w .= exp.(-w ./ (2 * bw^2))
end

function fit!(opg::OPG; ndir=2, verbosity=0)

    (; y, Z, ZC, trans, family, bw, models) = opg

    n, p = size(Z)
    m = size(ZC, 1)
    w = zeros(n)
    Q = zeros(m, p)

    XX = zeros(n, p+1)
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

    for c in eachcol(opg.dirs)
        c ./= norm(c)
    end

    opg.eigs = s[1:ndir]
end

# Set the bandwidth as a percentile of the distribution of pairwise distances.
function get_bandwidth(X; pr=0.01)

    n, _ = size(X)

    if n <= 100
        # Exhaustively enumerate all pairs
        di = zeros(Int(n*(n-1)/2))
        for i in 1:n
            for j in 1:i-1
                di[ii] = norm(X[i, :] - X[j, :])
                ii += 1
            end
        end
    else
        # Randomly sample pairs
        nsamp = 10000
        di = zeros(nsamp)
        ii = 1
        for i in 1:nsamp
            j1, j2 = sample(1:n, 2; replace=false)
            di[ii] = norm(X[j1, :] - X[j2, :])
            ii += 1
        end
    end

    return quantile(di, pr)
end

function fit(::Type{OPG}, X::AbstractMatrix, y::AbstractVector; family=Normal(),
             bw=-1, ndir=2, n_centers::Int=-1, verbosity::Int=0)

    opg = OPG(X, y, family; n_centers=n_centers, verbosity=verbosity, bw=bw)
    if bw <= 0
        opg.bw = get_bandwidth(X)
    end

    fit!(opg; ndir=ndir, verbosity=verbosity)
    return opg
end

function dimension_test(opg::OPG; maxdim::Int=nvar(opg), method=:diva, args...)

    if !(method in [:diva])
        @error("Unknown dimension test method '$(method)'")
    end

    if method == :diva
        return _dimension_test_diva(opg; maxdim=maxdim, args...)
    end
end
