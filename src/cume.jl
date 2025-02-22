
mutable struct CumulativeSlicingEstimation <: DimensionReductionModel

    # The covariate data
    X::AbstractMatrix

    # The data used for cumulative moment estimation, may be the centered
    # data, the standardized data, or the decorrelated data according to the
    # value of `scale` (by default it is the decorrelated data).
    Z::AbstractMatrix

    # The response data
    y::AbstractVector

    # Scaling matrix: Z = X / R
    R::AbstractMatrix

    # The kernel matrix
    M::AbstractMatrix

    "`dirs`: the columns are a basis for the estimated sufficient dimension reduction (SDR) space"
    dirs::AbstractMatrix

    "`eigs`: the eigenvalues of `M`, sorted by decreasing eigenvalue"
    eigs::AbstractVector

    # cov, sd, or none based on the type of standardization/decorrelation that is done to the covariates.
    scale::Symbol
end

function fit!(cm::CumulativeSlicingEstimation; ndir=3)

    (; y, Z, R) = cm

    A = cumsum(Z; dims=1)
    M = Symmetric(A' * A) / length(y)

    a, b = eigen(M)
    a = reverse(a)
    b = reverse(b; dims=2)
    cm.dirs = R \ b[:, 1:ndir]
    cm.eigs = a
end

function fit(::Type{CumulativeSlicingEstimation}, X, y; ndir=2, scale=:cov)

    if !issorted(y)
        error("The responses 'y' must be sorted")
    end

    Xc = X .- mean(X; dims=1)
    n, p = size(Xc)

    if scale == :none
        R = I(p)
    elseif scale == :sd
        R = Diagonal(std(Xc; dims=1)[:])
    elseif scale == :cov
        R = ssqrt(Symmetric(cov(Xc)))
    else
        error("invalid value $(scale) of scale")
    end

    Z = Xc / R

    cm = CumulativeSlicingEstimation(X, Z, y, R, zeros(0, 0), zeros(0, 0), zeros(0), scale)

    fit!(cm; ndir=ndir)
    return cm
end
