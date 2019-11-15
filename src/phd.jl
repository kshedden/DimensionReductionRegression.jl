"""
    phd(y, x; ndir=2)

Use Principal Hessian Directions (PHD) to estimate the effective dimension reduction (EDR) space.
"""
function phd(y::Array{S}, x::Array{T, 2}; ndir::Integer=2)::DimensionReductionEigen where {S,T<:AbstractFloat}

    # Dimensions of the problem
    n = size(x)[1]
    p = size(x)[2]

    x = copy(x)
    _center!(x)
    y = copy(y)
    y = y .- mean(y)

    cm = zeros(Float64, p, p)
    for i in 1:n
        for j in 1:p
            for k in 1:p
                cm[j, k] += y[i] * x[i, j] * x[i, k]
            end
        end
    end
    cm /= n

    cx = StatsBase.cov(x)
    cb = cx\cm

    eg = eigen(cb)

    ii = sortperm(-abs.(eg.values))
    eigs = eg.values[ii]
    dirs = eg.vectors[:, ii[1:ndir]]

    return DimensionReductionEigen(dirs, eigs)

end
