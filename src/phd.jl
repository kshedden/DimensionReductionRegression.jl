
"""
    PHDResults

The result of Principal Hession Directions.
"""
struct PHDResults

    "`dirs`: a basis for the estimated effective dimension reduction (EDR) space"
    dirs::Array{Float64, 2}

    "`eigs`: the eigenvalues"
    eigs::Array{Float64}

end


"""
    phd(y, x; ndir=2)

Use Principal Hessian Directions (PHD) to estimate the effective dimension reduction (EDR) space.
"""
function phd(y::Array{S}, x::Array{T, 2}; ndir::Integer=2)::PHDResults where {S,T<:AbstractFloat}

    # Dimensions of the problem
    n, p = size(x)

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

    return PHDResults(dirs, eigs)

end
