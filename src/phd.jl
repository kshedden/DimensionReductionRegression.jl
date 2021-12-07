
"""
    PHDResults

The result of Principal Hession Directions.
"""
struct PHDResults

    "`dirs`: a basis for the estimated effective dimension reduction (EDR) space"
    dirs::Array{Float64,2}

    "`eigs`: the eigenvalues"
    eigs::Array{Float64}

    "`method`: one of 'y', 'r', or 'q'."
    method::String

    "`n`: the sample size"
    n::Int

end

function _resid(y::Array{S}, x::Array{S,2}, method::String) where {S<:AbstractFloat}

    y = y .- mean(y)

    if method == "y"
        y .= y ./ std(y)
        return y
    elseif method == "r"
        q, _ = qr(x)
        y .= y - q * (q' * y)
        y .= y / std(y)
        return y
    elseif method == "q"
        error("q-method PHD is not implemented yet")
    end

    return y

end


"""
    phd(y, x; ndir=2)

Use Principal Hessian Directions (PHD) to estimate the effective dimension reduction (EDR) space.
"""
function phd(
    y::Array{S},
    x::Array{T,2};
    method::String = "y",
    ndir::Integer = 2,
)::PHDResults where {S,T<:AbstractFloat}

    # Dimensions of the problem
    n, p = size(x)

    x = copy(x)
    _center!(x)
    y = copy(y)

    y = _resid(y, x, method)

    cm = zeros(Float64, p, p)
    for i = 1:n
        for j = 1:p
            for k = 1:p
                cm[j, k] += y[i] * x[i, j] * x[i, k]
            end
        end
    end
    cm /= n

    cx = StatsBase.cov(x)
    cb = cx \ cm

    eg = eigen(cb)

    ii = sortperm(-abs.(eg.values))
    eigs = eg.values[ii]
    dirs = eg.vectors[:, ii[1:ndir]]

    return PHDResults(dirs, eigs, method, length(y))

end

"""
    phd_test(s)

Returns p-values and Chi-squared statistics for the null hypotheses
that only the largest k eigenvalues are non-null.
"""
function phd_test(s::PHDResults)

    p = length(s.eigs)
    cs = zeros(p)
    pv = zeros(p)
    df = zeros(Int, p)

    for k = 0:p-1

        cs[k+1] = s.n * sum(abs2, s.eigs[k+1:end]) / 2
        df[k+1] = div((p - k + 1) * (p - k), 2)
        pv[k+1] = 1 - cdf(Chisq(df[k+1]), cs[k+1])

    end

    return tuple(pv, cs, df)

end
