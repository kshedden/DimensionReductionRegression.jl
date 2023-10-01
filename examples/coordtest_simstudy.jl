using DataFrames
using LinearAlgebra
using DimensionReductionRegression
using Statistics
using StableRNGs
using Printf

rng = StableRNG(123)

# R-squared
r2 = 0.2

# Nominal level
nomlevel = 0.1

function genx(n, p)
    X = randn(rng, n, p)
    # Half the variables are correlated, half are uncorrelated.
    r = 0.8
    for j in 2:div(p, 2)
        X[:, j] = r*X[:, j-1] + sqrt(1-r^2)*X[:, j]
    end
    return X
end

function sortyx(y, X)
    ii = sortperm(y)
    y = y[ii]
    X = X[ii, :]
    return y, X
end

function gendat1(n, p; r2=0.5)
    X = genx(n, p)
    ey = 1 ./ X[:, 1]
    s = sqrt((1 - r2) / r2)
    y = ey + s*randn(rng, n)
    return sortyx(y, X)
end

function gendat2(n, p; r2=0.5)
    X = genx(n, p)
    ey = exp.(X[:, 1]) .* sign.(X[:, 2])
    s = sqrt((1 - r2) / r2)
    y = ey + s*randn(rng, n)
    return sortyx(y, X)
end

function gendat3(n, p; r2=0.5)
    X = genx(n, p)
    ey = log.(abs.(X[:, 1]))
    s = sqrt((1 - r2) / r2)
    y = ey + s*randn(rng, n)
    return sortyx(y, X)
end

function gendat4(n, p; r2=0.5)
    X = genx(n, p)
    ey = 0.4*X[:, 1].^2 + 3*sin.(X[:, 2]/4)
    s = sqrt((1 - r2) / r2)
    y = ey + s*randn(rng, n)
    return sortyx(y, X)
end

function gendat(n, p, model; r2=0.5)
    if model == 1
        return gendat1(n, p; r2=r2)
    elseif model == 2
        return gendat2(n, p; r2=r2)
    elseif model == 3
        return gendat3(n, p; r2=r2)
    elseif model == 4
        return gendat4(n, p; r2=r2)
    else
        error("unknown model=$(model)")
    end
end

function genhyp(p)
    Hyp1 = zeros(p, 2)
    Hyp1[3:4, 1:2] = I(2) # null is true
    Hyp2 = zeros(p, 2)
    Hyp2[1:2, 1:2] = I(2) # null is false
    return Hyp1, Hyp2
end

function check_method(n, p, fitter; resid=false, method=:chisq, nrep=1000, dt_kwds=(), ct_kwds=())
    rslt = DataFrame(model=Int[], mode=String[], reject=Float64[])
    Hyp1, Hyp2 = genhyp(p)
    pvals = zeros(nrep)
    for model in [1, 2]
        for (jhyp, hyp) in enumerate([Hyp1, Hyp2])
            push!(rslt.model, model)
            if jhyp == 1
                push!(rslt.mode, "Level")
            else
                push!(rslt.mode, "Power")
            end
            for i in 1:nrep
                y, X = gendat(n, p, model; r2=r2)
                kwds = (resid=resid,)
                if resid
                    kwds = (kwds..., ct_kwds=ct_kwds, dt_kwds=dt_kwds)
                else
                    kwds = (kwds..., method=method)
                end
                m = fitter(X, y; ndir=model, nslice=5)
                ct = coordinate_test(m, hyp; kwds...)
                pvals[i] = pvalue(ct)
            end
            push!(rslt.reject, mean(pvals .< nomlevel))
        end
    end
    return rslt
end

nrep = 1000

sir_fitter = (X, y; ndir, nslice) -> fit(SlicedInverseRegression, X, y; ndir, nslice)
save_fitter = (X, y; ndir, nslice) -> fit(SlicedAverageVarianceEstimation, X, y; ndir=2, nslice=5)

function simstudy(fitter; resid=false, method=:chisq, dt_kwds=(), ct_kwds=())
    rslts = []
    for n in [200, 400]
        for p in [4, 8]
            r = check_method(n, p, fitter; resid=resid, method=method,
                             nrep=nrep,
                             dt_kwds=dt_kwds, ct_kwds=ct_kwds)
            r[:, :resid] .= resid
            r[:, :n] .= n
            r[:, :p] .= p
            push!(rslts, r)
        end
    end
    return vcat(rslts...)
end

function main(fitter)

    rslt1 = simstudy(fitter; resid=false, method=:chisq)
    rslt1[:, :method] .= "Chi^2 coordinate test"

    rslt2 = simstudy(fitter; resid=true)
    rslt2[:, :method] .= "Chi^2 dimension test on residuals"

    rslt3 = simstudy(fitter; resid=true, dt_kwds=(method=:diva, s=1))
    rslt3[:, :method] .= "DIVA dimension test on residuals"

    return vcat(rslt1, rslt2, rslt3)
end

rslt = main(sir_fitter)
display(rslt)

rslt = main(save_fitter)
display(rslt)
