
# Generate Beta-referenced p-values for DIVA.  These
# are internal values and not directly reported as
# results.
function _diva_pvals(m::DimensionReductionModel, r)

    Xw = whitened_predictors(m)
    n, p = size(Xw)

    E = randn(n, r)
    Z = hcat(Xw, E)
    mm = fit(typeof(m), Z, response(m); ndir=r)

    cc = coef(mm)
    b1 = cc[1:p, :]
    ss = sum(abs2, b1; dims=1)

    pvals = zeros(length(ss))
    for j in eachindex(ss)
        pvals[j] = 1 - cdf(Beta((p-j+1)/2, r/2), ss[j])
    end

    return pvals
end

# Use the DIVA approach to test the dimension of a dimension reduction regression
# model.  'maxdim' is the greatest dimension considered in testing, 's' is the number
# of times DIVA is repeated, and 'r' is the number of pseudo-covariates used to augment
# the regression design matrix.  If 'r' or 'maxdim' are not provided, their default values
# are the number of observed covariates.
function _dimension_test_diva(m::DimensionReductionModel; maxdim::Int=nvar(m), s::Int=1,
                              r::Int=nvar(m), alpha::Float64=0.05)

    if s > 1
        return _dimension_test_diva_stabilized(m, s, maxdim, r, alpha)
    end

    pvals = _diva_pvals(m, r)
    for j in eachindex(pvals)
        if pvals[j] > alpha
            return j - 1
        end
    end

    return length(pvals)
end

function _dimension_test_diva_stabilized(m::DimensionReductionModel, s::Int, maxdim::Int,
                                         r::Int, alpha::Float64)

    # Run DIVA s times
    pvals = [_diva_pvals(m, r) for _ in 1:s]

    pva = []
    for j in 1:maxdim
        # The candidate index set
        c = if j == 1
            collect(1:s)
        else
            [i for i in 1:s if maximum(pvals[i][1:j-1]) <= alpha]
        end
        pv = if length(c) == 0
            1.0
        else
            pp = [pvals[i][j] for i in c]
            sort!(pp)
            length(pp) * minimum(pp ./ (1:length(pp)))
        end
        push!(pva, pv)
    end

    pva = clamp.(pva, 0, 1)

    j = findfirst(pva .> alpha)
    d0 = min(p, r)
    return isnothing(j) ? d0 : j - 1
end
