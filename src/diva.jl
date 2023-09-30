
struct DIVADimensionTest
    dimension::Int
    pvals::Vector{Float64}
    stats::Matrix{Float64}
    dof1::Matrix{Float64}
    dof2::Matrix{Float64}
    raw_pvals::Matrix{Float64}
end

# Generate Beta-referenced p-values for DIVA.  These
# are internal values and not directly reported as
# results.
function _diva(m::DimensionReductionModel, r)

    Z = whitened_predictors(m)
    n, p = size(Z)

    E = randn(n, r)
    ZZ = hcat(Z, E)
    mm = fit(typeof(m), ZZ, response(m); ndir=r)

    cc = coef(mm)
    b1 = cc[1:p, :]
    ss = sum(abs2, b1; dims=1)
    q = min(p, length(ss))
    ss = ss[1:q]
    dof1 = zeros(Float64, q)
    dof2 = zeros(Float64, q)

    pvals = zeros(q)
    for j in eachindex(ss)
        dof1[j] = (p - j + 1) / 2
        dof2[j] = r / 2
        pvals[j] = 1 - cdf(Beta(dof1[j], dof2[j]), ss[j])
    end

    return ss, pvals, dof1, dof2
end

# Use the DIVA approach to assess the dimension of a dimension
# reduction regression model using sequential hypothesis tests.
# 'alpha' is the type-1 error rate, 'maxdim' is the greatest dimension
# considered in testing, 's' is the number of times DIVA is repeated,
# and 'r' is the number of pseudo-covariates used to augment the
# regression design matrix.  If 'r' or 'maxdim' are not provided,
# their default values are the number of observed covariates.
function _dimension_test_diva(m::DimensionReductionModel; maxdim::Int=nvar(m), s::Int=1,
                              r::Int=nvar(m), alpha::Float64=0.05)

    if s > 1
        return _dimension_test_diva_stabilized(m, s, maxdim, r, alpha)
    end

    stats, pvals, dof1, dof2 = _diva(m, r)
    for j in eachindex(pvals)
        if pvals[j] > alpha
            return DIVADimensionTest(j - 1, pvals, stats[:, :], dof1[:, :], dof2[:, :], pvals[:, :])
        end
    end

    return DIVADimensionTest(length(pvals), pvals, stats[:, :], dof1[:, :], dof2[:, :], pvals[:, :])
end

function _dimension_test_diva_stabilized(m::DimensionReductionModel, s::Int, maxdim::Int,
                                         r::Int, alpha::Float64)

    p = nvar(m)

    # Run DIVA s times
    di = [_diva(m, r) for _ in 1:s]
    stats = hcat([x[1] for x in di]...)
    pvals = hcat([x[2] for x in di]...)
    dof1 = hcat([x[3] for x in di]...)
    dof2 = hcat([x[4] for x in di]...)

    pva = []
    for j in 1:maxdim
        # The candidate index set
        c = if j == 1
            collect(1:s)
        else
            [i for i in 1:s if maximum(pvals[1:j-1, i]) <= alpha]
        end
        pv = if length(c) == 0
            1.0
        else
            pp = pvals[j, c]
            sort!(pp)
            length(pp) * minimum(pp ./ (1:length(pp)))
        end
        push!(pva, pv)
    end

    pva = clamp.(pva, 0, 1)

    j = findfirst(pva .> alpha)
    d0 = min(p, r)
    d = isnothing(j) ? d0 : j - 1

    return DIVADimensionTest(d, pva, stats, dof1, dof2, pvals)
end

function _coord_test_diva(m::DimensionReductionModel, H0::AbstractMatrix; s::Int=30,
                          alpha::Float64=0.05, r::Int=min(10, nvar(m)))

    if nvar(m) != size(H0, 1)
        throw(ArgumentError("Hypothesis matrix must have same number of rows as number of variables"))
    end

    # Basis for space of hypothesized null variables
    E1, e1, _ = svd(H0)
    E1 = E1[:, e1 .> 1e-12]

    # Basis for space of hypothesized non-null variables
    p = nvar(m)
    E0, e0, _ = svd(I(p) - E1 * E1')
    E0 = E0[:, e0 .> 1e-12]

    X = modelmatrix(m)

    Z0, e, _ = svd(X * E0)
    Z0 = Z0[:, e .> 1e-12]

    Z1, e, _ = svd(X*E1 - Z0*Z0'*X*E1)
    Z1 = Z1[:, e .> 1e-12]

    m0 = fit(typeof(m), Z1, response(m))
    return _dimension_test_diva(m0; s=s, alpha=alpha, maxdim=1, r=r)
end
