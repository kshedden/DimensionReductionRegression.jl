abstract type DimensionReductionModel <: RegressionModel end

function coef(m::T) where {T<:DimensionReductionModel}
    return m.dirs
end

function whitened_predictors(m::T) where {T<:DimensionReductionModel}
    return m.Z
end

function modelmatrix(m::T) where {T<:DimensionReductionModel}
    return m.X
end

function nobs(m::T) where {T<:DimensionReductionModel}
    return length(m.y)
end

function nvar(m::T) where {T<:DimensionReductionModel}
    return size(m.X, 2)
end

function response(m::T) where {T<:DimensionReductionModel}
    return m.y
end

struct DimensionTest <: HypothesisTest
    stat::Vector{Float64}
    dof::Vector{Int64}
end

function show(io::IO, dt::DimensionTest)

    stat = teststat(dt)
    pv = pvalue(dt)
    d = length(pv)
    header = ["Null dimension", "Alternative dimension", "Statistic", "P-value"]

    da = hcat(["$(j-1)" for j in 1:d], [">=$(j)" for j in 1:d], stat, pv)

    println(io, "Dimension test:")

    pretty_table(io, da; header=header, tf=tf_simple)
end

function teststat(dt::DimensionTest)
    return dt.stat
end

function pvalue(dt::DimensionTest)
    (; stat, dof) = dt
    return 1 .- cdf.(Chisq.(dof), stat)
end

struct CoordinateTest
    tstat::Float64
    dof::Float64
    rstat::Float64
    pvalue::Float64
end

function pvalue(ct::CoordinateTest)
    return ct.pvalue
end

function teststat(ct::CoordinateTest)
    return ct.tstat
end

# Use the Bentler-Xie approach to obtain approximate
# Chi^2 statistics from a weighted sum of Chi^2(1).
# The weights are the eigenvalues of Omega and T
# is the value of the test statistic.
function bx_pvalues(Omega, T, dof, start)
    eg = eigen(Symmetric(Omega))
    eigs = eg.values[start:end]
    eigs = repeat(eigs, dof)
    degf = sum(eigs)^2 / sum(eigs .^ 2)
    T *= degf / sum(eigs)
    pval = 1 - cdf(Chisq(degf), T)
    return T, degf, pval
end

# Use simulation to estimate p-values from a weighted
# sum of Chi^2 draws.  The weights are the eigenvalues of
# Omega, T is the value of the test statistic, dof is the
# degrees of freedom for each Chi^2 value.
function sim_pvalues(Omega, T, dof, start; nrep = 10000)
    eg = eigen(Symmetric(Omega))
    eigs = eg.values[start:end]
    p = length(eigs)

    pval = 0.0
    for i = 1:nrep
        x = rand(Chisq(dof), p)
        if dot(x, eigs) > T
            pval += 1
        end
    end
    pval /= nrep
    return T, 0, pval
end

function ct_pvalues(Omega, T, dof, pmethod; start=1)
    if pmethod == :bx
        return bx_pvalues(Omega, T, dof, start)
    elseif pmethod == :sim
        return sim_pvalues(Omega, T, dof, start)
    else
        error("Unknown p-value method $(pmethod)")
    end
end

"""
    coordinate_test(m, H0; resid=false, fit_kwds=(), dt_kwds=(), kwds...)

Test the null hypothesis that Hyp' * B = 0, where B is a basis for the
estimated SDR subspace.

If `resid` is false, a method-specific chi^2 test is used, depending
on the type of `m`.  In this case, `fit_kwds` and `dt_kwds` are
ignored and any additional keyword arguments are passed to the
method-specific dimension test.

If `resid` is true, the covariates are residualized relative to H0 and
a dimension test is used to assess the null hypothesis that the
dimension is zero.  In this case, `fit_kwds` specifies how the model
is fit to the residualized data, and `dt_kwds` specifies how the
dimension test is conducted.

References:

DR Cook. Testing predictor contributions in sufficient dimension
reduction.  Annals of Statistics (2004), 32:3.
https://arxiv.org/pdf/math/0406520.pdf

Huang SH, Shedden K, Chang HW. Inference for the dimension of a
regression relationship using pseudo-covariates. Biometrics. 2023
Sep;79(3):2394-2403. doi: 10.1111/biom.13812.
"""
function coordinate_test(m::S, H0::T; resid=false, fit_kwds=(), dt_kwds=(),
                         kwds...) where {S<:DimensionReductionModel,T<:AbstractMatrix}

    if resid
        return _coord_test_resid(m, H0; fit_kwds=fit_kwds, dt_kwds=dt_kwds)
    else
        # Call a method specific chi^2 test.
        return _coord_test(m, H0; kwds...)
    end
end

# Perform a coordinate test by residualizing the covariates and
# applying a dimension test.
function _coord_test_resid(m::T, H0; fit_kwds=(), dt_kwds=()) where {T<:DimensionReductionModel}

    @assert nvar(m) == size(H0, 1)

    # Basis for space of hypothesized null variables
    E1, s, _ = svd(H0)
    E1 = E1[:, s .> 1e-12]

    # Basis for space of hypothesized non-null variables
    p = nvar(m)
    E0, s, _ = svd(I(p) - E1 * E1')
    E0 = E0[:, s .> 1e-12]

    X = modelmatrix(m)

    Z0, s, _ = svd(X * E0)
    Z0 = Z0[:, s .> 1e-12]

    Z1, s, _ = svd(X*E1 - Z0*Z0'*X*E1)
    Z1 = Z1[:, s .> 1e-12]

    m0 = fit(typeof(m), Z1, response(m); fit_kwds...)
    dt = dimension_test(m0; maxdim=0, dt_kwds...)

    ct = CoordinateTest(first(teststat(dt)), 0, 0, first(pvalue(dt)))
    return ct
end

# Returns the symmetric square root of A.
function ssqrt(A::Symmetric)
    eg = eigen(A)
    F = eg.vectors
    Q = sqrt.(eg.values)
    return F * Diagonal(Q) * F'
end

# Returns the inverse of the symmetric square root of A.
function ssqrti(A::Symmetric)
    eg = eigen(A)
    F = eg.vectors
    Q = sqrt.(eg.values)
    return F * Diagonal(1 ./ Q) * F'
end
