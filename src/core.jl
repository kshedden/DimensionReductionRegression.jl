"""
    grass_opt(params, fun, grad!, maxiter, gtol) -> Tuple

Minimize a function on a Grassmann manifold using steepest descent.

# Arguments
- "`params::Array{Float64, 2}`" Starting value for the optimization.
- "`fun::Function`" The function to be minimized.
- "`grad!::Function`" The gradient of fun.
- "`maxiter::Integer`" The maximum number of iterations.
- "`gtol::Float64`" Convergence occurs when the gradient norm falls below this value.

The function returns a 3-tuple containing the parameter value that minimizes the function,
the minimizing function value, and a Boolean indicating whether the search converged.

# Reference
A Edelman, TA Arias, ST Smith (1998).  The geometry of algorithms with
orthogonality constraints. SIAM J Matrix Anal Appl.
http://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf
"""
function grass_opt(
    params::Array{Float64,2},
    fun::Function,
    grad!::Function,
    maxiter::Integer,
    gtol::Float64,
)::Tuple{Array{Float64},Float64,Bool,Vector}

    p, d = size(params)

    # Just in case the starting values are not feasible
    params = svd(params).U

    # History of function values and gradient norms
    hist = Vector()

    # Workspaces
    pa = zeros(Float64, p, d)
    pa0 = zeros(Float64, p, d)
    pa1 = zeros(Float64, p, d)
    g = zeros(Float64, p, d)
    h = zeros(Float64, p, d)

    # Initial function value
    f0 = fun(params)

    # Convergence status
    cnvrg = false

    for iter = 1:maxiter

        # Gradient
        grad!(params, g)
        g .= g .- params * (transpose(params) * g)
        gn = sqrt(dot(g, g))
        push!(hist, (f0, gn))
        if gn < gtol
            cnvrg = true
            return (params, f0, true, hist)
        end

        # Search direction
        h .= -g
        s = svd(h)

        pa0 .= params * s.V

        function geo!(t::Float64, pa::Array{Float64,2})::Array{Float64}
            # Parameterize the geodesic path in the direction
            # of the gradient as a function of a real value t.
            pa1 .= pa0 * Diagonal(cos.(s.S * t)) + s.U * Diagonal(sin.(s.S * t))
            pa .= pa1 * s.Vt
        end

        # Try to find a downhill step along the geodesic path.
        step = 2.0
        success = false
        while step > 1e-30
            geo!(step, pa)

            f1 = fun(pa)
            if f1 < f0
                params .= pa
                f0 = f1
                success = true
                break
            end
            step /= 2
        end

        if !success
            # Not sure what to do here
            break
        end

    end

    return (params, f0, cnvrg, hist)

end

# A private struct for maintaining the state of the CORE fitting
# algorithm.
mutable struct _core{S<:AbstractFloat,T<:Integer}

    # The sample covariance matrices
    covs::Array{Array{S,2}}

    # The sample size per group
    ns::Array{T}

    # The marginal covariance
    covm::Array{Float64,2}

    # The size of each covariance matrix is p x p
    p::Int64

    # The number of covariance matrices
    ng::Int64

    # The total sample size
    nobs::Int64

    # The reduced dimension
    ndim::Int64

end

# Fill in the marginal covariance matrix based on the group-wise
# covariance matrices and sample sizes.
function _marg_cov!(c::_core)

    p = size(c.covs[1])[1]
    covm = zeros(Float64, p, p)

    s = 0
    for i = 1:length(c.covs)
        if size(c.covs[i]) != (p, p)
            error("Expected $p x $p covariance matrix in position $i\n")
        end
        covm += c.ns[i] * c.covs[i]
        s += c.ns[i]
    end
    covm /= s

    c.covm = covm

end

function _check_gradient(co, pt::Array{Float64})

    p, d = size(pt)
    fun = x -> core_loglike(co, x)

    # Analytic gradient
    ag = zeros(p, d)
    core_score!(co, pt, ag)

    # Numerical gradient
    e = 1e-8
    ng = zeros(p, d)
    f0 = fun(pt)
    for i = 1:p
        for j = 1:d
            pt[i, j] += e
            ng[i, j] = (fun(pt) - f0) / e
            pt[i, j] -= e
        end
    end

    @assert isapprox(ag, ng, atol = 0.01, rtol = 0.01)

end

# The log-likelihood of a CORE model.
function core_loglike(co::_core, params::Array{Float64,2})::Float64

    b = transpose(params) * co.covm * params
    ldet = logdet(b)
    v = co.nobs * ldet / 2

    for i = 1:length(co.covs)
        b = transpose(params) * co.covs[i] * params
        ldet = logdet(b)
        v -= co.ns[i] * ldet / 2
    end

    return v

end

# The score vector of a CORE model.
function core_score!(co::_core, params::Array{Float64,2}, g::Array{Float64,2})

    c0 = transpose(params) * co.covm * params
    cP = co.covm * params
    g .= co.nobs * cP / c0

    for i = 1:length(co.covs)
        c0 .= transpose(params) * co.covs[i] * params
        cP .= co.covs[i] * params
        g .= g .- co.ns[i] * cP / c0
    end

end

"""
    CORE

CORE (covariance reduction) is a dimension reduction method for understanding
the unique features of matrices within a collection of covariance matrices.
The `CORE` result contains the projection directions `proj` and the optimized
log-likelihood function value `llf`.
"""
struct CORE

    # The covariance reduction matrix
    proj::Array{Float64,2}

    # The log-likelihood at the optimum
    llf::Float64

    # Log-likeihood, score norm pairs for each iteration
    hist::Vector

    # A convergence message
    msg::String
end

"""
    core(covs, ns, ndim=1, params=Array{Float64}(0, 0), maxiter, gtol)

Given a collection of covariance matrices C1, ..., Cm, covariance reduction (CORE)
finds an orthogonal matrix Q such that the reduced matrices Q'*Cj*Q capture most of
the variation among the Cj.

# Arguments
- `covs::Array{Array{S, 2}}` : the covariance matrices to reduce
- `ns::Array{T}` : the sample size used to estimate each covariance matrix
- `ndim::Integer=1` : number of dimensions in the reduction (number of columns of Q)
- `params::Array{Float64}` : starting values
- `maxiter::Integer` : the maximum number of iterations in the optimization
- `gtol::Float64` : return when the gradient norm is smaller than this value

# References
DR Cook, L Forzani (2008).  Covariance reducing models: an alternative
to spectral modeling of covariance matrices.  Biometrika 95:4.

A Edelman, TA Arias, ST Smith (1998).  The geometry of algorithms with
orthogonality constraints. SIAM J Matrix Anal Appl.
http://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf
"""
function core(
    covs::Array{Array{S,2}},
    ns::Array{T};
    ndim::Integer = 1,
    params::Array{Float64} = Array{Float64}(undef, 0),
    maxiter::Integer = 10000,
    gtol::Real = 1e-10,
) where {S<:AbstractFloat,T<:Integer}

    p = size(covs[1])[1]
    nobs = sum(ns)

    co = _core(covs, ns, zeros(Float64, 1, 1), p, length(covs), nobs, ndim)
    _marg_cov!(co)

    # Starting value for params
    if size(params) != (p, ndim)
        params = randn(p, ndim)
        params = svd(params).U
    end

    # Wrap the score function so we can flip the polarity.
    function score!(x::Array{Float64,2}, g::Array{Float64,2})
        core_score!(co, x, g)
        g .*= -1
    end

    # Used for debugging
    if false
        _check_gradient(co, params)
    end

    params, llf, cnvrg, hist =
        grass_opt(params, x -> -core_loglike(co, x), score!, maxiter, gtol)

    # Flip the polarity back to the usual direction
    llf *= -1

    # If the algorithm did not converge, provide a warning.
    msg = "CORE optimization converged successfully"
    if !cnvrg
        g = zeros(Float64, p, ndim)
        gn = hist[end][end]
        msg = Printf.@sprintf("CORE optimization did not converge, |g|=%14.10f\n", gn)
    end

    results = CORE(params, llf, hist, msg)

    return results
end
