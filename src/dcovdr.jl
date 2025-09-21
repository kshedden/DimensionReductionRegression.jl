
mutable struct DCovDR <: DimensionReductionModel

    "`X`: the centered explanatory variables.  The observations (variables) are in rows (columns) of `X`."
    X::AbstractMatrix

    "`Y`: the response variables"
    Y::AbstractMatrix

    alpha::Float64

    eta::Float64

    # Pairwise distances among rows of Y.
    B::AbstractMatrix

    # The estimated coefficient matrix
    dirs::AbstractMatrix

    fitted::Bool
end

function pairdist(X, alpha)

    n, p = size(X)
    u = zeros(p)
    A = zeros(n, n)

    for i in 1:n
        for j in 1:i
            u .= X[i, :] - X[j, :]
            A[i, j] = norm(u)^alpha
            A[j, i] = A[i, j]
        end
    end

    A .-= mean(A)
    A .-= mean(A; dims=1)
    A .-= mean(A; dims=2)

    return A
end

function DCovDR(X, Y; alpha=1.0, eta=1e-5)

    nx, p = size(X)
    ny, q = size(Y)

    if nx != ny
        error("X and Y must have the same number of observations (rows)")
    end

    B = pairdist(Y, alpha)

    dc = DCovDR(X, Y, alpha, eta, B, zeros(0, 0), false)

    return dc
end

# The objective function for dimension reduction using distance covariance.
function dcovdr_objective(dc::DCovDR, C)

    (; X, B, alpha, eta) = dc

    n, p = size(X)
    p1, d = size(C)
    @assert p1 == p
    u = zeros(d)

    XC = X * C
    lv = 0.0
    q = alpha / 2

    for i in 1:n
        for j in 1:i-1
            u .= XC[i, :] - XC[j, :]
            v = sum(abs2, u) + eta
            lv += v^q * B[i, j]
        end
    end

    return 2 * lv / n^2
end

function dcovdr_grad!(dc::DCovDR, C, grad)

    (; X, B, alpha, eta) = dc

    grad .= 0
    n, p = size(X)
    p1, d = size(C)
    @assert p == p1
    u = zeros(p)
    v = zeros(d)

    XC = X * C
    q = (2 - alpha) / 2

    for i in 1:n
        for j in 1:i-1
            u .= X[i, :] - X[j, :]
            v .= XC[i, :] - XC[j, :]
            grad .+= u * v' * B[i, j] / (sum(abs2, v) + eta)^q
        end
    end

    grad .*= 2 * alpha / n^2
end

function get_start(dc::DCovDR, d::Int)

    (; X, Y) = dc

    cc = []
    for j in 1:size(Y, 2)
        y = Y[:, j]
        ii = sortperm(y)
        m = fit(SlicedInverseRegression, X[ii, :], Y[ii, j])
        push!(cc, coef(m)[:, 1:d])
    end

    cc = hcat(cc...)
    u,_,_ = svd(cc)

    return u[:, 1:d]
end

function fit(::Type{DCovDR}, X, Y; alpha=1.0, eta=1e-6)
    m = DCovDR(X, Y; alpha=alpha, eta=eta)
    fit!(m)
    return m
end

function dcovdr_proj(C, d)
    u, s, v = svd(C)
    return u[:, 1:d] * v[:, 1:d]'
end

function fit!(dc::DCovDR; maxiter::Int=500, d::Int=2, minstep::Float64=1e-12, gtol::Float64=1e-6, verbosity::Int=0)

    (; X, Y) = dc
    n, p = size(X)
    step0 = 1.0

    C = get_start(dc, d)

    gr = zeros(p, d)

    obj0 = dcovdr_objective(dc, C)

    for iter in 1:maxiter
        dcovdr_grad!(dc, C, gr)
        gr = (I - C*C') * gr
        if norm(gr) < gtol
            break
        end
        if verbosity > 0
            println(@sprintf("%16.8f %16.8f", obj0, sum(abs2, gr)))
        end
        step = step0
        success = false

        # Line search
        while step > minstep
            C1 = dcovdr_proj(C + step*gr, d)
            obj1 = dcovdr_objective(dc, C1)
            if obj1 > obj0
                C = C1
                obj0 = obj1
                success = true
                break
            end
            step /= 2
        end

        if !success
            @warn("DCovDR failed to find uphill step, exiting early")
            break
        end
    end

    dc.dirs = C
    dc.fitted = true
end
