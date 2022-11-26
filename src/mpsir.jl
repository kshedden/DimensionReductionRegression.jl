using LinearAlgebra

mutable struct MPSIR

    # Centered Y variables
    Y::AbstractMatrix

    # Orthogonal matrix 
    Yo::AbstractMatrix

    # Yo = Y * Ty
    Ty::AbstractMatrix

    # Centered X variables
    X::AbstractMatrix

    # Orthogonal matrix
    Xo::AbstractMatrix

    # Xo = X * Tx
    Tx::AbstractMatrix

    # The dimension reduction coefficients, in the original
    # data coordinates
    Bx::AbstractMatrix
    By::AbstractMatrix

    # Eigenvalues
    eigx::AbstractVector
    eigy::AbstractVector
end

function canonical_angles(A, B)
    A, _, _ = svd(A)
    B, _, _ = svd(B)
    _, s, _ = svd(A' * B)
    return acos.(clamp.(s, 0, 1))
end

function MPSIR(Y::AbstractMatrix, X::AbstractMatrix)

    @assert size(X, 1) == size(Y, 1)

    Y = center(Y)
    X = center(X)

    # Transform Y to orthogonality
    Yo, s, v = svd(Y)
    Ty = v * diagm(1 ./ s)

    # Transform X to orthogonality
    Xo, s, v = svd(X)
    Tx = v * diagm(1 ./ s)

    return MPSIR(Y, Yo, Ty, X, Xo, Tx, zeros(0, 0), zeros(0, 0), zeros(0), zeros(0))
end

function mpslice(X, nslice)::Vector{Int}
    n = size(X, 1)
    R = zeros(Int, size(X))
    for j = 1:size(X, 2)
        R[:, j] = sortperm(sortperm(X[:, j])) .- 1
    end
    R = div.(R * nslice, n)
    q = [nslice^k for k = 0:size(X, 2)-1]
    return R * q
end

function dosir(X, Y, nslice, d)

    ii = mpslice(Y, nslice)
    mn = zeros(size(X, 2), nslice^size(Y, 2))
    n = zeros(Int, nslice^size(Y, 2))
    for (i, j) in enumerate(ii)
        n[j+1] += 1
        mn[:, j+1] += X[i, :]
    end

    ii = findall(n .> 0)
    n = n[ii]
    mn = mn[:, ii]

    mn ./= n[:, :]'
    cm = mn * diagm(n) * mn' / sum(n)
    u, s, v = svd(cm)
    return u[:, 1:d], s
end

function StatsBase.fit!(
    mp::MPSIR;
    dx::Int = 2,
    dy::Int = 2,
    nslicex::Int = 10,
    nslicey::Int = 10,
)

    Xo, Yo = mp.Xo, mp.Yo

    # Initial projection of Y using PC's
    u, s, v = svd(Yo)
    By = v[:, 1:dy]

    success = false
    Bx, eigx, eigy = nothing, nothing, nothing
    for itr = 1:100
        Bx1, eigx = dosir(Xo, Yo * By, nslicey, dx)
        By1, eigy = dosir(Yo, Xo * Bx1, nslicex, dy)
        if itr > 1
            a1 = maximum(abs, canonical_angles(Xo * Bx1, Xo * Bx))
            a2 = maximum(abs, canonical_angles(Yo * By1, Yo * By))
            if max(a1, a2) < 0.01
                success = true
                break
            end
        end

        Bx = Bx1
        By = By1
    end

    if !success
        @warn "mpsir did not converge"
    end

    # Convert to the original coordinates
    cx = cov(mp.X)
    bx = mp.Tx * Bx
    for b in eachcol(bx)
        b ./= sqrt(b' * cx * b)
    end

    # Convert to the original coordinates
    cy = cov(mp.Y)
    by = mp.Ty * By
    for b in eachcol(by)
        b ./= sqrt(b' * cy * b)
    end

    mp.Bx = bx
    mp.By = by
    mp.eigx = eigx
    mp.eigy = eigy
end

function StatsBase.coef(mp::MPSIR)
    return (mp.By, mp.Bx)
end

function eig(mp::MPSIR)
    return (mp.eigy, mp.eigx)
end
