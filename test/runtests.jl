using Dimred
using Test
using Random
using LinearAlgebra
using Statistics
using StatsBase
using RCall
using StableRNGs

function canonical_angles(A, B)
    A, _, _ = svd(A)
    B, _, _ = svd(B)
    _, s, _ = svd(A' * B)
    @assert maximum(abs, s) < 1 + 1e-10
    s = clamp.(s, -1, 1)
    return acos.(s)
end

@testset "slicer" begin

    y = [1, 1, 3, 3, 3, 4, 4, 5, 5, 5, 5, 9]
    u = unique(y)
    s = Dimred.slice1(y, u)
    @test isapprox(s, [1, 3, 6, 8, 12, 13])
    s = Dimred.slice2(y, u, 2)
    @test isapprox(s, [1, 8, 13])

    y = [1, 3, 3, 4, 4, 4, 5, 5, 5, 5, 9, 10, 10]
    u = unique(y)
    s = Dimred.slice1(y, u)
    @test isapprox(s, [1, 2, 4, 7, 11, 12, 14])
    s = Dimred.slice2(y, u, 3)
    @test isapprox(s, [1, 7, 11, 14])
end

@testset "MPSIR" begin

    Random.seed!(123)

    # Sample size
    n = 1000

    # Bases for true subspaces
    ty = [1 0 0 0 0; 0 1 0 0 0]'
    tx = [0 1 0 0 0; 0 1 1 0 0]'

    # Use these to introduce correlation into X and Y
    rx = randn(5, 5)
    ry = randn(5, 5)

    Y = randn(n, 5) * rx
    X = randn(n, 5) * ry
    y1 = X * tx[:, 1] + 0.5 * randn(n)
    y2 = X * tx[:, 2] + 0.5 * randn(n)
    Y[:, 1] = y1 + y2
    Y[:, 2] = y1 - y2

    mp = MPSIR(Y, X)
    fit!(mp; dx = 2, dy = 2)
    by, bx = StatsBase.coef(mp)

    @test maximum(abs, canonical_angles(X * bx, X * tx)) < 0.1
    @test maximum(abs, canonical_angles(Y * by, Y * ty)) < 0.1
end

function gendat(n, p, rng)

    X = randn(rng, n, p)
    for j in 1:p
        X[:, j] .-= mean(X[:, j])
    end
    y = X[:, 1] - X[:, 2] + 0.5*randn(rng, n)
    ii = sortperm(y)
    X = X[ii, :]
    y = y[ii]

    return X, y
end

@testset "SIR1-R" begin

    n = 200
    p = 5
    ndir = 3
    rng = StableRNG(123)

    X, y = gendat(n, p, rng)

    @rput X
    @rput y

    hyp = randn(p, 2)
    @rput hyp

    R"
    library(dr)
    r = dr.compute(X, y, array(1, length(y)))
    tst = dr.test(r)
    M = r$M
    "

    @rget r
    @rget tst
    @rget M
    evectors = r[:evectors][:, 1:ndir]

    m = fit(SlicedInverseRegression, X, y; ndir=ndir)
    mt = dimension_test(m)

    # Check the kernel matrix
    @test isapprox(eigen(m.M).values, eigen(M).values)
    @test isapprox(m.M, m.M')

    # Check that the dimension inference is the same.
    @test isapprox(pvalue(mt)[1:4], tst[1:4, :p_value])
    @test isapprox(mt.stat[1:4], tst[1:4, :Stat])
    @test isapprox(dof(mt)[1:4], tst[1:4, :df])

    @test all(size(evectors) .== size(coef(m)))

    # Check that the estimated SDR subspaces are identical.
    yr = X * evectors
    ym = X * coef(m)
    ang = canonical_angles(yr, ym)
    @test maximum(abs, ang) .< 1e-6
end

@testset "SIR1" begin

    Random.seed!(2144)

    n = 1000     # Sample size
    r = 0.5      # Correlation between variables
    r2 = 0.5     # R-squared

    # Test different population dimensions
    for j = 1:2

        # Explanatory variables
        x = randn(n, 2)
        x[:, 2] .= r * x[:, 1] + sqrt(1 - r^2) * x[:, 2]

        ey = if j == 1
            # Single index model
            lp = x[:, 2] - x[:, 1]
            1 ./ (1 .+ (lp .+ 1) .^ 2)
        else
            # Two index model
            (1 .+ x[:, 1]) .^ 1 ./ (1 .+ (1 .+ x[:, 2]) .^ 2)
        end

        # Generate the response with the appropriate R^2.
        ey ./= std(ey)
        s = sqrt((1 - r2) / r2)
        y = ey + s * randn(n)

        ii = sortperm(y)
        y = y[ii]
        x = x[ii, :]

        nslice = 50
        si = SlicedInverseRegression(y, x, nslice)
        fit!(si; ndir = 2)
        dt = dimension_test(si)
        @test pvalue(dt)[1] < 1e-3
        if j == 1
            ed = si.dirs[:, 1]
            td = [-1, 1]
            @test isapprox(ed[2] / ed[1], td[2] / td[1], atol = 0.01, rtol = 0.05)
            @test pvalue(dt)[2] > 0.1
            @test abs(si.eigs[1] / si.eigs[2]) > 5
        else
            @test pvalue(dt)[2] < 0.005
        end
    end
end

@testset "PHDy-R" begin

    n = 200
    p = 5
    ndir = 3
    rng = StableRNG(123)

    X, y = gendat(n, p, rng)

    @rput X
    @rput y

    R"
    library(dr)
    r = dr.compute(X, y, array(1, length(y)), method='phdy')
    tst = dr.test(r, numdir=5)
    M = r$M
    "

    @rget r
    @rget tst
    @rget M
    evectors = r[:evectors][:, 1:ndir]

    m = fit(PrincipalHessianDirections, X, y; ndir=ndir, method="y")
    mt = dimension_test(m)

    # Compare the kernel matrices
    @test isapprox(eigen(M).values, eigen(m.M).values)
    @test isapprox(m.M, m.M')

    # Check that the dimension inference is the same.
    @test isapprox(pvalue(mt)[1:4], tst[1:4, :p_value])
    @test isapprox(mt.stat[1:4], tst[1:4, :Stat])
    @test isapprox(dof(mt)[1:4], tst[1:4, :df])

    @test all(size(evectors) .== size(coef(m)))

    # Check that the estimated SDR subspaces are identical.
    yr = X * evectors
    ym = X * coef(m)
    ang = canonical_angles(yr, ym)
    @test maximum(abs, ang) .< 1e-6
end

@testset "PHDres-R" begin

    n = 200
    p = 5
    ndir = 3
    rng = StableRNG(123)

    X, y = gendat(n, p, rng)

    @rput X
    @rput y

    R"
    library(dr)
    r = dr.compute(X, y, array(1, length(y)), method='phdres')
    tst = dr.test(r, numdir=5)
    M = r$M
    "

    @rget r
    @rget tst
    @rget M
    evectors = r[:evectors][:, 1:ndir]

    m = fit(PrincipalHessianDirections, X, y; ndir=ndir, method="r")
    mt = dimension_test(m; maxdim=4)

    # Compare the kernel matrices
    @test isapprox(eigen(M).values, eigen(m.M).values)
    @test isapprox(m.M, m.M')

    # Check that the dimension inference is the same.
    @test isapprox(pvalue(mt)[1:4], tst[1:4, "Normal theory"])
    @test isapprox(mt.stat[1:4], tst[1:4, :Stat])
    @test isapprox(dof(mt)[1:4], tst[1:4, :df])

    @test all(size(evectors) .== size(coef(m)))

    # Check that the estimated SDR subspaces are identical.
    yr = X * evectors
    ym = X * coef(m)
    ang = canonical_angles(yr, ym)
    @test maximum(abs, ang) .< 1e-6
end

@testset "Check PHD estimates" begin

    Random.seed!(2142)

    n = 2500     # Sample size
    r = 0.5      # Correlation between variables
    td = [1, -2] # True direction

    # Test with different floating point widths
    for j = 1:2

        # A nonlinear single-index model
        x = randn(n, 2)
        x[:, 2] .= r * x[:, 1] + sqrt(1 - r^2) * x[:, 2]
        lp = x * td
        y = 0.1 * randn(n) + 1 ./ (1.0 .+ lp .^ 2)

        xx = j == 1 ? x : Array{Float32}(x)
        yy = j == 1 ? y : Array{Float32}(y)

        rd = fit(PrincipalHessianDirections, xx, yy; ndir = 1)

        ed = rd.dirs[:, 1]
        @test isapprox(ed[2] / ed[1], td[2] / td[1], atol = 0.01, rtol = 0.05)
        @test abs(rd.eigs[1] / rd.eigs[2]) > 10
    end
end

@testset "SAVE-R" begin

    n = 200
    p = 5
    ndir = 3
    rng = StableRNG(123)

    X, y = gendat(n, p, rng)

    @rput X
    @rput y
    @rput ndir

    R"
    library(dr)
    r = dr.compute(X, y, array(1, length(y)), method='save')
    tst = dr.test(r, numdir=ndir+1)
    M = r$M
    "

    @rget r
    @rget tst
    @rget M
    evectors = r[:evectors][:, 1:ndir]

    m = fit(SlicedAverageVarianceEstimation, X, y; ndir=ndir)
    mt = dimension_test(m; maxdim=ndir)

    # Compare the kernel matrices
    @test isapprox(eigen(M).values, eigen(m.M).values)
    @test isapprox(M, M')

    # Check that the dimension inference is the same.
    @test isapprox(pvalue(mt; method=:normal), tst[1:ndir+1, "p_value(Nor)"])
    @test isapprox(pvalue(mt, method=:general), tst[1:ndir+1, "p_value(Gen)"])
    @test isapprox(teststat(mt, method=:normal), tst[1:ndir+1, :Stat])
    @test isapprox(dof(mt, method=:normal), tst[1:ndir+1, "df(Nor)"])

    @test all(size(evectors) .== size(coef(m)))

    # Check that the estimated SDR subspaces are identical.
    yr = X * evectors
    ym = X * coef(m)
    ang = canonical_angles(yr, ym)
    @test maximum(abs, ang) .< 1e-6
end

@testset "Check PHD tests" begin

    Random.seed!(2142)

    n = 2500     # Sample size
    r = 0.5      # Correlation between variables
    td = [1, -2] # True direction

    cs2, df2 = [], []
    for j = 1:500
        # A nonlinear single-index model
        X = randn(n, 2)
        X[:, 2] .= r * X[:, 1] + sqrt(1 - r^2) * X[:, 2]
        lp = X * td
        y = 0.1 * randn(n) + 1 ./ (1.0 .+ (1 .+ lp) .^ 2)

        ph = fit(PrincipalHessianDirections, X, y; ndir = 1)

        dt = dimension_test(ph)
        push!(cs2, dt.stat[2])
        push!(df2, dof(dt)[2])
    end

    # cs2 should behave like a sample of chi^2(2) values
    @test abs(mean(cs2) - 1) < 0.03
    @test abs(var(cs2) - 2) < 0.4
end


@testset "CORE" begin

    Random.seed!(4543)

    p = 10
    n = 5
    covs = Array{Array{Float64,2}}(undef, n)
    ns = Array{Int64}(undef, n)
    q = 3

    # The first q x q block is varying, the rest of the matrix
    # is constant
    for i = 1:n
        m = Array(Diagonal(ones(p)))
        r = 1 / (i + 1)
        for j1 = 1:q
            for j2 = 1:q
                m[j1, j2] = r^abs(j1 - j2)
            end
        end
        covs[i] = m
        ns[i] = 20 + 2 * i
    end

    # Start at the truth
    params = zeros(p, 3)
    params[1:3, 1:3] .= I(3)
    b = core(covs, ns, ndim = 3, maxiter = 100, params = params)

    # Start at a random point
    nfail1 = 0
    nfail2 = 0
    for k = 1:100
        params = randn(p, 3)
        s = svd(params)
        params = s.U
        c = core(covs, ns, params = params, ndim = 3)

        # Check that the projection matrices are the same
        t1 = b.proj * transpose(b.proj)
        t2 = c.proj * transpose(c.proj)
        e1 = maximum(abs.(t1 - t2))
        if e1 > 0.001
            nfail1 += 1
        end

        # Check that the log-likelihood is about the same
        e2 = b.llf - c.llf
        if e2 > 0.001
            nfail2 += 1
        end

    end

    @test nfail1 <= 10
    @test nfail2 <= 10
end
