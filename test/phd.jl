@testset "PHDy-R" begin

    n = 200
    p = 5
    ndir = 3
    rng = StableRNG(123)

    X, y = gendat_linear(n, p, rng)

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

    # Check that the coefficients are normalized
    @test all(isapprox.(sum(abs2, coef(m); dims=1), 1))

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

    X, y = gendat_linear(n, p, rng)

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

    rng = StableRNG(123)

    n = 2500     # Sample size
    r = 0.5      # Correlation between variables
    td = [1, -2] # True direction

    # Test with different floating point widths
    for j = 1:2

        # A nonlinear single-index model
        x = randn(rng, n, 2)
        x[:, 2] .= r * x[:, 1] + sqrt(1 - r^2) * x[:, 2]
        lp = x * td
        y = 0.1 * randn(rng, n) + 1 ./ (1.0 .+ lp .^ 2)

        xx = j == 1 ? x : Matrix{Float32}(x)
        yy = j == 1 ? y : Vector{Float32}(y)

        rd = fit(PrincipalHessianDirections, xx, yy; ndir = 1)

        ed = rd.dirs[:, 1]
        @test isapprox(ed[2] / ed[1], td[2] / td[1], atol = 0.01, rtol = 0.05)
        @test abs(rd.eigs[1] / rd.eigs[2]) > 10
    end
end

@testset "Check PHD tests" begin

    rng = StableRNG(123)

    n = 2500     # Sample size
    r = 0.5      # Correlation between variables
    td = [1, -2] # True direction

    cs2, df2 = [], []
    for j = 1:500
        # A nonlinear single-index model
        X = randn(rng, n, 2)
        X[:, 2] .= r * X[:, 1] + sqrt(1 - r^2) * X[:, 2]
        lp = X * td
        y = 0.1 * randn(rng, n) + 1 ./ (1.0 .+ (1 .+ lp) .^ 2)

        ph = fit(PrincipalHessianDirections, X, y; ndir = 1)

        dt = dimension_test(ph)
        push!(cs2, dt.stat[2])
        push!(df2, dof(dt)[2])
    end

    # cs2 should behave like a sample of chi^2(2) values
    @test abs(mean(cs2) - 1) < 0.04
    @test abs(var(cs2) - 2) < 0.4
end

