@testset "SAVE-R" begin

    n = 200
    p = 5
    ndir = 3
    rng = StableRNG(123)

    X, y = gendat_linear(n, p, rng)

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

    # Check that the coefficients are normalized
    @test all(isapprox.(sum(abs2, coef(m); dims=1), 1))

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
