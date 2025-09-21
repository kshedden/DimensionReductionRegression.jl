
DR = DimensionReductionRegression

@testset "slicer" begin

    y = [1, 1, 3, 3, 3, 4, 4, 5, 5, 5, 5, 9]
    u = unique(y)
    s = DR.slice1(y, u)
    @test isapprox(s, [1, 3, 6, 8, 12, 13])
    s = DR.slice2(y, u, 2)
    @test isapprox(s, [1, 8, 13])

    y = [1, 3, 3, 4, 4, 4, 5, 5, 5, 5, 9, 10, 10]
    u = unique(y)
    s = DR.slice1(y, u)
    @test isapprox(s, [1, 2, 4, 7, 11, 12, 14])
    s = DR.slice2(y, u, 3)
    @test isapprox(s, [1, 7, 11, 14])
end

@testset "MPSIR" begin

    rng = StableRNG(123)

    # Sample size
    n = 1000

    # Bases for true subspaces
    ty = [1 0 0 0 0; 0 1 0 0 0]'
    tx = [0 1 0 0 0; 0 1 1 0 0]'

    # Use these to introduce correlation into X and Y
    rx = randn(rng, 5, 5)
    ry = randn(rng, 5, 5)

    Y = randn(rng, n, 5) * rx
    X = randn(rng, n, 5) * ry
    y1 = X * tx[:, 1] + 0.5 * randn(rng, n)
    y2 = X * tx[:, 2] + 0.5 * randn(rng, n)
    Y[:, 1] = y1 + y2
    Y[:, 2] = y1 - y2

    mp = MPSIR(Y, X)
    fit!(mp; dx = 2, dy = 2)
    by, bx = StatsBase.coef(mp)

    @test maximum(abs, canonical_angles(X * bx, X * tx)) < 0.1
    @test maximum(abs, canonical_angles(Y * by, Y * ty)) < 0.1
end

@testset "SIR1-R" begin

    n = 200
    p = 5
    ndir = 3
    rng = StableRNG(123)

    X, y = gendat_linear(n, p, rng)

    @rput X
    @rput y

    hyp = randn(rng, p, 2)
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

    # Check that the coefficients are normalized
    @test all(isapprox.(sum(abs2, coef(m); dims=1), 1))

    # Check the kernel matrix
    @test isapprox(eigen(m.M).values, eigen(M).values)
    @test isapprox(m.M, m.M')

    # Check that the chi^2 dimension inference is the same.
    mt = dimension_test(m)
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

    rng = StableRNG(123)

    n = 1000     # Sample size
    r = 0.5      # Correlation between variables
    r2 = 0.5     # R-squared

    # Test different population dimensions
    for j = 1:2

        # Explanatory variables
        x = randn(rng, n, 2)
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
        y = ey + s * randn(rng, n)

        ii = sortperm(y)
        y = y[ii]
        x = x[ii, :]

        nslice = 50
        si = SlicedInverseRegression(y, x, nslice)
        fit!(si; ndir=2)
        dt = dimension_test(si)
        @test pvalue(dt)[1] < 1e-3
        if j == 1
            ed = si.dirs[:, 1]
            td = [-1, 1]
            @test isapprox(ed[2] / ed[1], td[2] / td[1], atol=0.01, rtol=0.1)
            @test pvalue(dt)[2] > 0.1
            @test abs(si.eigs[1] / si.eigs[2]) > 5
        else
            @test pvalue(dt)[2] < 0.005
        end
    end
end
