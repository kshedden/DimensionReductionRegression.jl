using Dimred, Test, Random, LinearAlgebra, Statistics

function canonical_angles(A, B)
	A, _, _ = svd(A)
	B, _, _ = svd(B)
	_, s, _ = svd(A' * B)
	return acos.(s)
end

@testset "MPSIR" begin

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
	y1 = X * tx[:, 1] + 0.5*randn(n)
	y2 = X * tx[:, 2] + 0.5*randn(n)
	Y[:, 1] = y1 + y2
	Y[:, 2] = y1 - y2

	mp = MPSIR(Y, X)
	fit!(mp, 2, 2)
	by, bx = coef(mp)

	@test maximum(abs, canonical_angles(X*bx, X*tx)) < 0.1
	@test maximum(abs, canonical_angles(Y*by, Y*ty)) < 0.1
end

@testset "Slicer" begin

    y = [1, 2, 2, 3, 3, 3]
    ii = slicer(y, 3)
    @test all(ii .== [1, 4, 7])

    y = [1, 2, 3, 4, 5, 6]
    ii = slicer(y, 3)
    @test all(ii .== [1, 3, 5, 7])

    y = [1, 2, 3, 4, 5, 6, 7]
    ii = slicer(y, 3)
    @test all(ii .== [1, 3, 5, 8])

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

        rd = fit(SlicedInverseRegression, y, x; nslice = 50, ndir = 2)
        pv = sir_test(rd)
        @test pv.Pvalues[1] < 1e-3
        if j == 1
	        ed = rd.dirs[:, 1]
	        td = [-1, 1]
    	    @test isapprox(ed[2] / ed[1], td[2] / td[1], atol = 0.01, rtol = 0.05)
    	    @test pv.Pvalues[2] > 0.1
            @test abs(rd.eigs[1] / rd.eigs[2]) > 5
    	else
    		@test pv.Pvalues[2] < 1e-3
    	end
    end
end

@testset "PHD1" begin

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

        rd = phd(yy, xx; ndir = 1)

        ed = rd.dirs[:, 1]
        @test isapprox(ed[2] / ed[1], td[2] / td[1], atol = 0.01, rtol = 0.05)
        @test abs(rd.eigs[1] / rd.eigs[2]) > 10
    end
end

@testset "PHD2" begin

    Random.seed!(2142)

    n = 2500     # Sample size
    r = 0.5      # Correlation between variables
    td = [1, -2] # True direction

    cs2 = []
    df2 = []
    for j = 1:100

        # A nonlinear single-index model
        x = randn(n, 2)
        x[:, 2] .= r * x[:, 1] + sqrt(1 - r^2) * x[:, 2]
        lp = x * td
        y = 0.1 * randn(n) + 1 ./ (1.0 .+ lp .^ 2)

        ph = phd(y, x; ndir = 1)

        pv, cs, df = phd_test(ph)
        push!(cs2, cs[2])
        push!(df2, df[2])

    end

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
