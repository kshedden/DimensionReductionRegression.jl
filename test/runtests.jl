using Dimred, Test, Random, LinearAlgebra

@testset "SIR1" begin

    Random.seed!(2142)

    n = 25002
    r = 0.5

    for j in 1:4
        x = randn(n, 2)
        x[:, 2] .= r*x[:, 1] + sqrt(1-r^2)*x[:, 2]
        y = 0.1*randn(n) + x * [1, -2]

        if j < 3
            xx = x
        else
            xx = convert(Array{Float32}, x)
        end

        if rem(j, 2) == 1
            yy = y
        else
            yy = convert(Array{Float32}, y)
        end

        rd = sir(yy, xx; ndir=1)

        @test isapprox(rd.dirs[2, 1] / rd.dirs[1, 1], -2, atol=0.01, rtol=0.05)
        @test isapprox(rd.eigs, [1, 0], atol=0.15)

    end
end

@testset "PHD" begin

    Random.seed!(2142)

    n = 25002
    r = 0.5

    for j in 1:4
        x = randn(n, 2)
        x[:, 2] .= r*x[:, 1] + sqrt(1-r^2)*x[:, 2]
        lp = x * [1, -2]
        y = 0.1*randn(n) + 1 ./ (1.0 .+ lp.^2)

        if j < 3
            xx = x
        else
            xx = convert(Array{Float32}, x)
        end

        if rem(j, 2) == 1
            yy = y
        else
            yy = convert(Array{Float32}, y)
        end

        rd = phd(yy, xx; ndir=1)

        @test isapprox(rd.dirs[2, 1] / rd.dirs[1, 1], -2, atol=0.01, rtol=0.05)
        @test abs(rd.eigs[1] / rd.eigs[2]) > 50
    end
end


@testset "CORE" begin

    Random.seed!(4543)

    p = 10
    n = 5
    covs = Array{Array{Float64, 2}}(undef, n)
    ns = Array{Int64}(undef, n)
    q = 3

    # The first q x q block is varying, the rest of the matrix
    # is constant
    for i in 1:n
        m = Array(Diagonal(ones(p)))
        r = 1/(i+1)
        for j1 in 1:q
            for j2 in 1:q
                m[j1, j2] = r^abs(j1-j2)
            end
        end
        covs[i] = m
        ns[i] = 20 + 2*i
    end

    # Start at the truth
    params = zeros(p, 3)
    params[1:3, 1:3] .= I(3)
    b = core(covs, ns, ndim=3, maxiter=100, params=params)

    # Start at a random point
    for k in 1:10
        params = randn(p, 3)
        s = svd(params)
        params = s.U
        c = core(covs, ns, params=params, maxiter=2000, ndim=3)

        # Check that the projection matrices are the same
        @test isapprox(b.dirs * transpose(b.dirs), c.dirs * transpose(c.dirs),
                       atol=0.01, rtol=0.01)

        # Check that the log-likelihood is about the same
        @test b.llf - c.llf < 0.001
    end

end

