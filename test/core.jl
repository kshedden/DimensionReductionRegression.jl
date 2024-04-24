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
