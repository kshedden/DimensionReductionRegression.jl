
@testset "DCovDR check objectrive function and gradient" begin

    rng = StableRNG(123)
    n, p, q, d = 200, 4, 2, 2

    X = randn(rng, n, p)
    Y = randn(rng, n, q)
    Y[:, 1] += X[:, 1] .* (1 .+ X[:, 2]).^2
    Y[:, 2] += X[:, 2] .* (1 .+ X[:, 1]).^2

    m = DCovDR(X, Y; )

    for alpha in [0.5, 1, 1.5]
        for k in 1:5

            C = randn(rng, p, d)
            C, _, _ = svd(C)
            x = C[:]

            function objective(v)
                C = reshape(v, (p, d))
                return DimensionReductionRegression.dcovdr_objective(m, C)
            end

            ng1 = grad(central_fdm(5, 1), objective, x)[1]
            ng = reshape(ng1, (p, d))
            ag = zeros(p, d)
            DimensionReductionRegression.dcovdr_grad!(m, C, ag)

            @test isapprox(ag, ng; rtol=1e-4, atol=1e-4)
        end
    end
end

@testset "DCovDR check fit" begin

    rng = StableRNG(123)
    n, p, q, d = 200, 4, 2, 2

    X = randn(rng, n, p)
    Y = randn(rng, n, q)
    Y[:, 1] += X[:, 1] .* (1 .+ X[:, 2]).^2
    Y[:, 2] += X[:, 2] .* (1 .+ X[:, 1]).^2

    m = fit(DCovDR, X, Y)

    G = zeros(p, 2)
    G[1, 1] = 1
    G[2, 2] = 1

    a = canonical_angles(G, m.dirs)
    @test all(abs.(a) .< 0.1)
end
