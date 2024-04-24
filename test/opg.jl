@testset "OPG linear" begin

    rng = StableRNG(123)
    n = 1000
    p = 5
    X, y = gendat_quadratic(n, p, rng)
    ct = zeros(p, 2)
    ct[:, 1] = [1, -1, 0, 0, 0]
    ct[:, 2] = [0, 0, 1, 1, 0]

    for n_centers in [-1, 100]
        opg = OPG(X, y, Normal(); n_centers=n_centers)
        fit!(opg; bw=sqrt(p), ndir=2)
        c = coef(opg)
        aa = canonical_angles(c, ct)
        @test maximum(aa) < 0.07
    end
end
