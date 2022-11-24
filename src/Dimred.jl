module Dimred

import StatsAPI: fit, RegressionModel

import Statistics, Random

export SlicedInverseRegression,
    PrincipalHessianDirections,
    CORE,
    sir,
    phd,
    core,
    sir_test,
    phd_test,
    slicer,
    eig,
    coef,
    fit,
    mpsir,
    fit!,
    MPSIR

include("sir.jl")
include("phd.jl")
include("core.jl")
include("mpsir.jl")

end
