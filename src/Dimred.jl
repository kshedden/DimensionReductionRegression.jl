module Dimred

using Distributions: cdf, Beta

import StatsAPI: fit, fit!, coef, response, RegressionModel

import Statistics, Random

export SlicedInverseRegression,
    PrincipalHessianDirections,
    SlicedAverageVarianceEstimation,
    MPSIR,
    CORE,
    core,
    dimension_test,
    coordinate_test,
    slicer,
    eig,
    coef,
    fit,
    mpsir,
    fit!,
    coef,
    response

include("sir.jl")
include("phd.jl")
include("save.jl")
include("core.jl")
include("mpsir.jl")
include("diva.jl")

end
