module DimensionReductionRegression

using Distributions: cdf, Beta

import GLM

import StatsAPI: fit, fit!, coef, response, RegressionModel, modelmatrix, nobs, HypothesisTest, pvalue, dof

import Statistics, Random

export SlicedInverseRegression,
    PrincipalHessianDirections,
    SlicedAverageVarianceEstimation,
    MPSIR,
    CORE,
    core,
    dimension_test,
    coordinate_test,
    coordinate_test_resid,
    slicer,
    eig,
    mpsir,
    OPG,
    teststat,

    # Add methods to StatsAPI
    coef,
    nvar,
    fit,
    fit!,
    coef,
    response,
    modelmatrix,
    pvalue,
    dof

include("common.jl")
include("sir.jl")
include("phd.jl")
include("save.jl")
include("core.jl")
include("opg.jl")
include("mpsir.jl")
include("diva.jl")

end
