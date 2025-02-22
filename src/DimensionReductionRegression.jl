module DimensionReductionRegression

using LinearAlgebra
using Distributions: cdf, Beta
using StatsBase: corkendall
using PrettyTables

import GLM
import Base: show
import StatsAPI: fit, fit!, coef, response, RegressionModel, modelmatrix, nobs, HypothesisTest, pvalue, dof

import Statistics, Random

export SlicedInverseRegression,
    PrincipalHessianDirections,
    SlicedAverageVarianceEstimation,
    MPSIR,
    CORE,
    CumulativeSlicingEstimation,
    core,
    dimension_test,
    coordinate_test,
    slicer,
    eig,
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
    dof,

    # Add methods to Base
    show

include("common.jl")
include("sir.jl")
include("phd.jl")
include("save.jl")
include("core.jl")
include("opg.jl")
include("mpsir.jl")
include("diva.jl")
include("cume.jl")
end
