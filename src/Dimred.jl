module Dimred

import StatsAPI: fit, fit!, coef, RegressionModel

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
	coef

include("sir.jl")
include("phd.jl")
include("save.jl")
include("core.jl")
include("mpsir.jl")

end
