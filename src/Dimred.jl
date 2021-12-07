module Dimred

import Statistics, Random

include("sir.jl")
include("phd.jl")
include("core.jl")
include("knockoff.jl")

export SlicedInverseRegression,
    CORE, sir, phd, core, sir_test, phd_test, slicer, knockoff_test

end
