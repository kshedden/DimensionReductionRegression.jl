using LinearAlgebra

module Dimred

    export SlicedInverseRegression, CORE, sir, phd, core

    include("sir.jl")
    include("phd.jl")
    include("core.jl")

end

