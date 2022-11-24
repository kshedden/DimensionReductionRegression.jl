using Dimred, Statistics

include("read.jl")

# Reduce to complete cases
v = [
    :BPXSY1,
    :RIAGENDR,
    :RIDAGEYR,
    :BMXWT,
    :BMXHT,
    :BMXBMI,
    :BMXLEG,
    :BMXARML,
    :BMXARMC,
    :BMXWAIST,
    :BMXHIP,
]
dx = df[:, v]
dx = dx[completecases(dx), :]

# Recode the sex variable to numeric
dx[!, :RIAGENDRx] = replace(dx[:, :RIAGENDR], "F" => 1, "M" => -1)
v = replace(v, :RIAGENDR => :RIAGENDRx)
dx = dx[:, v]

# Center all variables
for m in names(dx)
    if eltype(dx[:, m]) <: Real
        dx[!, m] = dx[:, m] .- mean(dx[:, m])
    end
end

# Get design matrix and response vector
xv = [u for u in v if u != :BPXSY1]
xx = Matrix{Float64}(dx[:, xv])
yy = Vector{Float64}(dx[:, :BPXSY1])

# Sort the cases according to the response variable
ii = sortperm(yy)
yy = yy[ii]
xx = xx[ii, :]

# Fit a model using sliced inverse regression
ms = fit(SlicedInverseRegression, xx, yy)

# Use chi^2 tests for the dimension
pvs = sir_test(ms)

# Fit a model using principal Hessian directions
mp = fit(PrincipalHessianDirections, xx, yy)

# Use chi^2 tests for the dimension
pvp = phd_test(mp)
