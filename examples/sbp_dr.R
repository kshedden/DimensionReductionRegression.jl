library(dplyr)
library(ggplot2)
library(readr)
#library(dr)
source("../test/dr/R/dr.R")
source("read.R")

# Reduce to complete cases
dx = df %>% dplyr::select(BPXSY1, RIAGENDR, RIDAGEYR, BMXWT, BMXHT, BMXBMI, BMXLEG, BMXARML, BMXARMC, BMXWAIST, BMXHIP)
dx = dx[complete.cases(dx),]

# Recode sex to numeric
dx$RIAGENDRx = recode(dx$RIAGENDR, "F"=1, "M"=-1)

# Center all variables
for (m in names(dx)) {
    if (is.numeric(dx[[m]])) {
        dx[[m]] = scale(dx[[m]], scale=F)
    }
}

# Fit a model using sliced inverse regression
ms = dr(BPXSY1 ~ RIAGENDRx + RIDAGEYR + BMXWT + BMXHT + BMXBMI + BMXLEG + BMXARML + BMXARMC + BMXWAIST + BMXHIP, dx)

# Use chi^2 tests for the dimension
print(dr.test(ms))

# Fit a model using principal Hessian directions
mp = dr(BPXSY1 ~ RIAGENDRx + RIDAGEYR + BMXWT + BMXHT + BMXBMI + BMXLEG + BMXARML + BMXARMC + BMXWAIST + BMXHIP, dx, method="phd")

# Use chi^2 tests for the dimension
print(dr.test(mp))

# Fit a model using sliced average variance estimation
ma = dr(BPXSY1 ~ RIAGENDRx + RIDAGEYR + BMXWT + BMXHT + BMXBMI + BMXLEG + BMXARML + BMXARMC + BMXWAIST + BMXHIP, dx, method="save")

# Use chi^2 tests for the dimension
print(dr.test(ma))
