rm(list=ls())
library(nlme)
library(lme4)

source("MixedModel.R")

source("DataSimulation.R")



fit_mixed(
  data    = dat,
  engine  = "C++",
  responses = list(
    list(
      formula = sbp ~ bmi + age,
      random  = ~ 0 + bmi + age | patient,
      response_type = "homo_normal"
    )
  )
)

fit_mixed(
  data    = dat,
  engine  = "C++",
  responses = list(
    list(
      formula = sbp ~ bmi + age,
      random  = ~ 0 + bmi + age | patient,
      var_pred = ~ 1,
      response_type = "hetero_normal"
    )
  )
)

# Check that these produce the same log-likelihood as the previous calls and each other. optim_value = 16037958 for current models. 