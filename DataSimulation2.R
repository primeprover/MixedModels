set.seed(1239)

## ----------------------------
## Simulation parameters
## ----------------------------
n_patients <- 1000002
min_obs <- 3
max_obs <- 120

## Fixed effects for the two responses
# SBP model: sbp ~ 1 + bmi + age
sbp_intercept <- 120
sbp_bmi <- 0.8
sbp_age <- 0.5

# HbA1c model: hba1c ~ 1 + HDL + age
hba1c_intercept <- 5.6
hba1c_hdl <- -0.03
hba1c_age <- 0.01

## Random effects structure
# Each response uses three correlated random effects for each patient:
# 1. random intercept
# 2. random slope for the response-specific covariate
# 3. random slope for age
# Together, the two responses contribute 6 correlated random effects.

# SBP block variances and correlations
sd_sbp_intercept <- 6.5
sd_sbp_slope <- 0.35
sd_sbp_age <- 0.22

rho_sbp_intercept_slope <- 0.15
rho_sbp_intercept_age <- -0.10
rho_sbp_slope_age <- 0.08

Sigma_sbp <- matrix(
  c(
    sd_sbp_intercept^2,
    rho_sbp_intercept_slope * sd_sbp_intercept * sd_sbp_slope,
    rho_sbp_intercept_age * sd_sbp_intercept * sd_sbp_age,
    rho_sbp_intercept_slope * sd_sbp_intercept * sd_sbp_slope,
    sd_sbp_slope^2,
    rho_sbp_slope_age * sd_sbp_slope * sd_sbp_age,
    rho_sbp_intercept_age * sd_sbp_intercept * sd_sbp_age,
    rho_sbp_slope_age * sd_sbp_slope * sd_sbp_age,
    sd_sbp_age^2
  ),
  nrow = 3,
  byrow = TRUE
)

# HbA1c block variances and correlations
sd_hba1c_intercept <- 4.5
sd_hba1c_slope <- 0.025
sd_hba1c_age <- 0.18

rho_hba1c_intercept_slope <- 0.12
rho_hba1c_intercept_age <- -0.05
rho_hba1c_slope_age <- 0.06

Sigma_hba1c <- matrix(
  c(
    sd_hba1c_intercept^2,
    rho_hba1c_intercept_slope * sd_hba1c_intercept * sd_hba1c_slope,
    rho_hba1c_intercept_age * sd_hba1c_intercept * sd_hba1c_age,
    rho_hba1c_intercept_slope * sd_hba1c_intercept * sd_hba1c_slope,
    sd_hba1c_slope^2,
    rho_hba1c_slope_age * sd_hba1c_slope * sd_hba1c_age,
    rho_hba1c_intercept_age * sd_hba1c_intercept * sd_hba1c_age,
    rho_hba1c_slope_age * sd_hba1c_slope * sd_hba1c_age,
    sd_hba1c_age^2
  ),
  nrow = 3,
  byrow = TRUE
)

# Combine the two blocks into a 6x6 correlated covariance matrix.
# The blocks are linked by modest cross-response correlations.
Sigma_b <- matrix(0, nrow = 6, ncol = 6)
Sigma_b[1:3, 1:3] <- Sigma_sbp
Sigma_b[4:6, 4:6] <- Sigma_hba1c

# Cross-response correlations
Sigma_b[1:3, 4:6] <- matrix(
  c(
    0.10 * sd_sbp_intercept * sd_hba1c_intercept,
    0.04 * sd_sbp_intercept * sd_hba1c_slope,
    -0.03 * sd_sbp_intercept * sd_hba1c_age,
    0.04 * sd_sbp_slope * sd_hba1c_intercept,
    0.06 * sd_sbp_slope * sd_hba1c_slope,
    0.02 * sd_sbp_slope * sd_hba1c_age,
    -0.03 * sd_sbp_age * sd_hba1c_intercept,
    0.02 * sd_sbp_age * sd_hba1c_slope,
    0.05 * sd_sbp_age * sd_hba1c_age
  ),
  nrow = 3,
  byrow = TRUE
)
Sigma_b[4:6, 1:3] <- t(Sigma_b[1:3, 4:6])

## Residual SDs
sigma_sbp <- 8
sigma_hba1c <- 0.35

## ----------------------------
## Simulate per-patient structure
## ----------------------------
n_obs_per_patient <- sample(min_obs:max_obs, n_patients, replace = TRUE)
patient_id <- rep(seq_len(n_patients), times = n_obs_per_patient)
N <- length(patient_id)

## ----------------------------
## Covariates
## ----------------------------
centralise = TRUE
if (centralise) {
  bmi <- rnorm(N, mean = 0, sd = 1)
  age <- rnorm(N, mean = 0, sd = 1)
  hdl <- rnorm(N, mean = 0, sd = 1)
  # Adjust fixed intercepts so the simulated outcomes retain their original macro scales
  sbp_intercept   <- sbp_intercept + (sbp_bmi * 27) + (sbp_age * 55)   # 169.1
  hba1c_intercept <- hba1c_intercept + (hba1c_hdl * 55) + (hba1c_age * 55) # 4.5
} else {
  bmi <- rnorm(N, mean = 27, sd = 4)
  age <- rnorm(N, mean = 55, sd = 10)
  hdl <- rnorm(N, mean = 55, sd = 12)
}

## ----------------------------
## Random effects
## ----------------------------
b_re <- MASS::mvrnorm(n_patients, mu = rep(0, 6), Sigma = Sigma_b)

# SBP response random effects
b_sbp_intercept <- b_re[, 1]
b_sbp_bmi <- b_re[, 2]
b_sbp_age <- b_re[, 3]

# HbA1c response random effects
b_hba1c_intercept <- b_re[, 4]
b_hba1c_hdl <- b_re[, 5]
b_hba1c_age <- b_re[, 6]

# Expand to observation level
b_sbp_intercept_obs <- b_sbp_intercept[patient_id]
b_sbp_bmi_obs <- b_sbp_bmi[patient_id]
b_sbp_age_obs <- b_sbp_age[patient_id]
b_hba1c_intercept_obs <- b_hba1c_intercept[patient_id]
b_hba1c_hdl_obs <- b_hba1c_hdl[patient_id]
b_hba1c_age_obs <- b_hba1c_age[patient_id]

## ----------------------------
## Linear predictors + responses
## ----------------------------
eta_sbp <- sbp_intercept +
  (sbp_bmi + b_sbp_bmi_obs) * bmi +
  (sbp_age + b_sbp_age_obs) * age +
  b_sbp_intercept_obs

eta_hba1c <- hba1c_intercept +
  (hba1c_hdl + b_hba1c_hdl_obs) * hdl +
  (hba1c_age + b_hba1c_age_obs) * age +
  b_hba1c_intercept_obs

sbp <- eta_sbp + rnorm(N, sd = sigma_sbp)
hba1c <- eta_hba1c + rnorm(N, sd = sigma_hba1c)

## ----------------------------
## Assemble data frame
## ----------------------------
dat2 <- data.frame(
  sbp = sbp,
  hba1c = hba1c,
  bmi = bmi,
  age = age,
  HDL = hdl,
  patient = factor(patient_id)
)

rm(patient_id, b_sbp_intercept_obs, b_sbp_bmi_obs, b_sbp_age_obs,
   b_hba1c_intercept_obs, b_hba1c_hdl_obs, b_hba1c_age_obs, b_re)
