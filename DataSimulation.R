
set.seed(123)

## ----------------------------
## Simulation parameters
## ----------------------------
n_patients <- 10000
min_obs <- 3
max_obs <- 100

# Fixed effects
c0 <- 120
c1 <- 0.8   # bmi
c2 <- 0.5   # age

# Random effects covariance (bmi, age)
rho_b <- 0.2
sd_bmi <- 0.5
sd_age <- 0.4
Sigma_b <- matrix(
  c(sd_bmi^2, rho_b * sd_bmi * sd_age,
    rho_b * sd_bmi * sd_age, sd_age^2),
  nrow = 2
)

# Residual SD
sigma_eps <- 5

## ----------------------------
## Simulate per-patient structure
## ----------------------------
n_obs_per_patient <- sample(
  min_obs:max_obs,
  n_patients,
  replace = TRUE
)

patient_id <- rep(seq_len(n_patients), times = n_obs_per_patient)
N <- length(patient_id)

## ----------------------------
## Covariates
## ----------------------------
bmi <- rnorm(N, mean = 27, sd = 4)
age <- rnorm(N, mean = 55, sd = 10)

## ----------------------------
## Random effects
## ----------------------------
b_re <- MASS::mvrnorm(n_patients, mu = c(0, 0), Sigma = Sigma_b)

bmi_re <- b_re[, 1]
age_re <- b_re[, 2]

## Expand random effects to observation level
bmi_re_obs <- bmi_re[patient_id]
age_re_obs <- age_re[patient_id]

## ----------------------------
## Linear predictor + response
## ----------------------------
eta <- c0 +
  (c1 + bmi_re_obs) * bmi +
  (c2 + age_re_obs) * age

sbp <- eta + rnorm(N, sd = sigma_eps)

## ----------------------------
## Assemble data frame
## ----------------------------
dat <- data.frame(
  sbp = sbp,
  bmi = bmi,
  age = age,
  patient = factor(patient_id)
)

rm(patient_id, bmi_re_obs, age_re_obs)  # free memory
