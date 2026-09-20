set.seed(1234)

## ----------------------------
## Simulation parameters (preserving your obs structure)
## ----------------------------
n_patients <- 100
min_obs    <- 20
max_obs    <- 40

## Fixed effects
sbp_intercept   <- 1; sbp_bmi       <- 0.8; sbp_age     <- 0.5
hba1c_intercept <- 0.5; hba1c_hdl   <- -0.08; hba1c_age <- 0.04

## Random effect SDs (minor HDL slope bump for numerical stability)
sd_sbp_intercept    <- 4.5; sd_sbp_slope      <- 1.35; sd_sbp_age        <- 0.82
sd_hba1c_intercept  <- 2.5; sd_hba1c_slope    <- 1.28; sd_hba1c_age      <- 0.98

## Within-response correlations (exactly as you specified)
#rho_sbp_int_slope   <- 0.15; rho_sbp_int_age     <- -0.10; rho_sbp_slope_age       <- 0.08
#rho_hba1c_int_slope <- 0.12; rho_hba1c_int_age    <- -0.05; rho_hba1c_slope_age      <- 0.06
rho_sbp_int_slope   <- 0.25; rho_sbp_int_age     <- -0.20; rho_sbp_slope_age       <- 0.18
rho_hba1c_int_slope <- 0.22; rho_hba1c_int_age    <- -0.15; rho_hba1c_slope_age      <- 0.16

Sigma_sbp <- matrix(
  c(sd_sbp_intercept^2, 
    rho_sbp_int_slope * sd_sbp_intercept * sd_sbp_slope,
    rho_sbp_int_age   * sd_sbp_intercept * sd_sbp_age,
    rho_sbp_int_slope * sd_sbp_intercept * sd_sbp_slope,
    sd_sbp_slope^2, 
    rho_sbp_slope_age * sd_sbp_slope * sd_sbp_age,
    rho_sbp_int_age   * sd_sbp_intercept * sd_sbp_age,
    rho_sbp_slope_age * sd_sbp_slope * sd_sbp_age,
    sd_sbp_age^2), nrow = 3, byrow = TRUE)

Sigma_hba1c <- matrix(
  c(sd_hba1c_intercept^2, 
    rho_hba1c_int_slope * sd_hba1c_intercept * sd_hba1c_slope,
    rho_hba1c_int_age   * sd_hba1c_intercept * sd_hba1c_age,
    rho_hba1c_int_slope * sd_hba1c_intercept * sd_hba1c_slope,
    sd_hba1c_slope^2, 
    rho_hba1c_slope_age * sd_hba1c_slope * sd_hba1c_age,
    rho_hba1c_int_age   * sd_hba1c_intercept * sd_hba1c_age,
    rho_hba1c_slope_age * sd_hba1c_slope * sd_hba1c_age,
    sd_hba1c_age^2), nrow = 3, byrow = TRUE)

## Cross-response block (weak correlations preserved exactly)
Sigma_b <- matrix(0, 6, 6)
Sigma_b[1:3, 1:3] <- Sigma_sbp
Sigma_b[4:6, 4:6] <- Sigma_hba1c

cc = c(0.10, 0.04, -0.03, 0.04, 0.06, 0.02, -0.03, 0.02, 0.05) * 5.0

cross_covs <- matrix(
  c(cc[1]*sd_sbp_intercept*sd_hba1c_intercept, cc[2]*sd_sbp_intercept*sd_hba1c_slope, cc[3]*sd_sbp_intercept*sd_hba1c_age,
    cc[4]*sd_sbp_slope*sd_hba1c_intercept,      cc[5]*sd_sbp_slope*sd_hba1c_slope,      cc[6]*sd_sbp_slope*sd_hba1c_age,
    cc[7]*sd_sbp_age*sd_hba1c_intercept,        cc[8]*sd_sbp_age*sd_hba1c_slope,        cc[9]*sd_sbp_age*sd_hba1c_age),
  nrow = 3, byrow = TRUE)

Sigma_b[1:3, 4:6] <- cross_covs
Sigma_b[4:6, 1:3] <- t(cross_covs)

## Residual SDs (restored to original values)
sigma_sbp   <- 4.0
sigma_hba1c <- 2.5

## ----------------------------
## Simulate data
## ----------------------------
n_obs_per_patient <- sample(min_obs:max_obs, n_patients, replace = TRUE)
patient_id        <- rep(seq_len(n_patients), times = n_obs_per_patient)
N                 <- length(patient_id)

bmi   <- rnorm(N, mean = 0, sd = 1)
age   <- rnorm(N, mean = 0, sd = 1)
hdl   <- rnorm(N, mean = 0, sd = 1)

b_re    <- MASS::mvrnorm(n_patients, mu = rep(0, 6), Sigma = Sigma_b)

b_sbp_int_obs     <- b_re[patient_id, 1]; b_sbp_bmi_obs   <- b_re[patient_id, 2]; b_sbp_age_obs      <- b_re[patient_id, 3]
b_hba1c_int_obs   <- b_re[patient_id, 4]; b_hba1c_hdl_obs <- b_re[patient_id, 5]; b_hba1c_age_obs    <- b_re[patient_id, 6]

eta_sbp     <- sbp_intercept + (sbp_bmi      + b_sbp_bmi_obs)   * bmi   + 
              (sbp_age       + b_sbp_age_obs)* age             +  
               b_sbp_int_obs
eta_hba1c   <- hba1c_intercept + (hba1c_hdl  + b_hba1c_hdl_obs)* hdl    + 
              (hba1c_age     + b_hba1c_age_obs)* age           + 
               b_hba1c_int_obs

sbp       <- eta_sbp      + rnorm(N, sd = sigma_sbp)
hba1c     <- eta_hba1c    + rnorm(N, sd = sigma_hba1c)

dat2   <- data.frame(sbp=sbp, hba1c=hba1c, bmi=bmi, age=age, HDL=hdl, patient=factor(patient_id))
rm(b_sbp_int_obs, b_sbp_bmi_obs, b_sbp_age_obs, 
  b_hba1c_int_obs, b_hba1c_hdl_obs, b_hba1c_age_obs, 
  b_re, n_obs_per_patient, patient_id, N)

## ============================================================
## EXACT TRUTH VECTOR - MATCHES C++ LOWER-TRIANGULAR PACKING ORDER
## CRITICAL: C++ uses a partial-correlation Cholesky parameterization.
## We extract partial correlations from Sigma_b to match the optimizer's space.
## ============================================================

q_global <- 6
sd_vec   <- c(sd_sbp_intercept, sd_sbp_slope, sd_sbp_age,
              sd_hba1c_intercept, sd_hba1c_slope, sd_hba1c_age)

# Convert full covariance matrix to packed partial correlations (matching build_cholesky.cpp)
cov_to_packed_partial_corrs <- function(Sigma) {
  q <- nrow(Sigma)
  sds <- sqrt(diag(Sigma))
  R_mat <- diag(1/sds) %*% Sigma %*% diag(1/sds) # Marginal correlation matrix
  
  L <- t(chol(R_mat)) # Lower triangular Cholesky: R = L L^T
  
  packed_pc <- numeric(q * (q - 1) / 2)
  k <- 1
  for (i in 2:q) {
    row_prod <- 1.0
    for (j in 1:(i - 1)) {
      # Partial correlation corresponding to L[i,j]
      packed_pc[k] <- L[i, j] / row_prod
      row_prod <- row_prod * sqrt(1.0 - packed_pc[k]^2)
      k <- k + 1
    }
  }
  return(packed_pc)
}

true_partial_corrs <- cov_to_packed_partial_corrs(Sigma_b)

# Assemble theta vector exactly as C++ expects:
# [1..6] log SDs | [7..21] atanh(partial correlations) | [22..23] log(residual variances)
true_theta <- c(
  log(sd_vec),                          # 1-6: log SDs
  atanh(true_partial_corrs),            # 7-21: unconstrained partial correlations
  2 * log(sigma_sbp),                   # 22: log(var_sbp)
  2 * log(sigma_hba1c)                  # 23: log(var_hba1c)
)

# Pretty print for direct comparison with fit$joint_fit$opt_par
param_labels <- c(
  paste0("log_sd[", 1:6, "]"),
  paste0("atanh_pc[", 7:21, "]"),
  "log_var_sbp", "log_var_hba1c"
)

truth_tbl <- data.frame(Param = param_labels, True_Value = round(true_theta, 6))
print(truth_tbl, row.names = FALSE)

## Save for later comparison
saveRDS(true_theta, "truth_theta_23.rds")
