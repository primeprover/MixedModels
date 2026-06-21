// reml_loglik.cpp
// C++ implementation of REML log-likelihood for mixed models using RcppEigen
// This file provides a compiled engine for fit_mixed_simple(..., engine = "C++")

#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath>
#include <vector>
#include <atomic>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace Eigen;

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

// ============================================================
// Helper structures and functions
// ============================================================

// [[Rcpp::export]]
void set_threads_cpp(int n_threads) {
#ifdef _OPENMP
    omp_set_num_threads(n_threads);
#endif
}

// Group data structure: precomputed matrices for a single group
struct GroupData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MatrixXd X;      // design matrix for fixed effects
  MatrixXd Z;      // design matrix for random effects
  VectorXd y;      // response vector
  int n;           // number of observations
  MatrixXd ZtZ;    // Z'Z (precomputed)
  VectorXd ZtY;    // Z'y (precomputed)
  MatrixXd ZtX;    // Z'X (precomputed)
};

struct PreparedModel {

  // static scalar parameters
  int p;
  int q;
  int n_corr;

  double lo_sigma;
  double hi_sigma;

  // static data
  VectorXd y;

  // grouped structure
  std::vector<GroupData> groups;
};

// Extract group data from R list
// This converts prepared$group_list (R list) into C++ vector of GroupData
// INPUT: Rcpp::List prepared
// OUTPUT: std::vector<GroupData>
std::vector<GroupData> extract_groups(const Rcpp::List& prepared) {
  Rcpp::List group_list = prepared["group_list"];
  int n_groups = group_list.size();
  std::vector<GroupData> groups(n_groups);

  for (int i = 0; i < n_groups; ++i) {
    Rcpp::List group_i = group_list[i];
    groups[i].X = Rcpp::as<MatrixXd>(group_i["X"]);
    groups[i].Z = Rcpp::as<MatrixXd>(group_i["Z"]);
    groups[i].y = Rcpp::as<VectorXd>(group_i["y"]);
    groups[i].n = Rcpp::as<int>(group_i["n"]);
    groups[i].ZtZ = Rcpp::as<MatrixXd>(group_i["ZtZ"]);
    groups[i].ZtY = Rcpp::as<VectorXd>(group_i["ZtY"]);
    groups[i].ZtX = Rcpp::as<MatrixXd>(group_i["ZtX"]);
  }

  return groups;
}

// [[Rcpp::export]]
Rcpp::XPtr<PreparedModel> init_model_cpp(const Rcpp::List& prepared) {

  auto model = new PreparedModel();

  model->p        = prepared["p"];
  model->q        = prepared["q"];
  model->n_corr   = prepared["n_corr"];
  model->lo_sigma = prepared["lo_sigma"];
  model->hi_sigma = prepared["hi_sigma"];

  model->y = Rcpp::as<VectorXd>(prepared["y"]);

  model->groups = extract_groups(prepared);

  return Rcpp::XPtr<PreparedModel>(model, true);
}

// Build Cholesky factor L from unconstrained parameters
// theta layout: [log_sd_1, ..., log_sd_q, corr_uncon_1, ..., corr_uncon_n_corr, log_sigma_resid]
// Returns L where G = L L' is the random effects covariance matrix
MatrixXd build_cholesky_cpp(const VectorXd& theta, int q) {
  // Extract and exponentiate log standard deviations
  VectorXd log_sds = theta.head(q);
  VectorXd sds = log_sds.array().exp().matrix();
  
  // Number of off-diagonal correlations
  int n_corr = q * (q - 1) / 2;
  
  // Apply tanh to unconstrained correlations to map to [-1, 1]
  VectorXd corrs_unconstrained = theta.segment(q, n_corr);
  VectorXd corrs = corrs_unconstrained.array().tanh().matrix();
  
  // Build lower-triangular correlation matrix L_corr
  MatrixXd L_corr = MatrixXd::Zero(q, q);
  
  int k = 0;
  for (int i = 0; i < q; ++i) {
    double row_prod = 1.0;
    for (int j = 0; j < i; ++j) {
      L_corr(i, j) = corrs(k) * row_prod;
      row_prod *= std::sqrt(1.0 - corrs(k) * corrs(k));
      k++;
    }
    L_corr(i, i) = row_prod;
  }
  
  // Scale rows by standard deviations: L = diag(sds) * L_corr
  // Using Eigen's row-wise scaling operation
  for (int i = 0; i < q; ++i) {
    L_corr.row(i) *= sds(i);
  }
  
  return L_corr;
}

// ============================================================
// Main REML log-likelihood computation
// ============================================================

// [[Rcpp::export]]
Rcpp::List reml_loglik_cpp_impl(
    const Eigen::VectorXd& theta,
    Rcpp::XPtr<PreparedModel> model,
    bool return_penalty,
    bool return_timing
) {
  // Convert theta to Eigen vector
  auto t_start = std::chrono::high_resolution_clock::now();
  //const VectorXd& theta = theta_r;
  const auto& groups = model->groups;
  const VectorXd& y = model->y;
  int p = model->p;
  int q = model->q;
  int n_corr = model->n_corr;
  double lo_sigma = model->lo_sigma;
  double hi_sigma = model->hi_sigma;
  
  int n_groups = groups.size();

  // Helper lambda for bad result
  auto bad_result = [&]() {
    Rcpp::List result;
    result["logLik_unpenalized"] = -1e20;
    result["logLik_penalized"] = -1e20;
    result["penalty"] = NA_REAL;
    if (return_timing && return_penalty) {
      result["timings"] = Rcpp::List::create(
        Rcpp::Named("build_G") = 0.0,
        Rcpp::Named("groups") = 0.0,
        Rcpp::Named("finalize") = 0.0
      );
    }
    return result;
  };
  
  // Extract parameters from theta
  VectorXd log_sds = theta.head(q);
  VectorXd corrs_unconstrained = theta.segment(q, n_corr);
  double log_sigma_resid = theta(q + n_corr);
  
  // Parameter validation
  VectorXd corrs = corrs_unconstrained.array().tanh().matrix();
  if ((corrs.array().abs() > 0.9999999).any()) return bad_result();
  if (log_sigma_resid < std::log(std::numeric_limits<double>::epsilon())) return bad_result();
  
  // Build Cholesky factor L and compute G = L L'
  MatrixXd L = build_cholesky_cpp(theta, q);
  MatrixXd G = L * L.transpose();
  
  // Add jitter for numerical stability
  G.diagonal().array() += 1e-8;
  
  // Compute G_inv using Eigen's LLT Cholesky solver (more stable than direct inversion)
  Eigen::LLT<MatrixXd> llt_G(G);
  if (llt_G.info() != Eigen::Success) return bad_result();
  MatrixXd G_inv = llt_G.solve(MatrixXd::Identity(q, q));
  
  // Compute log-determinant of G from Cholesky factor
  double logdetG = 2.0 * llt_G.matrixLLT().diagonal().array().log().sum();
  
  double sigma = std::exp(log_sigma_resid);
  double sigma2 = sigma * sigma;
  
  // Compute penalty on log-SDs
  double penalty = 0.0;
  for (int i = 0; i < q; ++i) {
    double excess_hi = std::max(0.0, log_sds(i) - hi_sigma);
    double excess_lo = std::max(0.0, lo_sigma - log_sds(i));
    penalty += excess_hi * excess_hi + excess_lo * excess_lo;
  }
  penalty *= 1e-6;
  auto t_after_build = std::chrono::high_resolution_clock::now();
  double t_build = std::chrono::duration<double>(t_after_build - t_start).count();
  
  // Accumulate sufficient statistics across groups
  MatrixXd XtVinvX = MatrixXd::Zero(p, p);
  VectorXd XtVinvy = VectorXd::Zero(p);
  double logdetV = 0.0;
  double quad = 0.0;
  std::atomic<bool> bad(false);

  int n_threads = 1;
#ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp master
    n_threads = omp_get_num_threads();
  }
#endif

  std::vector<MatrixXd> XtVinvX_loc(n_threads, MatrixXd::Zero(p, p));
  std::vector<VectorXd> XtVinvy_loc(n_threads, VectorXd::Zero(p));
  std::vector<double> logdetV_loc(n_threads, 0.0);
  std::vector<double> quad_loc(n_threads, 0.0);

  auto t_loop_start = std::chrono::high_resolution_clock::now();
  //std::cout << "Build to groups time: " << std::chrono::duration<double>(t_loop_start - t_after_build).count() << " seconds" << std::endl;
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int ig = 0; ig < n_groups; ++ig) {
    if (bad.load(std::memory_order_relaxed)) continue;
    const auto& group = groups[ig];

    // Compute K = G_inv + Z_i' Z_i / sigma^2
    MatrixXd K = G_inv + group.ZtZ / sigma2;

    // Cholesky decompose K for efficient solve and log-det
    Eigen::LLT<MatrixXd> llt_K(K);
    if (llt_K.info() != Eigen::Success) {
      bad.store(true, std::memory_order_relaxed);
      continue;
    }

    // Log-determinant of K from Cholesky factor
    double logdetK = 2.0 * llt_K.matrixLLT().diagonal().array().log().sum();
    double t_logdetV = group.n * std::log(sigma2) + logdetK;

    // Solve K^{-1} Z_i' y and K^{-1} Z_i' X
    VectorXd K_inv_Ziy = llt_K.solve(group.ZtY);
    MatrixXd K_inv_Zix = llt_K.solve(group.ZtX);

    // Compute Vi_inv_y and Vi_inv_X (using Woodbury identity implicitly)
    VectorXd Vi_inv_y = group.y / sigma2 - group.Z * K_inv_Ziy / (sigma2 * sigma2);
    MatrixXd Vi_inv_X = group.X / sigma2 - group.Z * K_inv_Zix / (sigma2 * sigma2);

    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif

    XtVinvX_loc[tid] += group.X.transpose() * Vi_inv_X;
    XtVinvy_loc[tid] += group.X.transpose() * Vi_inv_y;
    logdetV_loc[tid] += t_logdetV;
    quad_loc[tid] += group.y.transpose() * Vi_inv_y;
  }

  if (bad.load(std::memory_order_relaxed)) return bad_result();

  for (int t = 0; t < n_threads; ++t) {
    XtVinvX += XtVinvX_loc[t];
    XtVinvy += XtVinvy_loc[t];
    logdetV += logdetV_loc[t];
    quad += quad_loc[t];
  }
  auto t_after_groups = std::chrono::high_resolution_clock::now();
  double t_groups = std::chrono::duration<double>(t_after_groups - t_loop_start).count();
  
  logdetV += n_groups * logdetG;
  
  // Compute XtVinvX^{-1} using Cholesky
  Eigen::LLT<MatrixXd> llt_XtVinvX(XtVinvX);
  if (llt_XtVinvX.info() != Eigen::Success) return bad_result();
  double logdet_XtVinvX = 2.0 * llt_XtVinvX.matrixLLT().diagonal().array().log().sum();
  
  // Solve for beta_hat
  VectorXd beta_hat = llt_XtVinvX.solve(XtVinvy);
  
  // Compute quadratic residual
  double quad_res = quad - beta_hat.transpose() * XtVinvy;
  quad_res = std::max(0.0, quad_res);
  
  // Compute REML log-likelihood
  int n = y.size();
  double ll = -0.5 * (logdetV + logdet_XtVinvX + quad_res + (n - p) * std::log(2.0 * M_PI));
  
  auto t_after_finalize = std::chrono::high_resolution_clock::now();
  double t_finalize = std::chrono::duration<double>(t_after_finalize - t_after_groups).count();

  // Return results
  Rcpp::List result;
  result["logLik_unpenalized"] = ll;
  result["logLik_penalized"] = ll - penalty;
  result["penalty"] = penalty;
  if (return_timing && return_penalty) {
    result["timings"] = Rcpp::List::create(
      Rcpp::Named("build_G") = t_build,
      Rcpp::Named("groups") = t_groups,
      Rcpp::Named("finalize") = t_finalize,
      Rcpp::Named("total") = 0 + t_build + t_groups + t_finalize
    );
  }
  
  return result;
}
