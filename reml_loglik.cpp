// reml_loglik.cpp
// C++ implementation of REML log-likelihood for mixed models using RcppEigen
// This file provides a compiled engine for fit_mixed_simple(..., engine = "C++")


#include "reml_loglik.hpp"
#include "response.hpp"
#include "homo_normal_response.hpp"
#include "hetero_normal_response.hpp"
#include "response_factory.hpp"
#include <cmath>
#include <vector>
#include <atomic>
#include <chrono>
#include <memory>


// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

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
    groups[i].Z = Rcpp::as<MatrixXd>(group_i["Z"]);
    groups[i].n = Rcpp::as<int>(group_i["n"]);
    groups[i].ZtZ = Rcpp::as<MatrixXd>(group_i["ZtZ"]);
  }

  return groups;
}

PreparedModel::~PreparedModel() = default;

// [[Rcpp::export]]
Rcpp::XPtr<PreparedModel> init_model_cpp(const Rcpp::List& prepared) {
  auto model = new PreparedModel();

  model->q        = prepared["q"];
  model->n_corr   = prepared["n_corr"];
  model->lo_sigma = prepared["lo_sigma"];
  model->hi_sigma = prepared["hi_sigma"];
  model->residual_variance_penalty_weight =
      prepared.containsElementNamed("residual_variance_penalty_weight")
          ? Rcpp::as<double>(prepared["residual_variance_penalty_weight"])
          : 0.0;
  if (model->residual_variance_penalty_weight < 0.0)
    Rcpp::stop("residual_variance_penalty_weight must be non-negative.");
  model->groups = extract_groups(prepared);
  model->responses.push_back(make_response(prepared));
  model->responses.back()->initialise_from_prepared(prepared);
  model->responses.back()->initialise_working();
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

double compute_penalty(
  const PreparedModel& model,  
  const VectorXd& log_sds,
  const ResponseModel& response
    ) {

  double penalty = 0.0;
  for (int i = 0; i < model.q; ++i) {
    double excess_hi = std::max(0.0, log_sds(i) - model.hi_sigma);
    double excess_lo = std::max(0.0, model.lo_sigma - log_sds(i));
    penalty += excess_hi * excess_hi + excess_lo * excess_lo;
    penalty += log_sds(i) * 1000000.0; // weak penalty to provide slope for flat regions (assuming penalty multiplied by 1e-6)
  }
  penalty *= 1e-6;
  penalty += model.residual_variance_penalty_weight *
      response.residual_variance_penalty();
  //std::cout << "Penalty: " << penalty << " Lo_sigma: " << lo_sigma << " Hi_sigma: " << hi_sigma << std::endl;
  return penalty;
}

//__attribute__((noinline))
bool accumulate_groups(
  const PreparedModel& model,
  const MatrixXd& G_inv,
  const double& sigma2,
  MatrixXd& XtVinvX,
  VectorXd& XtVinvy,
  double& logdetV,
  double& quad
) {
  int n_threads = 1;
  Eigen::setNbThreads(1);
#ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp master
    n_threads = omp_get_num_threads();
  }
#endif

  const auto& groups = model.groups;
  const auto& response = *model.responses.front();

  // std::vector<MatrixXd> XtVinvX_loc(n_threads, MatrixXd::Zero(model.p, model.p));
  // std::vector<VectorXd> XtVinvy_loc(n_threads, VectorXd::Zero(model.p));
  // std::vector<double> logdetV_loc(n_threads, 0.0);
  // std::vector<double> quad_loc(n_threads, 0.0);
  std::atomic<bool> bad(false);

  const double inv_sigma2    = 1.0 / sigma2;
  const double inv_sigma2_sq = inv_sigma2 * inv_sigma2;

  const int p = XtVinvX.rows();

  #ifdef _OPENMP
  #pragma omp parallel
  #endif
  {
      const int tid =
  #ifdef _OPENMP
          omp_get_thread_num();
  #else
          0;
  #endif

      MatrixXd K = G_inv;
      MatrixXd K_inv_Zix;
      VectorXd K_inv_Ziy;
      MatrixXd Vi_inv_X;
      VectorXd Vi_inv_y;

      MatrixXd XtVinvX_local = MatrixXd::Zero(p, p);
      VectorXd XtVinvy_local = VectorXd::Zero(p);
      double   logdetV_local = 0.0;
      double   quad_local    = 0.0;

      Eigen::LLT<MatrixXd, Eigen::Lower> llt;

  #ifdef _OPENMP
  #pragma omp for schedule(guided)
  #endif
      for (int ig = 0; ig < (int)groups.size(); ++ig) {
          if (bad.load(std::memory_order_relaxed)) continue;

          const auto& group = groups[ig];
          const auto& response_group = response.groups[ig];

          K.noalias() = G_inv;
          K.noalias() += group.ZtZ * inv_sigma2;

          llt.compute(K);
          if (llt.info() != Eigen::Success) {
              bad.store(true, std::memory_order_relaxed);
              continue;
          }

          const auto& L = llt.matrixLLT();
          const double logdetK   = 2.0 * L.diagonal().array().log().sum();
          const double t_logdetV = group.n * std::log(sigma2) + logdetK;

          K_inv_Ziy = llt.solve(response_group.ZtY);
          K_inv_Zix = llt.solve(response_group.ZtX);

          Vi_inv_y = response_group.y * inv_sigma2;
          Vi_inv_y.noalias() -= group.Z * K_inv_Ziy * inv_sigma2_sq;

          Vi_inv_X = response_group.X * inv_sigma2;
          Vi_inv_X.noalias() -= group.Z * K_inv_Zix * inv_sigma2_sq;

          XtVinvX_local.noalias() += response_group.X.transpose() * Vi_inv_X;
          XtVinvy_local.noalias() += response_group.X.transpose() * Vi_inv_y;
          logdetV_local += t_logdetV;
          quad_local    += response_group.y.dot(Vi_inv_y);
      }

  #ifdef _OPENMP
  #pragma omp critical
  #endif
      {
          XtVinvX += XtVinvX_local;
          XtVinvy += XtVinvy_local;
          logdetV += logdetV_local;
          quad    += quad_local;
      }
  }


  if (bad.load(std::memory_order_relaxed)) return false;

  // for (int t = 0; t < n_threads; ++t) {
  //   XtVinvX += XtVinvX_loc[t];
  //   XtVinvy += XtVinvy_loc[t];
  //   logdetV += logdetV_loc[t];
  //   quad += quad_loc[t];
  // }
  return true;
}

inline bool random_effects_compute(
    const GroupData& group,
    const MatrixXd& G_inv,
    GroupWorkspace& ws)
{
    ws.ZtRinvZ.noalias() =
        group.Z.transpose() * ws.RinvZ;

    ws.ZtRinvX.noalias() =
        group.Z.transpose() * ws.RinvX;

    ws.ZtRinvy.noalias() =
        group.Z.transpose() * ws.Rinvy;

    ws.K = G_inv;
    ws.K.noalias() += ws.ZtRinvZ;

    ws.llt.compute(ws.K);

    if (ws.llt.info() != Eigen::Success)
      return false;

    ws.logdetK =
        2.0 *
        ws.llt.matrixLLT().diagonal().array().log().sum();

    ws.KinvZtRinvX =
        ws.llt.solve(ws.ZtRinvX);

    ws.KinvZtRinvy =
        ws.llt.solve(ws.ZtRinvy);

    ws.VinvX.noalias() =
        ws.RinvX -
        ws.RinvZ * ws.KinvZtRinvX;

    ws.Vinvy.noalias() =
        ws.Rinvy -
        ws.RinvZ * ws.KinvZtRinvy;
    
    return true;
}

bool accumulate_groups_response(
    const PreparedModel& model,
    const ResponseModel& response,
    const MatrixXd& G_inv,
    MatrixXd& XtVinvX,
    VectorXd& XtVinvy,
    double& logdetV,
    double& quad
)
{
    Eigen::setNbThreads(1);

    const auto& groups = model.groups;
    const int p = response.groups.front().X.cols();

    std::atomic<bool> bad(false);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        MatrixXd XtVinvX_local = MatrixXd::Zero(p, p);
        VectorXd XtVinvy_local = VectorXd::Zero(p);

        double logdetV_local = 0.0;
        double quad_local = 0.0;

        GroupWorkspace workspace;

#ifdef _OPENMP
#pragma omp for schedule(guided)
#endif
        for (int ig = 0; ig < (int)groups.size(); ++ig) {

            const auto& group = groups[ig];
            const auto& response_group = response.groups[ig];

            response.compute_group(group, response_group, ig, workspace);

            if (!random_effects_compute(group, G_inv, workspace))
            {
                bad.store(true, std::memory_order_relaxed);
                continue;
            }
            
            // ----------------------------------------------------
            // accumulate
            // ----------------------------------------------------


            XtVinvX_local.noalias() += response_group.X.transpose() * workspace.VinvX;
            XtVinvy_local.noalias() += response_group.X.transpose() * workspace.Vinvy;

            quad_local += response_group.y.dot(workspace.Vinvy);

            // group contribution:
            // log|R| + log|G^{-1}+Z'R^{-1}Z|
            logdetV_local += workspace.logdetR + workspace.logdetK;
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        {
            XtVinvX += XtVinvX_local;
            XtVinvy += XtVinvy_local;
            logdetV += logdetV_local;
            quad += quad_local;
        }
    }

    return !bad.load();
}

// ============================================================
// Main REML log-likelihood computation
// ============================================================

// [[Rcpp::export]]
Rcpp::List reml_loglik_cpp_impl(
    Rcpp::XPtr<PreparedModel> model_xptr,
    const Eigen::VectorXd& theta,
    bool return_penalty,
    bool return_timing
) {
  PreparedModel& model = *model_xptr;
  // Convert theta to Eigen vector
  auto t_start = std::chrono::high_resolution_clock::now();
  //const VectorXd& theta = theta_r;
  const auto& groups = model.groups;
  if (model.responses.size() != 1) {
    Rcpp::stop("The current REML accumulator supports one response.");
  }
  ResponseModel& response = *model.responses.front();
  const int p = response.groups.front().X.cols();
  int q = model.q;
  int n_corr = model.n_corr;
  const int response_offset = q + n_corr;
  if (theta.size() != response_offset + response.parameter_count()) {
    Rcpp::stop("theta has the wrong number of response parameters.");
  }
  VectorXd log_sds = theta.head(q);
  VectorXd corrs_unconstrained = theta.segment(q, n_corr);
  VectorXd response_parameters = theta.segment(
      response_offset, response.parameter_count());
  response.update_working(response_parameters);
  //double lo_sigma = model.lo_sigma;
  //double hi_sigma = model.hi_sigma;

  //int n_groups = groups.size();
  
  // Extract parameters from theta
  
  // Parameter validation
  VectorXd corrs = corrs_unconstrained.array().tanh().matrix();
  if ((corrs.array().abs() > 0.9999999).any()) return bad_result();
  if (!response.parameters_are_valid(response_parameters))
    return bad_result();
  
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
  
  // Calculate random effects penalties for flat regions and extreme values
  double penalty = compute_penalty(model, log_sds, response);

  auto t_after_build = std::chrono::high_resolution_clock::now();
  double t_build = std::chrono::duration<double>(t_after_build - t_start).count();
  
  // Accumulate sufficient statistics across groups
  MatrixXd XtVinvX = MatrixXd::Zero(p, p);
  VectorXd XtVinvy = VectorXd::Zero(p);
  double logdetV = 0.0;
  double quad = 0.0;

  auto t_loop_start = std::chrono::high_resolution_clock::now();
  bool success = accumulate_groups_response(
      model, response, G_inv, XtVinvX, XtVinvy, logdetV, quad);
  if (!success) return bad_result();
  auto t_after_groups = std::chrono::high_resolution_clock::now();
  double t_groups = std::chrono::duration<double>(t_after_groups - t_loop_start).count();
  
  //if (!model.heteroscedastic)
    logdetV += groups.size() * logdetG;
  
  // Compute XtVinvX^{-1} using Cholesky
  Eigen::LLT<MatrixXd> llt_XtVinvX(XtVinvX);
  if (llt_XtVinvX.info() != Eigen::Success) return bad_result();
  double logdet_XtVinvX = 2.0 * llt_XtVinvX.matrixLLT().diagonal().array().log().sum();
  
  // Solve for beta_hat
  VectorXd beta_hat = llt_XtVinvX.solve(XtVinvy);
  
  // Compute quadratic residual
  double quad_res = 0.0;
  
  quad_res = quad - beta_hat.transpose() * XtVinvy;
  quad_res = std::max(0.0, quad_res);
  
  // Compute REML log-likelihood
  int n = 0;
  for (const auto& group : response.groups) n += group.y.size();
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

// [[Rcpp::export]]
VectorXd compute_beta_hat_cpp(
    const Eigen::VectorXd& theta,
    Rcpp::XPtr<PreparedModel> model
) {
    PreparedModel& prepared = *model;
    if (prepared.responses.size() != 1)
      Rcpp::stop("The current beta calculation supports one response.");

    ResponseModel& response = *prepared.responses.front();
    const int response_offset = prepared.q + prepared.n_corr;
    if (theta.size() != response_offset + response.parameter_count())
      Rcpp::stop("theta has the wrong number of response parameters.");
    response.update_working(theta.segment(
        response_offset, response.parameter_count()));

    MatrixXd L = build_cholesky_cpp(theta, prepared.q);
    MatrixXd G = L * L.transpose();
    G.diagonal().array() += 1e-8;
    Eigen::LLT<MatrixXd> llt_G(G);
    if (llt_G.info() != Eigen::Success)
      Rcpp::stop("Could not factor random-effects covariance matrix.");
    MatrixXd G_inv = llt_G.solve(MatrixXd::Identity(prepared.q, prepared.q));

    const int p = response.groups.front().X.cols();
    MatrixXd XtVinvX = MatrixXd::Zero(p, p);
    VectorXd XtVinvy = VectorXd::Zero(p);
    double logdetV = 0.0;
    double quad = 0.0;
    if (!accumulate_groups_response(
            prepared, response, G_inv, XtVinvX, XtVinvy, logdetV, quad))
      Rcpp::stop("Could not accumulate response groups.");

    Eigen::LLT<MatrixXd> llt_XtVinvX(XtVinvX);
    if (llt_XtVinvX.info() != Eigen::Success)
      Rcpp::stop("Could not factor fixed-effects information matrix.");
    return llt_XtVinvX.solve(XtVinvy);
}
