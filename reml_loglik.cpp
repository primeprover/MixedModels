// reml_loglik.cpp
// C++ implementation of REML log-likelihood for mixed models using RcppEigen
// This file provides a compiled engine for fit_mixed_simple(..., engine = "C++")


#include "reml_loglik.hpp"
#include "Response.hpp"
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
    groups[i].X = Rcpp::as<MatrixXd>(group_i["X"]);
    groups[i].Z = Rcpp::as<MatrixXd>(group_i["Z"]);
    groups[i].y = Rcpp::as<VectorXd>(group_i["y"]);
    groups[i].n = Rcpp::as<int>(group_i["n"]);
    groups[i].ZtZ = Rcpp::as<MatrixXd>(group_i["ZtZ"]);
    groups[i].ZtY = Rcpp::as<VectorXd>(group_i["ZtY"]);
    groups[i].ZtX = Rcpp::as<MatrixXd>(group_i["ZtX"]);
    if (prepared["heteroscedastic"]) {
      groups[i].W = Rcpp::as<MatrixXd>(group_i["W"]);
    }
  }

  return groups;
}

WorkingModel initialise_working_model(
    const PreparedModel& model
)
{
    WorkingModel working;

    working.groups.resize(model.groups.size());
    for (int g = 0; g < (int)model.groups.size(); ++g) {
        const int n = model.groups[g].n;

        working.groups[g].eta.resize(n);
        working.groups[g].inv_resid_var.resize(n);
    }
    return working;
}

// [[Rcpp::export]]
Rcpp::XPtr<PreparedModel> init_model_cpp(const Rcpp::List& prepared) {
  auto model = new PreparedModel();

  model->p        = prepared["p"];
  model->q        = prepared["q"];
  model->n_corr   = prepared["n_corr"];
  model->lo_sigma = prepared["lo_sigma"];
  model->hi_sigma = prepared["hi_sigma"];
  model->heteroscedastic = prepared["heteroscedastic"];

  model->y = Rcpp::as<VectorXd>(prepared["y"]);
  model->groups = extract_groups(prepared);
  std::cout << "Data prepared apart from working" << std::endl;
  //_sleep(1000); // Sleep for 1 second to allow time for the message to be viewed
  model->working = initialise_working_model(*model);
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
  const VectorXd& gamma,
  const double log_sigma_resid
    ) {

  // Compute penalty on log-SDs
  double penalty = 0.0;
  if (model.heteroscedastic) {
    for (const auto& g : model.working.groups)
      penalty -= g.eta.squaredNorm();
    penalty *= 10.0; // weak penalty to provide slope for flat regions (assuming penalty multiplied by 1e-6)
  }
  else {
    for (const auto& g : model.groups)
    //  penalty -= g.n * std::exp(2.0 * log_sigma_resid);
      penalty -= g.n * 4.0 * log_sigma_resid * log_sigma_resid;
    //std::cout << "Penalty from log_sigma_resid: " << penalty << std::endl;
    penalty *= 10.0; // weak penalty to provide slope for flat regions (assuming penalty multiplied by 1e-6)
  }
  //std::cout << "Penalty from residuals: " << penalty << std::endl;
  // TODO: fix above penalty to be the same in both cases (heteroscedastic and homoscedastic)
  // Maybe it should be on the parameters
  penalty = 0.0;
  for (int i = 0; i < model.q; ++i) {
    double excess_hi = std::max(0.0, log_sds(i) - model.hi_sigma);
    double excess_lo = std::max(0.0, model.lo_sigma - log_sds(i));
    penalty += excess_hi * excess_hi + excess_lo * excess_lo;
    penalty += log_sds(i) * 1000000.0; // weak penalty to provide slope for flat regions (assuming penalty multiplied by 1e-6)
  }
  //penalty += gamma.squaredNorm() * 100000000.0; // weak penalty to provide slope for flat regions (assuming penalty multiplied by 1e-6)
  penalty *= 1e-6;
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
  const VectorXd& y = model.y;

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

          K_inv_Ziy = llt.solve(group.ZtY);
          K_inv_Zix = llt.solve(group.ZtX);

          Vi_inv_y = group.y * inv_sigma2;
          Vi_inv_y.noalias() -= group.Z * K_inv_Ziy * inv_sigma2_sq;

          Vi_inv_X = group.X * inv_sigma2;
          Vi_inv_X.noalias() -= group.Z * K_inv_Zix * inv_sigma2_sq;

          XtVinvX_local.noalias() += group.X.transpose() * Vi_inv_X;
          XtVinvy_local.noalias() += group.X.transpose() * Vi_inv_y;
          logdetV_local += t_logdetV;
          quad_local    += group.y.dot(Vi_inv_y);
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

bool accumulate_groups_hetero(
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
    const int p = model.p;

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

            response.compute_group(group, model.working.groups[ig], workspace);

            if (!random_effects_compute(group, G_inv, workspace))
            {
                bad.store(true, std::memory_order_relaxed);
                continue;
            }
            
            // ----------------------------------------------------
            // accumulate
            // ----------------------------------------------------


            XtVinvX_local.noalias() += group.X.transpose() * workspace.VinvX;
            XtVinvy_local.noalias() += group.X.transpose() * workspace.Vinvy;

            quad_local += group.y.dot(workspace.Vinvy);

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
  const VectorXd& y = model.y;
  int p = model.p;
  int q = model.q;
  int n_corr = model.n_corr;
  VectorXd log_sds = theta.head(q);
  VectorXd corrs_unconstrained = theta.segment(q, n_corr);
  double log_sigma_resid = 0.0;
  int k = 0;
  VectorXd gamma;
  std::unique_ptr<ResponseModel> response;

  if (!model.heteroscedastic) {
    log_sigma_resid = 0.5*theta(q + n_corr);
  } else {
    response = std::make_unique<HeteroNormalResponse>();
  
    k = model.groups[0].W.cols();
    gamma = theta.segment(q + n_corr, k);
    //std::cout << "Updating working model with " << k << " columns" << std::endl;
    response->update_working(model, gamma);
  }
  //double lo_sigma = model.lo_sigma;
  //double hi_sigma = model.hi_sigma;

  //int n_groups = groups.size();
  
  // Extract parameters from theta
  
  // Parameter validation
  VectorXd corrs = corrs_unconstrained.array().tanh().matrix();
  if ((corrs.array().abs() > 0.9999999).any()) return bad_result();
  if (!model.heteroscedastic && log_sigma_resid < std::log(std::numeric_limits<double>::epsilon())) return bad_result();
  
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
  double penalty = compute_penalty(model, log_sds, gamma, log_sigma_resid);

  auto t_after_build = std::chrono::high_resolution_clock::now();
  double t_build = std::chrono::duration<double>(t_after_build - t_start).count();
  
  // Accumulate sufficient statistics across groups
  MatrixXd XtVinvX = MatrixXd::Zero(p, p);
  VectorXd XtVinvy = VectorXd::Zero(p);
  double logdetV = 0.0;
  double quad = 0.0;

  auto t_loop_start = std::chrono::high_resolution_clock::now();
  if (!model.heteroscedastic) {
    double sigma = std::exp(log_sigma_resid);
    double sigma2 = sigma * sigma;
  
    bool success = accumulate_groups(model, G_inv, sigma2, XtVinvX, XtVinvy, logdetV, quad);
    if (!success) return bad_result();
  }
  else {
    bool success = accumulate_groups_hetero(model, *response, G_inv, XtVinvX, XtVinvy, logdetV, quad);
    if (!success) return bad_result();
  }
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

// [[Rcpp::export]]
VectorXd compute_beta_hat_cpp(
    const Eigen::VectorXd& theta,
    Rcpp::XPtr<PreparedModel> model
) {
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    const int p = model->p;
    const int q = model->q;
    const int n_groups = model->groups.size();

    // --- build G ---
    MatrixXd L = build_cholesky_cpp(theta, q);
    MatrixXd G = L * L.transpose();

    double sigma = std::exp(theta(theta.size() - 1));
    double sigma2 = sigma * sigma;

    // jitter + inverse
    MatrixXd G_jittered = G + 1e-8 * MatrixXd::Identity(q, q);
    MatrixXd G_inv = G_jittered.inverse();

    MatrixXd XtVinvX = MatrixXd::Zero(p, p);
    VectorXd XtVinvy = VectorXd::Zero(p);

    // precompute constants
    double inv_sigma2 = 1.0 / sigma2;
    double inv_sigma4 = inv_sigma2 * inv_sigma2;

    for (int g = 0; g < n_groups; ++g) {

        const auto& group = model->groups[g];

        const MatrixXd& Xi = group.X;
        const MatrixXd& Zi = group.Z;
        const VectorXd& yi = group.y;

        const MatrixXd& ZiZi = group.ZtZ;
        const MatrixXd& Zix  = group.ZtX;
        const VectorXd& Ziy  = group.ZtY;

        // K = G^{-1} + Z'Z / sigma2
        MatrixXd K = G_inv + ZiZi * inv_sigma2;

        Eigen::LLT<MatrixXd> llt(K);

        // ---- solve for BOTH RHS at once ----
        MatrixXd RHS(q, p + 1);
        RHS << Ziy, Zix;   // careful: adjust if dimensions differ

        MatrixXd K_inv_RHS =
            llt.solve(RHS);

        VectorXd K_inv_Ziy = K_inv_RHS.col(0);
        MatrixXd K_inv_Zix = K_inv_RHS.rightCols(p);

        // V^{-1} y
        VectorXd V_inv_y =
            (yi * inv_sigma2) - (Zi * K_inv_Ziy) * inv_sigma4;

        // V^{-1} X
        MatrixXd V_inv_X =
            (Xi * inv_sigma2) - (Zi * K_inv_Zix) * inv_sigma4;

        XtVinvX.noalias() += Xi.transpose() * V_inv_X;
        XtVinvy.noalias() += Xi.transpose() * V_inv_y;
    }

    // beta_hat
    VectorXd beta_hat = XtVinvX.ldlt().solve(XtVinvy);

    return beta_hat;
}
