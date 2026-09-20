// reml_loglik.cpp
// C++ implementation of REML log-likelihood for mixed models using RcppEigen
// This file provides a compiled engine for fit_mixed_simple(..., engine = "C++")


#include "reml_loglik.hpp"
#include "response.hpp"
#include "homo_normal_response.hpp"
#include "hetero_normal_response.hpp"
#include "response_factory.hpp"
#include "covariance_utils.hpp"
#include "random_effects_utils.hpp"
#include <cmath>
#include <vector>
#include <atomic>
#include <chrono>
#include <memory>
#include <limits>


// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

// Extract group data from R list
// This converts prepared$group_list (R list) into C++ vector of GroupData
// INPUT: Rcpp::List prepared
// OUTPUT: std::vector<GroupData>
namespace {

std::vector<GroupData> extract_groups_from_payload(const Rcpp::List& prepared) {
  const Rcpp::List group_list = prepared.containsElementNamed("shared_group_list")
      ? prepared["shared_group_list"]
      : prepared.containsElementNamed("group_list")
          ? prepared["group_list"]
          : Rcpp::List();
  const int n_groups = group_list.size();
  std::vector<GroupData> groups(n_groups);

  for (int i = 0; i < n_groups; ++i) {
    const Rcpp::List group_i = group_list[i];
    groups[i].Z = Rcpp::as<MatrixXd>(group_i["Z"]);
    groups[i].n = Rcpp::as<int>(group_i["n"]);
    groups[i].ZtZ = Rcpp::as<MatrixXd>(group_i["ZtZ"]);
  }

  return groups;
}

} // namespace

std::vector<GroupData> extract_groups(const Rcpp::List& prepared) {
  return extract_groups_from_payload(prepared);
}

PreparedModel::~PreparedModel() = default;

namespace {

VectorXd solve_fixed_effects(
    const MatrixXd& XtVinvX,
    const VectorXd& XtVinvy)
{
  Eigen::LLT<MatrixXd> llt_XtVinvX(XtVinvX);
  if (llt_XtVinvX.info() != Eigen::Success) {
    std::cout << "XtVinvX: " << std::endl << XtVinvX << std::endl;
    Rcpp::stop("Could not factor fixed-effects information matrix.");
  }
  return llt_XtVinvX.solve(XtVinvy);
}

double logdet_from_cholesky(const MatrixXd& XtVinvX)
{
  Eigen::LLT<MatrixXd> llt_XtVinvX(XtVinvX);
  if (llt_XtVinvX.info() != Eigen::Success) {
    std::cout << "XtVinvX: " << std::endl << XtVinvX << std::endl;
    Rcpp::stop("Could not factor fixed-effects information matrix.");
  }
  return 2.0 * llt_XtVinvX.matrixLLT().diagonal().array().log().sum();
}

} // namespace

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
  if (prepared.containsElementNamed("response_spec")) {
    const Rcpp::List response_spec = prepared["response_spec"];
    if (response_spec.containsElementNamed("random_map")) {
      const Rcpp::IntegerVector random_map = response_spec["random_map"];
      model->global_random_effect_sizes.assign(random_map.begin(), random_map.end());
      model->global_random_effect_offsets.assign(random_map.size(), 0);
    }
  }

  if (prepared.containsElementNamed("responses")) {
    const Rcpp::List responses = prepared["responses"];
    int parameter_offset = 0;
    int fixed_effect_offset = 0;
    for (int i = 0; i < responses.size(); ++i) {
      const Rcpp::List response_prepared = responses[i];
      auto response = make_response(response_prepared);
      response->initialise_from_prepared(response_prepared);
      response->set_parameter_offset(parameter_offset);
      response->set_fixed_effect_offset(fixed_effect_offset);
      response->initialise_working();
      parameter_offset += response->parameter_count();
      fixed_effect_offset += response->fixed_effect_dimension();
      model->responses.push_back(std::move(response));
    }
  } else {
    auto response = make_response(prepared);
    response->initialise_from_prepared(prepared);
    response->initialise_working();
    model->responses.push_back(std::move(response));
  }
  return Rcpp::XPtr<PreparedModel>(model, true);
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
    penalty += log_sds(i)*log_sds(i) * 10000.0; // weak penalty to provide slope for flat regions (assuming penalty multiplied by 1e-6)
  }
  penalty *= 1e-6;
  penalty += model.residual_variance_penalty_weight *
      response.residual_variance_penalty();
  //std::cout << "Penalty: " << penalty << " Lo_sigma: " << lo_sigma << " Hi_sigma: " << hi_sigma << std::endl;
  return penalty;
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
    const int p = response.fixed_effect_dimension();
    ConditionalContribution contribution(p);

    std::atomic<bool> bad(false);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        ConditionalContribution local_contribution(p);

        ResponseGroupWorkspace response_workspace;
        RandomEffectsIntegrationWorkspace integration_workspace;

#ifdef _OPENMP
#pragma omp for schedule(guided)
#endif
        for (int ig = 0; ig < (int)groups.size(); ++ig) {
            const auto& group = groups[ig];
            const auto& response_group = response.groups[ig];
            if (!response_group.observed) {
                continue;
            }

            response.compute_group(group, response_group, ig, response_workspace);

            if (!random_effects_compute(group, G_inv, response_workspace, integration_workspace)) {
                bad.store(true, std::memory_order_relaxed);
                continue;
            }

            local_contribution.add_group_contribution(response_group, response_workspace, integration_workspace, response.fixed_effect_offset(), response.fixed_effect_dimension());
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        {
            contribution.merge_from(local_contribution);
        }
    }

    XtVinvX = contribution.XtVinvX;
    XtVinvy = contribution.XtVinvy;
    logdetV = contribution.logdetV;
    quad = contribution.quad;

    return !bad.load();
}

bool accumulate_conditional_contributions(
    const PreparedModel& model,
    const MatrixXd& G_inv_global,
    ConditionalContribution& contribution)
{
  if (model.responses.empty()) return false;

  // 1. Global fixed-effect dimension
  int P_total = 0;
  for (const auto& resp : model.responses) P_total += resp->fixed_effect_dimension();
  if (P_total <= 0) return false;

  contribution = ConditionalContribution(P_total);
  const auto& groups = model.groups;
  std::atomic<bool> bad(false);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    ConditionalContribution local_contribution(P_total);
    std::vector<ResponseGroupWorkspace> resp_ws(model.responses.size());
    std::vector<RandomEffectsIntegrationWorkspace> integ_ws(model.responses.size());

#ifdef _OPENMP
#pragma omp for schedule(guided)
#endif
    for (int ig = 0; ig < (int)groups.size(); ++ig) {
      const auto& group = groups[ig];

      // 2. Global accumulator: Z' R⁻¹ Z blocks across responses
      MatrixXd sum_ZtRinvZ_global(model.q, model.q);
      sum_ZtRinvZ_global.setZero();

      for (int ir = 0; ir < (int)model.responses.size(); ++ir) {
        const ResponseModel& resp = *model.responses[ir];
        const ResponseGroupData& resp_grp = resp.groups[ig];
        if (!resp_grp.observed) continue;

        // Update response-specific R⁻¹ working buffers
        resp.compute_group(group, resp_grp, ig, resp_ws[ir]);

        // Slice global Z → response-specific Z_r (version-agnostic column extraction)
        int Q_resp = resp_grp.random_effect_indices.size();
        MatrixXd Z_r(group.Z.rows(), Q_resp);
        for (int k = 0; k < Q_resp; ++k) {
          int idx = resp_grp.random_effect_indices[k] - 1; // R is 1-based
          if (idx >= 0 && idx < group.Z.cols()) {
            Z_r.col(k) = group.Z.col(idx);
          }
        }

        // Scatter Q_resp × Q_resp block into global Q_global × Q_global space
        MatrixXd block = Z_r.transpose() * resp_ws[ir].RinvZ;
        for (int i = 0; i < Q_resp; ++i) {
          int gi = resp_grp.random_effect_indices[i] - 1;
          for (int j = 0; j <= i; ++j) { // Exploit symmetry
            int gj = resp_grp.random_effect_indices[j] - 1;
            if (gi >= 0 && gj >= 0 && gi < model.q && gj < model.q) {
              sum_ZtRinvZ_global(gi, gj) += block(i, j);
              if (i != j) sum_ZtRinvZ_global(gj, gi) += block(i, j);
            }
          }
        }
      }
      //std::cout << "Got here5.1" << std::endl;
  
      // 3. Form & factor shared K = G⁻¹ + Z' R⁻¹ Z
      MatrixXd K_global = G_inv_global + sum_ZtRinvZ_global;
      Eigen::LLT<MatrixXd> llt(K_global);
      if (llt.info() != Eigen::Success) { bad.store(true); continue; }
      double logdetK = 2.0 * llt.matrixLLT().diagonal().array().log().sum();

      // 4. Woodbury V⁻¹ solves per response using shared factorization
      for (int ir = 0; ir < (int)model.responses.size(); ++ir) {
        const ResponseModel& resp = *model.responses[ir];
        const ResponseGroupData& resp_grp = resp.groups[ig];
        if (!resp_grp.observed) continue;

        RandomEffectsIntegrationWorkspace& ws = integ_ws[ir];
        int Q_resp = resp_grp.random_effect_indices.size();
        
        // Re-slice Z_r for RHS solves (same indices as above)
        MatrixXd Z_r(group.Z.rows(), Q_resp);
        for (int k = 0; k < Q_resp; ++k) {
          int idx = resp_grp.random_effect_indices[k] - 1;
          if (idx >= 0 && idx < group.Z.cols()) Z_r.col(k) = group.Z.col(idx);
        }

        ws.ZtRinvX.noalias() = Z_r.transpose() * resp_ws[ir].RinvX;
        ws.ZtRinvy.noalias() = Z_r.transpose() * resp_ws[ir].Rinvy;

        // Solve against global K. Note: llt is model.q x model.q, but ZtRinvX/ZtRinvy are Q_resp-sized.
        // We must embed them into a model.q-sized vector/matrix before solving to match dimensions.
        MatrixXd tmp_X_rhs(model.q, ws.ZtRinvX.cols());
        tmp_X_rhs.setZero();
        VectorXd tmp_y_rhs(model.q);
        tmp_y_rhs.setZero();

        for (int k = 0; k < Q_resp; ++k) {
            int gi = resp_grp.random_effect_indices[k] - 1;
            if (gi >= 0 && gi < model.q) {
                tmp_X_rhs.row(gi) = ws.ZtRinvX.row(k);
                tmp_y_rhs(gi) = ws.ZtRinvy(k);
            }
        }

        MatrixXd solved_X_full = llt.solve(tmp_X_rhs);
        VectorXd solved_y_full = llt.solve(tmp_y_rhs);

        ws.KinvZtRinvX.resize(Q_resp, ws.ZtRinvX.cols());
        ws.KinvZtRinvy.resize(Q_resp);
        for (int k = 0; k < Q_resp; ++k) {
            int gi = resp_grp.random_effect_indices[k] - 1;
            if (gi >= 0 && gi < model.q) {
                ws.KinvZtRinvX.row(k) = solved_X_full.row(gi);
                ws.KinvZtRinvy(k) = solved_y_full(gi);
            }
        }

        ws.VinvX.noalias() = resp_ws[ir].RinvX - resp_ws[ir].RinvZ * ws.KinvZtRinvX;
        ws.Vinvy.noalias() = resp_ws[ir].Rinvy - resp_ws[ir].RinvZ * ws.KinvZtRinvy;
        ws.logdetK = logdetK;

        // Scatter into global conditional contribution using correct fixed-effect offsets
        local_contribution.add_group_contribution(resp_grp, resp_ws[ir], ws,
            resp.fixed_effect_offset(), resp.fixed_effect_dimension());
      }
      //std::cout << "Got here5.2 " << ig << std::endl;
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    { contribution.merge_from(local_contribution); }
  }
  return !bad.load();
}

// ============================================================
// Main REML log-likelihood computation
// ============================================================

// [[Rcpp::export]]
Rcpp::List reml_loglik_cpp_impl(
    Rcpp::XPtr<PreparedModel> model_xptr,
    const Eigen::VectorXd& theta_raw,
    bool return_penalty,
    bool return_timing
) {
  PreparedModel& model = *model_xptr;
  auto t_start = std::chrono::high_resolution_clock::now();
  //std::cout << "Got here" << std::endl;
  //std::cout << "Theta raw: " << theta_raw.transpose() << std::endl;
  const int q_global = model.q;
  const int n_corr = model.n_corr;
  const int response_offset = q_global + n_corr;

  if (model.responses.empty()) {
    Rcpp::stop("The current REML accumulator requires at least one response.");
  }
  int p = 0;
  int total_parameter_count = 0;
  for (const auto& response_ptr : model.responses) {
    const ResponseModel& response = *response_ptr;
    total_parameter_count += response.parameter_count();
    p += response.fixed_effect_dimension();
  }
  if (theta_raw.size() != response_offset + total_parameter_count) {
    Rcpp::stop("theta has the wrong number of response parameters.");
  }

  VectorXd log_sds = theta_raw.head(q_global);
  VectorXd corrs_unconstrained = theta_raw.segment(q_global, n_corr);
  for (const auto& response_ptr : model.responses) {
    ResponseModel& response = *response_ptr;
    response.update_working_from_theta(theta_raw, q_global, n_corr, response.parameter_offset());
    if (!response.parameters_are_valid_for_theta(theta_raw, q_global, n_corr, response.parameter_offset()))
      return bad_result();
  }
  //std::cout << "Got here3" << std::endl;
  
  VectorXd corrs = corrs_unconstrained.array().tanh().matrix();
  if ((corrs.array().abs() > 0.9999999).any()) return bad_result();

  RandomEffectsSystem random_effects = build_random_effects_system(
      q_global, log_sds, corrs_unconstrained);
  if (random_effects.G_inv.size() == 0) return bad_result();

  double penalty = 0.0;
  for (const auto& response_ptr : model.responses) {
    penalty += compute_penalty(model, log_sds, *response_ptr);
  }
  penalty = 0.0;
  auto t_after_build = std::chrono::high_resolution_clock::now();
  double t_build = std::chrono::duration<double>(t_after_build - t_start).count();
  //std::cout << "Got here4" << std::endl;
  
  ConditionalContribution contribution(p);
  bool success = accumulate_conditional_contributions(
      model, random_effects.G_inv, contribution);
  if (!success) return bad_result();
  //std::cout << "Got here5" << std::endl;
  
  auto t_after_groups = std::chrono::high_resolution_clock::now();
  double t_groups = std::chrono::duration<double>(t_after_groups - t_after_build).count();
  //std::cout << "Got here6" << std::endl;
  double logdetV = contribution.logdetV +
      static_cast<double>(model.groups.size()) * random_effects.logdetG;

  VectorXd beta_hat = solve_fixed_effects(contribution.XtVinvX, contribution.XtVinvy);
  double quad_res = std::max(0.0, contribution.quad - beta_hat.transpose() * contribution.XtVinvy);

  const int n = total_observation_count(model.responses.front()->groups);
  double ll = -0.5 * (
      logdetV +
      logdet_from_cholesky(contribution.XtVinvX) +
      quad_res +
      (n - p) * std::log(2.0 * M_PI));
  //std::cout << "Got here7" << std::endl;
  auto t_after_finalize = std::chrono::high_resolution_clock::now();
  double t_finalize = std::chrono::duration<double>(t_after_finalize - t_after_groups).count();

  Rcpp::List result;
  result["logLik_unpenalized"] = ll;
  result["logLik_penalized"] = ll - penalty;
  result["penalty"] = penalty;
  if (return_timing && return_penalty) {
    result["timings"] = Rcpp::List::create(
      Rcpp::Named("build_G") = t_build,
      Rcpp::Named("groups") = t_groups,
      Rcpp::Named("finalize") = t_finalize,
      Rcpp::Named("total") = t_build + t_groups + t_finalize
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
    if (prepared.responses.empty())
      Rcpp::stop("The current beta calculation requires at least one response.");

    const int response_offset = prepared.q + prepared.n_corr;
    int p = 0;
    int total_parameter_count = 0;
    for (const auto& response_ptr : prepared.responses) {
      const ResponseModel& response = *response_ptr;
      total_parameter_count += response.parameter_count();
      p += response.fixed_effect_dimension();
    }
    if (theta.size() != response_offset + total_parameter_count)
      Rcpp::stop("theta has the wrong number of response parameters.");
    for (const auto& response_ptr : prepared.responses) {
      ResponseModel& response = *response_ptr;
      response.update_working_from_theta(theta, prepared.q, prepared.n_corr, response.parameter_offset());
    }

    MatrixXd L = build_cholesky(theta, prepared.q);
    MatrixXd G = L * L.transpose();
    G.diagonal().array() += 1e-8;
    Eigen::LLT<MatrixXd> llt_G(G);
    if (llt_G.info() != Eigen::Success)
      Rcpp::stop("Could not factor random-effects covariance matrix.");
    MatrixXd G_inv = llt_G.solve(MatrixXd::Identity(prepared.q, prepared.q));

    ConditionalContribution contribution(p);
    if (!accumulate_conditional_contributions(prepared, G_inv, contribution))
      Rcpp::stop("Could not accumulate response groups.");

    Eigen::LLT<MatrixXd> llt_XtVinvX(contribution.XtVinvX);
    if (llt_XtVinvX.info() != Eigen::Success) {
      std::cout << "XtVinvX: " << std::endl << contribution.XtVinvX << std::endl;
      Rcpp::stop("Could not factor fixed-effects information matrix.");
    }
    return llt_XtVinvX.solve(contribution.XtVinvy);
}
