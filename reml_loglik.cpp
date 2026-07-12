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

namespace {

VectorXd solve_fixed_effects(
    const MatrixXd& XtVinvX,
    const VectorXd& XtVinvy)
{
  Eigen::LLT<MatrixXd> llt_XtVinvX(XtVinvX);
  if (llt_XtVinvX.info() != Eigen::Success)
    Rcpp::stop("Could not factor fixed-effects information matrix.");
  return llt_XtVinvX.solve(XtVinvy);
}

double logdet_from_cholesky(const MatrixXd& XtVinvX)
{
  Eigen::LLT<MatrixXd> llt_XtVinvX(XtVinvX);
  if (llt_XtVinvX.info() != Eigen::Success)
    Rcpp::stop("Could not factor fixed-effects information matrix.");
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
  model->responses.push_back(make_response(prepared));
  model->responses.back()->initialise_from_prepared(prepared);
  model->responses.back()->initialise_working();
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
    penalty += log_sds(i) * 1000000.0; // weak penalty to provide slope for flat regions (assuming penalty multiplied by 1e-6)
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

            response.compute_group(group, response_group, ig, response_workspace);

            if (!random_effects_compute(group, G_inv, response_workspace, integration_workspace)) {
                bad.store(true, std::memory_order_relaxed);
                continue;
            }

            local_contribution.add_group_contribution(response_group, response_workspace, integration_workspace);
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
    const MatrixXd& G_inv,
    ConditionalContribution& contribution)
{
    if (model.responses.empty())
        return false;

    const int p = model.responses.front()->fixed_effect_dimension();
    if (p <= 0)
        return false;

    for (const auto& response_ptr : model.responses) {
        if (response_ptr->fixed_effect_dimension() != p)
            return false;
    }

    contribution = ConditionalContribution(p);

    const auto& groups = model.groups;
    std::atomic<bool> bad(false);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        ConditionalContribution local_contribution(p);
        std::vector<ResponseGroupWorkspace> response_workspaces(model.responses.size());
        std::vector<RandomEffectsIntegrationWorkspace> integration_workspaces(model.responses.size());

#ifdef _OPENMP
#pragma omp for schedule(guided)
#endif
        for (int ig = 0; ig < (int)groups.size(); ++ig) {
            const auto& group = groups[ig];
            RandomEffectsIntegrationWorkspace shared_integration;
            shared_integration.ZtRinvZ.setZero(group.Z.cols(), group.Z.cols());
            shared_integration.ZtRinvX.setZero(group.Z.cols(), p);
            shared_integration.ZtRinvy.setZero(group.Z.cols());

            for (int ir = 0; ir < (int)model.responses.size(); ++ir) {
                const ResponseModel& response = *model.responses[ir];
                response.compute_group(group, response.groups[ig], ig, response_workspaces[ir]);
                random_effects_begin(group, response_workspaces[ir], integration_workspaces[ir]);
                shared_integration.ZtRinvZ.noalias() += integration_workspaces[ir].ZtRinvZ;
                shared_integration.ZtRinvX.noalias() += integration_workspaces[ir].ZtRinvX;
                shared_integration.ZtRinvy.noalias() += integration_workspaces[ir].ZtRinvy;
            }

            shared_integration.K = G_inv;
            shared_integration.K.noalias() += shared_integration.ZtRinvZ;
            shared_integration.llt.compute(shared_integration.K);
            if (shared_integration.llt.info() != Eigen::Success) {
                bad.store(true, std::memory_order_relaxed);
                continue;
            }

            shared_integration.logdetK = 2.0 * shared_integration.llt.matrixLLT().diagonal().array().log().sum();

            for (int ir = 0; ir < (int)model.responses.size(); ++ir) {
                const ResponseModel& response = *model.responses[ir];
                const ResponseGroupData& response_group = response.groups[ig];
                RandomEffectsIntegrationWorkspace& integration_workspace = integration_workspaces[ir];
                random_effects_finalize(shared_integration.llt, response_workspaces[ir], integration_workspace);
                integration_workspace.logdetK = shared_integration.logdetK;
                local_contribution.add_group_contribution(response_group, response_workspaces[ir], integration_workspace);
            }
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        {
            contribution.merge_from(local_contribution);
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
    const Eigen::VectorXd& theta_raw,
    bool return_penalty,
    bool return_timing
) {
  PreparedModel& model = *model_xptr;
  auto t_start = std::chrono::high_resolution_clock::now();

  const int q = model.q;
  const int n_corr = model.n_corr;
  const int response_offset = q + n_corr;

  if (model.responses.empty()) {
    Rcpp::stop("The current REML accumulator requires at least one response.");
  }
  int p = 0;
  for (const auto& response_ptr : model.responses) {
    const ResponseModel& response = *response_ptr;
    if (theta_raw.size() != response_offset + response.parameter_count()) {
      Rcpp::stop("theta has the wrong number of response parameters.");
    }
    if (p == 0) {
      p = response.fixed_effect_dimension();
    } else if (response.fixed_effect_dimension() != p) {
      Rcpp::stop("The current multi-response accumulator requires responses with the same fixed-effect dimension.");
    }
  }

  VectorXd log_sds = theta_raw.head(q);
  VectorXd corrs_unconstrained = theta_raw.segment(q, n_corr);
  for (const auto& response_ptr : model.responses) {
    ResponseModel& response = *response_ptr;
    response.update_working_from_theta(theta_raw, q, n_corr);
    if (!response.parameters_are_valid_for_theta(theta_raw, q, n_corr))
      return bad_result();
  }

  VectorXd corrs = corrs_unconstrained.array().tanh().matrix();
  if ((corrs.array().abs() > 0.9999999).any()) return bad_result();

  RandomEffectsSystem random_effects = build_random_effects_system(
      q, log_sds, corrs_unconstrained);
  if (random_effects.G_inv.size() == 0) return bad_result();

  double penalty = 0.0;
  for (const auto& response_ptr : model.responses) {
    penalty += compute_penalty(model, log_sds, *response_ptr);
  }

  auto t_after_build = std::chrono::high_resolution_clock::now();
  double t_build = std::chrono::duration<double>(t_after_build - t_start).count();

  ConditionalContribution contribution(p);
  bool success = accumulate_conditional_contributions(
      model, random_effects.G_inv, contribution);
  if (!success) return bad_result();

  auto t_after_groups = std::chrono::high_resolution_clock::now();
  double t_groups = std::chrono::duration<double>(t_after_groups - t_after_build).count();

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
    for (const auto& response_ptr : prepared.responses) {
      const ResponseModel& response = *response_ptr;
      if (theta.size() != response_offset + response.parameter_count())
        Rcpp::stop("theta has the wrong number of response parameters.");
      if (p == 0) {
        p = response.fixed_effect_dimension();
      } else if (response.fixed_effect_dimension() != p) {
        Rcpp::stop("The current multi-response accumulator requires responses with the same fixed-effect dimension.");
      }
    }
    for (const auto& response_ptr : prepared.responses) {
      ResponseModel& response = *response_ptr;
      response.update_working_from_theta(theta, prepared.q, prepared.n_corr);
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
    if (llt_XtVinvX.info() != Eigen::Success)
      Rcpp::stop("Could not factor fixed-effects information matrix.");
    return llt_XtVinvX.solve(contribution.XtVinvy);
}
