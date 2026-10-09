// reml_loglik.cpp - R-facing entry points for the mixed-model REML fit.
#include "reml_loglik.hpp"
#include "response.hpp"
#include "homo_normal_response.hpp"
#include "hetero_normal_response.hpp"
#include "response_factory.hpp"
#include "covariance_utils.hpp"
#include "random_effects_utils.hpp"
#include "contribution_accumulator.hpp"
#include "matrix_solve_utils.hpp"
#include <cmath>
#include <vector>
#include <chrono>
#include <memory>
#include <limits>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

namespace {

std::vector<GroupData> extract_groups(const Rcpp::List& prepared)
{
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
  }

  return groups;
}

} // namespace

PreparedModel::~PreparedModel() = default;

// [[Rcpp::export]]
Rcpp::XPtr<PreparedModel> init_model_cpp(const Rcpp::List& prepared)
{
  auto model = new PreparedModel();
  model->q = prepared["q"];
  model->n_corr = prepared["n_corr"];
  model->lo_sigma = prepared["lo_sigma"];
  model->hi_sigma = prepared["hi_sigma"];
  model->residual_variance_penalty_weight =
      prepared.containsElementNamed("residual_variance_penalty_weight")
          ? Rcpp::as<double>(prepared["residual_variance_penalty_weight"])
          : 0.0;
  if (model->residual_variance_penalty_weight < 0.0)
    Rcpp::stop("penalty must be non-negative.");
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
    const ResponseModel& response)
{
  double penalty = 0.0;
  for (int i = 0; i < model.q; ++i) {
    const double excess_hi = std::max(0.0, log_sds(i) - model.hi_sigma);
    const double excess_lo = std::max(0.0, model.lo_sigma - log_sds(i));
    penalty += excess_hi * excess_hi + excess_lo * excess_lo
        + log_sds(i) * log_sds(i) * 10000.0;
  }
  return penalty * 1e-6 + model.residual_variance_penalty_weight
      * response.residual_variance_penalty();
}

// ============================================================
// MAIN
// ============================================================

// [[Rcpp::export]]
Rcpp::List reml_loglik_cpp_impl(
    Rcpp::XPtr<PreparedModel> model_xptr,
    const Eigen::VectorXd& theta_raw,
    bool return_penalty,
    bool return_timing)
{
  PreparedModel& model = *model_xptr;
  auto t0 = std::chrono::high_resolution_clock::now();
  const int qg = model.q;
  const int nc = model.n_corr;
  const int ro = qg + nc;

  if (model.responses.empty()) {
    Rcpp::stop("Need at least one response.");
  }

  int p = 0;
  int total_p = 0;
  for (const auto& rp : model.responses) {
    const ResponseModel& r = *rp;
    total_p += r.parameter_count();
    p += r.fixed_effect_dimension();
  }
  if (theta_raw.size() != ro + total_p) {
    Rcpp::stop("theta size mismatch.");
  }

  VectorXd log_sds = theta_raw.head(qg);
  VectorXd cuc = theta_raw.segment(qg, nc);
  for (const auto& rp : model.responses) {
    ResponseModel& r = *rp;
    r.update_working_from_theta(theta_raw, qg, nc, rp->parameter_offset());
    if (!r.parameters_are_valid_for_theta(theta_raw, qg, nc, rp->parameter_offset())) {
      return bad_result();
    }
  }
  VectorXd corrs = cuc.array().tanh().matrix();
  if ((corrs.array().abs() > 0.9999999).any()) {
    return bad_result();
  }

  RandomEffectsSystem rs = build_random_effects_system(qg, log_sds, cuc);
  if (rs.G_inv.size() == 0) {
    return bad_result();
  }

  double penalty = 0.0;
  for (const auto& rp : model.responses) {
    penalty += compute_penalty(model, log_sds, *rp);
  }
  penalty = 0.0;
  auto t1 = std::chrono::high_resolution_clock::now();
  double t_build = std::chrono::duration<double>(t1 - t0).count();

  ConditionalContribution contribution(p);
  if (!accumulate_conditional_contributions(model, rs.G_inv, contribution)) {
    return bad_result();
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  double t_groups = std::chrono::duration<double>(t2 - t1).count();
  double logdetV = contribution.logdetV +
      static_cast<double>(model.groups.size()) * rs.logdetG;

  bool fe_ok;
  VectorXd beta = solve_fe(contribution.XtVinvX, contribution.XtVinvy, fe_ok);
  if (!fe_ok) {
    return bad_result();
  }
  double qr = std::max(0.0, contribution.quad - beta.transpose() * contribution.XtVinvy);

  // BUG 1 FIX: Sum observations across ALL responses
  int total_n = 0;
  for (const auto& rp : model.responses) {
    for (const auto& gr : rp->groups) {
      total_n += static_cast<int>(gr.y.size());
    }
  }

  bool det_ok;
  double logdetX = logdet_chol(contribution.XtVinvX, det_ok);
  if (!det_ok) {
    return bad_result();
  }
  double ll = -0.5 * (logdetV + logdetX + qr + (total_n - p) * std::log(2.0 * M_PI));
  auto t3 = std::chrono::high_resolution_clock::now();
  double t_finalize = std::chrono::duration<double>(t3 - t2).count();

  Rcpp::List result;
  result["logLik_unpenalized"] = ll;
  result["logLik_penalized"] = ll - penalty;
  result["penalty"] = penalty;
  if (return_timing && return_penalty) {
    result["timings"] = Rcpp::List::create(
        Rcpp::Named("build_G") = t_build,
        Rcpp::Named("groups") = t_groups,
        Rcpp::Named("finalize") = t_finalize,
        Rcpp::Named("total") = t_build + t_groups + t_finalize);
  }
  return result;
}

// [[Rcpp::export]]
VectorXd compute_beta_hat_cpp(const Eigen::VectorXd& theta, Rcpp::XPtr<PreparedModel> model)
{
  PreparedModel& p = *model;
  if (p.responses.empty()) {
    Rcpp::stop("Need at least one response.");
  }
  const int ro = p.q + p.n_corr;
  int p_ = 0;
  int tp = 0;
  for (const auto& rp : p.responses) {
    const ResponseModel& r = *rp;
    tp += r.parameter_count();
    p_ += r.fixed_effect_dimension();
  }
  if (theta.size() != ro + tp) {
    Rcpp::stop("theta size mismatch.");
  }
  for (const auto& rp : p.responses) {
    ResponseModel& r = *rp;
    r.update_working_from_theta(theta, p.q, p.n_corr, rp->parameter_offset());
  }
  MatrixXd L = build_cholesky(theta, p.q);
  MatrixXd G = L * L.transpose();
  G.diagonal().array() += 1e-8;
  Eigen::LLT<MatrixXd> lg(G);
  if (lg.info() != Eigen::Success) {
    Rcpp::stop("Could not factor G.");
  }
  MatrixXd Gi = lg.solve(MatrixXd::Identity(p.q, p.q));
  ConditionalContribution contribution(p_);
  if (!accumulate_conditional_contributions(p, Gi, contribution)) {
    Rcpp::stop("Could not accumulate.");
  }
  bool ok;
  VectorXd bh = solve_fe(contribution.XtVinvX, contribution.XtVinvy, ok);
  (void)ok;
  return bh;
}
