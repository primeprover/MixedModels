
// [[Rcpp::depends(RcppEigen)]]

#include "response.hpp"
#include <limits>

namespace {
std::vector<ResponseGroupData> extract_response_groups(
    const Rcpp::List& prepared,
    const bool include_variance_design)
{
  Rcpp::List group_list = prepared["group_list"];
  std::vector<ResponseGroupData> groups(group_list.size());
  for (int i = 0; i < group_list.size(); ++i) {
    Rcpp::List group_i = group_list[i];
    groups[i].X = Rcpp::as<MatrixXd>(group_i["X"]);
    groups[i].y = Rcpp::as<VectorXd>(group_i["y"]);
    groups[i].ZtY = Rcpp::as<VectorXd>(group_i["ZtY"]);
    groups[i].ZtX = Rcpp::as<MatrixXd>(group_i["ZtX"]);
    if (include_variance_design)
      groups[i].W = Rcpp::as<MatrixXd>(group_i["W"]);
  }
  return groups;
}

void compute_normal_group(const GroupData& random_effects_group,
                          const ResponseGroupData& response_group,
                          const ResponseWorkingGroup& response_working,
                          GroupWorkspace& ws)
{
    const auto& inv_resid_var = response_working.inv_resid_var;
    ws.RinvZ = random_effects_group.Z.array().colwise() * inv_resid_var.array();
    ws.RinvX = response_group.X.array().colwise() * inv_resid_var.array();
    ws.Rinvy = response_group.y.array() * inv_resid_var.array();
    ws.logdetR = response_working.eta.sum();
}

double log_residual_variance_penalty(const ResponseWorking& working)
{
  double penalty = 0.0;
  for (const auto& group : working.groups)
    penalty += group.eta.squaredNorm();
  return penalty;
}
}

HomoNormalResponse::HomoNormalResponse(const Rcpp::List& prepared)
{
  groups = extract_response_groups(prepared, false);
}

void HomoNormalResponse::initialise_working()
{
  working.groups.resize(groups.size());
  for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
    const int n = groups[g].y.size();
    working.groups[g].eta.resize(n);
    working.groups[g].inv_resid_var.resize(n);
  }
}

void HomoNormalResponse::update_working(const VectorXd& parameters)
{
  const double log_resid_variance = parameters(0);
  for (auto& group : working.groups) {
    group.eta.setConstant(log_resid_variance);
    group.inv_resid_var.setConstant(std::exp(-log_resid_variance));
  }
}

bool HomoNormalResponse::parameters_are_valid(const VectorXd& parameters) const
{
  return parameters.size() == 1 &&
      0.5 * parameters(0) >= std::log(std::numeric_limits<double>::epsilon());
}

double HomoNormalResponse::residual_variance_penalty() const
{
  return log_residual_variance_penalty(working);
}

void HomoNormalResponse::compute_group(
    const GroupData& random_effects_group,
    const ResponseGroupData& response_group,
    const ResponseWorkingGroup& response_working,
    GroupWorkspace& ws) const
{
  compute_normal_group(random_effects_group, response_group, response_working, ws);
}

HeteroNormalResponse::HeteroNormalResponse(const Rcpp::List& prepared)
{
  groups = extract_response_groups(prepared, true);
}

int HeteroNormalResponse::parameter_count() const
{
  return groups.empty() ? 0 : groups.front().W.cols();
}

void HeteroNormalResponse::initialise_working()
{
  working.groups.resize(groups.size());
  for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
    const int n = groups[g].y.size();
    working.groups[g].eta.resize(n);
    working.groups[g].inv_resid_var.resize(n);
  }
}

void HeteroNormalResponse::update_working(
    const Eigen::VectorXd& gamma
)
{
  for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
    working.groups[g].eta.noalias() = groups[g].W * gamma;
    working.groups[g].inv_resid_var.array() =
        (-working.groups[g].eta.array()).exp();
  }
}

bool HeteroNormalResponse::parameters_are_valid(const VectorXd& parameters) const
{
  return parameters.size() == parameter_count();
}

double HeteroNormalResponse::residual_variance_penalty() const
{
  return log_residual_variance_penalty(working);
}


void HeteroNormalResponse::compute_group(
    const GroupData& random_effects_group,
    const ResponseGroupData& response_group,
    const ResponseWorkingGroup& response_working,
    GroupWorkspace& ws) const
{
    compute_normal_group(random_effects_group, response_group, response_working, ws);
}
