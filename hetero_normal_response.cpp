// [[Rcpp::depends(RcppEigen)]]

#include "hetero_normal_response.hpp"

namespace {
std::vector<ResponseGroupData> extract_hetero_groups(const Rcpp::List& prepared)
{
    Rcpp::List group_list = prepared["group_list"];
    std::vector<ResponseGroupData> groups(group_list.size());
    for (int i = 0; i < group_list.size(); ++i) {
        Rcpp::List group_i = group_list[i];
        groups[i].X = Rcpp::as<MatrixXd>(group_i["X"]);
        groups[i].y = Rcpp::as<VectorXd>(group_i["y"]);
        groups[i].ZtY = Rcpp::as<VectorXd>(group_i["ZtY"]);
        groups[i].ZtX = Rcpp::as<MatrixXd>(group_i["ZtX"]);
        groups[i].W = Rcpp::as<MatrixXd>(group_i["W"]);
    }
    return groups;
}
} // namespace

HeteroNormalResponse::HeteroNormalResponse(const Rcpp::List&) {}

void HeteroNormalResponse::initialise_from_prepared(const Rcpp::List& prepared)
{
    groups = extract_hetero_groups(prepared);
}

int HeteroNormalResponse::parameter_count() const
{
    return groups.empty() ? 0 : groups.front().W.cols();
}

void HeteroNormalResponse::initialise_working()
{
    working_groups_.resize(groups.size());
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
        const int n = static_cast<int>(groups[g].y.size());
        working_groups_[g].eta.resize(n);
        working_groups_[g].inv_resid_var.resize(n);
    }
}

void HeteroNormalResponse::update_working(const VectorXd& parameters)
{
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
        working_groups_[g].eta.noalias() = groups[g].W * parameters;
        working_groups_[g].inv_resid_var.array() =
            (-working_groups_[g].eta.array()).exp();
    }
}

bool HeteroNormalResponse::parameters_are_valid(const VectorXd& parameters) const
{
    return parameters.size() == parameter_count();
}

double HeteroNormalResponse::residual_variance_penalty() const
{
    double penalty = 0.0;
    for (const auto& group : working_groups_)
        penalty += group.eta.squaredNorm();
    return penalty;
}

void HeteroNormalResponse::compute_group(
    const GroupData& random_effects_group,
    const ResponseGroupData& response_group,
    int group_index,
    GroupWorkspace& ws) const
{
    const auto& working_group = working_groups_[group_index];
    compute_hetero_normal_group(
        random_effects_group,
        response_group,
        working_group.inv_resid_var,
        working_group.eta,
        ws);
}

void compute_hetero_normal_group(
    const GroupData& random_effects_group,
    const ResponseGroupData& response_group,
    const VectorXd& inv_resid_var,
    const VectorXd& eta,
    GroupWorkspace& ws)
{
    ws.RinvZ = random_effects_group.Z;
    ws.RinvX = response_group.X;
    ws.Rinvy = response_group.y;

    const int n = random_effects_group.Z.rows();
    for (int i = 0; i < n; ++i) {
        ws.RinvZ.row(i) *= inv_resid_var(i);
        ws.RinvX.row(i) *= inv_resid_var(i);
    }
    ws.Rinvy.array() *= inv_resid_var.array();
    ws.logdetR = eta.sum();
}
