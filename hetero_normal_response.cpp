// [[Rcpp::depends(RcppEigen)]]

#include "hetero_normal_response.hpp"
#include "covariance_utils.hpp"
#include "random_effects_utils.hpp"

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
        if (group_i.containsElementNamed("observed")) {
            groups[i].observed = Rcpp::as<bool>(group_i["observed"]);
        }
        if (group_i.containsElementNamed("random_effect_indices")) {
            const Rcpp::IntegerVector indices = group_i["random_effect_indices"];
            groups[i].random_effect_indices.assign(indices.begin(), indices.end());
        }
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

int HeteroNormalResponse::fixed_effect_dimension() const
{
    return groups.empty() ? 0 : groups.front().X.cols();
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

void HeteroNormalResponse::update_working(const VectorXd& parameters) const
{
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
        working_groups_[g].eta.noalias() = groups[g].W * parameters;
        working_groups_[g].inv_resid_var.array() =
            (-working_groups_[g].eta.array()).exp();
    }
}

void HeteroNormalResponse::update_working_from_theta(
    const VectorXd& theta,
    int q,
    int n_corr,
    int parameter_offset) const
{
    update_working(theta.segment(q + n_corr + parameter_offset, parameter_count()));
}

bool HeteroNormalResponse::parameters_are_valid(const VectorXd& parameters) const
{
    return parameters.size() == parameter_count();
}

bool HeteroNormalResponse::parameters_are_valid_for_theta(
    const VectorXd& theta,
    int q,
    int n_corr,
    int parameter_offset) const
{
    return parameters_are_valid(theta.segment(q + n_corr + parameter_offset, parameter_count()));
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
    ResponseGroupWorkspace& ws) const
{
    const auto& working_group = working_groups_[group_index];
    compute_hetero_normal_group(
        random_effects_group,
        response_group,
        working_group.inv_resid_var,
        working_group.eta,
        ws);
}

VectorXd HeteroNormalResponse::compute_beta_hat(
    const PreparedModel& prepared,
    const MatrixXd& G_inv,
    const VectorXd& theta) const
{
    update_working_from_theta(theta, prepared.q, prepared.n_corr, 0);

    MatrixXd XtVinvX = MatrixXd::Zero(groups.front().X.cols(), groups.front().X.cols());
    VectorXd XtVinvy = VectorXd::Zero(groups.front().X.cols());
    double logdetV = 0.0;
    double quad = 0.0;

    if (!accumulate_groups_response(prepared, *this, G_inv, XtVinvX, XtVinvy, logdetV, quad)) {
        Rcpp::stop("Could not accumulate response groups.");
    }

    Eigen::LLT<MatrixXd> llt_XtVinvX(XtVinvX);
    if (llt_XtVinvX.info() != Eigen::Success)
        Rcpp::stop("Could not factor fixed-effects information matrix.");
    return llt_XtVinvX.solve(XtVinvy);
}

void compute_hetero_normal_group(
    const GroupData& random_effects_group,
    const ResponseGroupData& response_group,
    const VectorXd& inv_resid_var,
    const VectorXd& eta,
    ResponseGroupWorkspace& ws)
{
    int n = random_effects_group.Z.rows();
    int q_r = response_group.random_effect_indices.size();
    
    ws.RinvZ.resize(n, q_r);
    for (int k = 0; k < q_r; ++k) {
        int col_idx = response_group.random_effect_indices[k] - 1;
        if (col_idx >= 0 && col_idx < random_effects_group.Z.cols()) {
            ws.RinvZ.col(k) = random_effects_group.Z.col(col_idx);
        }
    }
    
    const int N_obs = static_cast<int>(response_group.y.size());
    ws.RinvX = response_group.X; 
    for (int i = 0; i < N_obs; ++i) {
        ws.RinvZ.row(i) *= inv_resid_var(i);
        ws.RinvX.row(i) *= inv_resid_var(i); 
    }

    ws.Rinvy = response_group.y;
    ws.Rinvy.array() *= inv_resid_var.array();
    ws.logdetR = eta.sum(); // log|diag(Var)| = sum(log(var)) = sum(eta) since inv_var = exp(-eta)
}
