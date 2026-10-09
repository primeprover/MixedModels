// [[Rcpp::depends(RcppEigen)]]

#include "homo_normal_response.hpp"
#include "covariance_utils.hpp"
#include "random_effects_utils.hpp"
#include <cmath>
#include <limits>

namespace {
std::vector<ResponseGroupData> extract_homo_groups(const Rcpp::List& prepared)
{
    Rcpp::List group_list = prepared["group_list"];
    std::vector<ResponseGroupData> groups(group_list.size());
    for (int i = 0; i < group_list.size(); ++i) {
        Rcpp::List group_i = group_list[i];
            groups[i].X = Rcpp::as<MatrixXd>(group_i["X"]);
        groups[i].y = Rcpp::as<VectorXd>(group_i["y"]);
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

HomoNormalResponse::HomoNormalResponse(const Rcpp::List&) {}

void HomoNormalResponse::initialise_from_prepared(const Rcpp::List& prepared)
{
    groups = extract_homo_groups(prepared);
}

int HomoNormalResponse::parameter_count() const
{
    return 1;
}

int HomoNormalResponse::fixed_effect_dimension() const
{
    return groups.empty() ? 0 : groups.front().X.cols();
}

void HomoNormalResponse::initialise_working()
{
    // No per-observation working buffers needed for the homo-normal scalar path.
}

void HomoNormalResponse::update_working(const VectorXd& parameters) const
{
    log_resid_variance_ = parameters(0);
    inv_resid_var_ = std::exp(-log_resid_variance_);
}

void HomoNormalResponse::update_working_from_theta(
    const VectorXd& theta,
    int q,
    int n_corr,
    int parameter_offset) const
{
    update_working(theta.segment(q + n_corr + parameter_offset, parameter_count()));
}

bool HomoNormalResponse::parameters_are_valid(const VectorXd& parameters) const
{
    return parameters.size() == 1 &&
        0.5 * parameters(0) >= std::log(std::numeric_limits<double>::epsilon());
}

bool HomoNormalResponse::parameters_are_valid_for_theta(
    const VectorXd& theta,
    int q,
    int n_corr,
    int parameter_offset) const
{
    return parameters_are_valid(theta.segment(q + n_corr + parameter_offset, parameter_count()));
}

double HomoNormalResponse::residual_variance_penalty() const
{
    const int n = total_observation_count(groups);
    return static_cast<double>(n) * (log_resid_variance_ * log_resid_variance_);
}

void HomoNormalResponse::compute_group(
    const GroupData& random_effects_group,
    const ResponseGroupData& response_group,
    int group_index,
    ResponseGroupWorkspace& ws) const
{
    compute_homo_normal_group(
        random_effects_group,
        response_group,
        inv_resid_var_,
        log_resid_variance_,
        ws);
}

void compute_homo_normal_group(
    const GroupData& random_effects_group,
    const ResponseGroupData& response_group,
    double inv_resid_var,
    double log_resid_variance,
    ResponseGroupWorkspace& ws)
{
    int n = random_effects_group.Z.rows();
    int q_r = response_group.random_effect_indices.size();
    
    // Extract only the columns belonging to this response
    ws.RinvZ.resize(n, q_r);
    for (int k = 0; k < q_r; ++k) {
        int col_idx = response_group.random_effect_indices[k] - 1; // R is 1-based
        if (col_idx >= 0 && col_idx < random_effects_group.Z.cols()) {
            ws.RinvZ.col(k) = random_effects_group.Z.col(col_idx);
        }
    }
    ws.RinvZ *= inv_resid_var;

    ws.RinvX = response_group.X;
    ws.RinvX *= inv_resid_var;
    ws.Rinvy = response_group.y;
    ws.Rinvy.array() *= inv_resid_var;
    ws.logdetR = static_cast<double>(response_group.y.size()) * log_resid_variance;
}
