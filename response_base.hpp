#pragma once

// [[Rcpp::depends(RcppEigen)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include "model_structs.hpp"

using namespace Eigen;

struct ResponseGroupData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MatrixXd X;
    VectorXd y;
    VectorXd ZtY;
    MatrixXd ZtX;
    MatrixXd W; // Empty for responses without a variance-design matrix.
};

inline std::vector<ResponseGroupData> extract_response_groups(
    const Rcpp::List& prepared,
    bool include_variance_design)
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

inline void compute_homo_normal_group(
    const GroupData& random_effects_group,
    const ResponseGroupData& response_group,
    double inv_resid_var,
    double log_resid_variance,
    GroupWorkspace& ws)
{
    ws.RinvZ = random_effects_group.Z;
    ws.RinvX = response_group.X;
    ws.Rinvy = response_group.y;
    ws.RinvZ *= inv_resid_var;
    ws.RinvX *= inv_resid_var;
    ws.Rinvy.array() *= inv_resid_var;
    ws.logdetR = static_cast<double>(response_group.y.size()) * log_resid_variance;
}

inline void compute_hetero_normal_group(
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

inline int total_observation_count(const std::vector<ResponseGroupData>& groups)
{
    int count = 0;
    for (const auto& group : groups)
        count += static_cast<int>(group.y.size());
    return count;
}

class ResponseModel
{
public:
    virtual ~ResponseModel() = default;

    std::vector<ResponseGroupData> groups;

    virtual int parameter_count() const = 0;
    virtual void initialise_working() = 0;
    virtual void update_working(const VectorXd& parameters) = 0;
    virtual bool parameters_are_valid(const VectorXd& parameters) const = 0;
    virtual double residual_variance_penalty() const = 0;
    virtual void compute_group(
        const GroupData& random_effects_group,
        const ResponseGroupData& response_group,
        int group_index,
        GroupWorkspace& ws) const = 0;
};
