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

inline int total_observation_count(const std::vector<ResponseGroupData>& groups)
{
    int count = 0;
    for (const auto& group : groups)
        count += static_cast<int>(group.y.size());
    return count;
}

bool accumulate_groups_response(
    const PreparedModel& model,
    const ResponseModel& response,
    const MatrixXd& G_inv,
    MatrixXd& XtVinvX,
    VectorXd& XtVinvy,
    double& logdetV,
    double& quad);

class ResponseModel
{
public:
    virtual ~ResponseModel() = default;

    std::vector<ResponseGroupData> groups;

    virtual void initialise_from_prepared(const Rcpp::List& prepared) = 0;
    virtual int parameter_count() const = 0;
    virtual void initialise_working() = 0;
    virtual void update_working(const VectorXd& parameters) const = 0;
    virtual void update_working_from_theta(
        const VectorXd& theta,
        int q,
        int n_corr) const = 0;
    virtual bool parameters_are_valid(const VectorXd& parameters) const = 0;
    virtual bool parameters_are_valid_for_theta(
        const VectorXd& theta,
        int q,
        int n_corr) const = 0;
    virtual double residual_variance_penalty() const = 0;
    virtual void compute_group(
        const GroupData& random_effects_group,
        const ResponseGroupData& response_group,
        int group_index,
        GroupWorkspace& ws) const = 0;
    virtual VectorXd compute_beta_hat(
        const PreparedModel& prepared,
        const MatrixXd& G_inv,
        const VectorXd& theta) const = 0;
};
