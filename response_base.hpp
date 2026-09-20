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
    bool observed = true;
    std::vector<int> random_effect_indices;
};

struct ConditionalContribution {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MatrixXd XtVinvX;
    VectorXd XtVinvy;
    double logdetV = 0.0;
    double quad = 0.0;

    explicit ConditionalContribution(int p)
        : XtVinvX(MatrixXd::Zero(p, p)), XtVinvy(VectorXd::Zero(p)) {}

    void add_group_contribution(
        const ResponseGroupData& response_group,
        const ResponseGroupWorkspace& response_workspace,
        const RandomEffectsIntegrationWorkspace& integration_workspace,
        int fixed_effect_offset,
        int fixed_effect_dimension)
    {
        // std::cout << "Got here3.9.4 " << std::endl;
        const MatrixXd info_block = response_group.X.transpose() * integration_workspace.VinvX;
        const VectorXd vinvy_block = response_group.X.transpose() * integration_workspace.Vinvy;
        XtVinvX.block(fixed_effect_offset, fixed_effect_offset, fixed_effect_dimension, fixed_effect_dimension).noalias() += info_block;
        // std::cout << "Got here3.9.5 " << std::endl;
        XtVinvy.segment(fixed_effect_offset, fixed_effect_dimension).noalias() += vinvy_block;
        quad += response_group.y.dot(integration_workspace.Vinvy);
        logdetV += response_workspace.logdetR + integration_workspace.logdetK;
    }

    void merge_from(const ConditionalContribution& other)
    {
        XtVinvX.noalias() += other.XtVinvX;
        XtVinvy.noalias() += other.XtVinvy;
        logdetV += other.logdetV;
        quad += other.quad;
    }
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
    int parameter_offset_ = 0;
    int fixed_effect_offset_ = 0;

    int parameter_offset() const { return parameter_offset_; }
    void set_parameter_offset(int value) { parameter_offset_ = value; }

    int fixed_effect_offset() const { return fixed_effect_offset_; }
    void set_fixed_effect_offset(int value) { fixed_effect_offset_ = value; }

    virtual void initialise_from_prepared(const Rcpp::List& prepared) = 0;
    virtual int parameter_count() const = 0;
    virtual int fixed_effect_dimension() const = 0;
    virtual void initialise_working() = 0;
    virtual void update_working(const VectorXd& parameters) const = 0;
    virtual void update_working_from_theta(
        const VectorXd& theta,
        int q,
        int n_corr,
        int parameter_offset) const = 0;
    virtual bool parameters_are_valid(const VectorXd& parameters) const = 0;
    virtual bool parameters_are_valid_for_theta(
        const VectorXd& theta,
        int q,
        int n_corr,
        int parameter_offset) const = 0;
    virtual double residual_variance_penalty() const = 0;
    virtual void compute_group(
        const GroupData& random_effects_group,
        const ResponseGroupData& response_group,
        int group_index,
        ResponseGroupWorkspace& ws) const = 0;
    virtual VectorXd compute_beta_hat(
        const PreparedModel& prepared,
        const MatrixXd& G_inv,
        const VectorXd& theta) const = 0;
};
