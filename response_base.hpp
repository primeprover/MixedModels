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

    ConditionalContribution() = default;
    explicit ConditionalContribution(int p)
        : XtVinvX(MatrixXd::Zero(p, p)), XtVinvy(VectorXd::Zero(p)) {}

};

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
};
