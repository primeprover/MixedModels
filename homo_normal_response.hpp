#pragma once

#include "response_base.hpp"

void compute_homo_normal_group(
    const GroupData& random_effects_group,
    const ResponseGroupData& response_group,
    double inv_resid_var,
    double log_resid_variance,
    GroupWorkspace& ws);

class HomoNormalResponse : public ResponseModel
{
public:
    explicit HomoNormalResponse(const Rcpp::List& prepared);

    void initialise_from_prepared(const Rcpp::List& prepared) override;
    int parameter_count() const override;
    void initialise_working() override;
    void update_working(const VectorXd& parameters) const override;
    bool parameters_are_valid(const VectorXd& parameters) const override;
    double residual_variance_penalty() const override;
    void compute_group(const GroupData& random_effects_group,
                       const ResponseGroupData& response_group,
                       int group_index,
                       GroupWorkspace& ws) const override;
    VectorXd compute_beta_hat(
        const PreparedModel& prepared,
        const MatrixXd& G_inv,
        const VectorXd& theta) const override;

private:
    mutable double log_resid_variance_ = 0.0;
    mutable double inv_resid_var_ = 1.0;
};
