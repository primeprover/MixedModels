#pragma once

#include "response_base.hpp"

void compute_homo_normal_group(
    const GroupData& random_effects_group,
    const ResponseGroupData& response_group,
    double inv_resid_var,
    double log_resid_variance,
    ResponseGroupWorkspace& ws);

class HomoNormalResponse : public ResponseModel
{
public:
    explicit HomoNormalResponse(const Rcpp::List& prepared);

    void initialise_from_prepared(const Rcpp::List& prepared) override;
    int parameter_count() const override;
    int fixed_effect_dimension() const override;
    void initialise_working() override;
    void update_working(const VectorXd& parameters) const override;
    void update_working_from_theta(
        const VectorXd& theta,
        int q,
        int n_corr,
        int parameter_offset) const override;
    bool parameters_are_valid(const VectorXd& parameters) const override;
    bool parameters_are_valid_for_theta(
        const VectorXd& theta,
        int q,
        int n_corr,
        int parameter_offset) const override;
    double residual_variance_penalty() const override;
    void compute_group(const GroupData& random_effects_group,
                       const ResponseGroupData& response_group,
                       int group_index,
                       ResponseGroupWorkspace& ws) const override;

private:
    mutable double log_resid_variance_ = 0.0;
    mutable double inv_resid_var_ = 1.0;
};
