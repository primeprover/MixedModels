#pragma once

#include "response_base.hpp"
#include <vector>

void compute_hetero_normal_group(
    const GroupData& random_effects_group,
    const ResponseGroupData& response_group,
    const VectorXd& inv_resid_var,
    const VectorXd& eta,
    ResponseGroupWorkspace& ws);

struct HeteroResponseWorkingGroup {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VectorXd eta;
    VectorXd inv_resid_var;
};

class HeteroNormalResponse : public ResponseModel
{
public:
    explicit HeteroNormalResponse(const Rcpp::List& prepared);

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
    VectorXd compute_beta_hat(
        const PreparedModel& prepared,
        const MatrixXd& G_inv,
        const VectorXd& theta) const override;

private:
    mutable std::vector<HeteroResponseWorkingGroup> working_groups_;
};
