#pragma once

#include "response_base.hpp"
#include <vector>

void compute_hetero_normal_group(
    const GroupData& random_effects_group,
    const ResponseGroupData& response_group,
    const VectorXd& inv_resid_var,
    const VectorXd& eta,
    GroupWorkspace& ws);

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
    void initialise_working() override;
    void update_working(const VectorXd& parameters) override;
    bool parameters_are_valid(const VectorXd& parameters) const override;
    double residual_variance_penalty() const override;
    void compute_group(const GroupData& random_effects_group,
                       const ResponseGroupData& response_group,
                       int group_index,
                       GroupWorkspace& ws) const override;

private:
    std::vector<HeteroResponseWorkingGroup> working_groups_;
};
