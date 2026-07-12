#pragma once

#include "response_base.hpp"
#include <vector>

struct HeteroResponseWorkingGroup {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VectorXd eta;
    VectorXd inv_resid_var;
};

class HeteroNormalResponse : public ResponseModel
{
public:
    explicit HeteroNormalResponse(const Rcpp::List& prepared)
    {
        groups = extract_response_groups(prepared, true);
    }

    int parameter_count() const override
    {
        return groups.empty() ? 0 : groups.front().W.cols();
    }

    void initialise_working() override
    {
        working_groups_.resize(groups.size());
        for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
            const int n = static_cast<int>(groups[g].y.size());
            working_groups_[g].eta.resize(n);
            working_groups_[g].inv_resid_var.resize(n);
        }
    }

    void update_working(const VectorXd& parameters) override
    {
        for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
            working_groups_[g].eta.noalias() = groups[g].W * parameters;
            working_groups_[g].inv_resid_var.array() =
                (-working_groups_[g].eta.array()).exp();
        }
    }

    bool parameters_are_valid(const VectorXd& parameters) const override
    {
        return parameters.size() == parameter_count();
    }

    double residual_variance_penalty() const override
    {
        double penalty = 0.0;
        for (const auto& group : working_groups_)
            penalty += group.eta.squaredNorm();
        return penalty;
    }

    void compute_group(const GroupData& random_effects_group,
                       const ResponseGroupData& response_group,
                       int group_index,
                       GroupWorkspace& ws) const override
    {
        const auto& working_group = working_groups_[group_index];
        compute_hetero_normal_group(
            random_effects_group,
            response_group,
            working_group.inv_resid_var,
            working_group.eta,
            ws);
    }

private:
    std::vector<HeteroResponseWorkingGroup> working_groups_;
};
