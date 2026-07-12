#pragma once

#include "response_base.hpp"
#include <cmath>
#include <limits>

class HomoNormalResponse : public ResponseModel
{
public:
    explicit HomoNormalResponse(const Rcpp::List& prepared)
    {
        groups = extract_response_groups(prepared, false);
    }

    int parameter_count() const override
    {
        return 1;
    }

    void initialise_working() override
    {
        // No per-observation working buffers needed for the homo-normal scalar path.
    }

    void update_working(const VectorXd& parameters) override
    {
        log_resid_variance_ = parameters(0);
        inv_resid_var_ = std::exp(-log_resid_variance_);
    }

    bool parameters_are_valid(const VectorXd& parameters) const override
    {
        return parameters.size() == 1 &&
            0.5 * parameters(0) >= std::log(std::numeric_limits<double>::epsilon());
    }

    double residual_variance_penalty() const override
    {
        const int n = total_observation_count(groups);
        return static_cast<double>(n) * (log_resid_variance_ * log_resid_variance_);
    }

    void compute_group(const GroupData& random_effects_group,
                       const ResponseGroupData& response_group,
                       int group_index,
                       GroupWorkspace& ws) const override
    {
        compute_homo_normal_group(
            random_effects_group,
            response_group,
            inv_resid_var_,
            log_resid_variance_,
            ws);
    }

private:
    double log_resid_variance_ = 0.0;
    double inv_resid_var_ = 1.0;
};
