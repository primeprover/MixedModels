#include "model_structs.hpp"

class ResponseModel
{
public:
    virtual ~ResponseModel() = default;

    virtual void update_working(
        PreparedModel& model,
        const VectorXd& gamma) = 0;

    virtual void compute_group(
        const GroupData& group,
        const WorkingGroup& working,
        GroupWorkspace& ws) const = 0;
};

class HeteroNormalResponse : public ResponseModel
{
public:

    void update_working(
        PreparedModel& model,
        const VectorXd& gamma) override;

    void compute_group(
        const GroupData& group,
        const WorkingGroup& working,
        GroupWorkspace& workspace) const override;
};

