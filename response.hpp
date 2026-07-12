#include "model_structs.hpp"

struct ResponseGroupData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MatrixXd X;
    VectorXd y;
    VectorXd ZtY;
    MatrixXd ZtX;
    MatrixXd W; // Empty for responses without a variance-design matrix.
};
struct ResponseWorkingGroup {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VectorXd eta;
    VectorXd inv_resid_var;
};

struct ResponseWorking {
    std::vector<ResponseWorkingGroup> groups;
};

class ResponseModel
{
public:
    virtual ~ResponseModel() = default;

    // Immutable response-specific data and parameter-dependent state.
    std::vector<ResponseGroupData> groups;
    ResponseWorking working;

    virtual int parameter_count() const = 0;
    virtual void initialise_working() = 0;
    virtual void update_working(const VectorXd& parameters) = 0;
    virtual bool parameters_are_valid(const VectorXd& parameters) const = 0;
    virtual double residual_variance_penalty() const = 0;

    virtual void compute_group(
        const GroupData& random_effects_group,
        const ResponseGroupData& response_group,
        const ResponseWorkingGroup& response_working,
        GroupWorkspace& ws) const = 0;
};

class HomoNormalResponse : public ResponseModel
{
public:
    explicit HomoNormalResponse(const Rcpp::List& prepared);

    int parameter_count() const override { return 1; }
    void initialise_working() override;
    void update_working(const VectorXd& parameters) override;
    bool parameters_are_valid(const VectorXd& parameters) const override;
    double residual_variance_penalty() const override;
    void compute_group(const GroupData& random_effects_group,
                       const ResponseGroupData& response_group,
                       const ResponseWorkingGroup& response_working,
                       GroupWorkspace& workspace) const override;
};

class HeteroNormalResponse : public ResponseModel
{
public:
    explicit HeteroNormalResponse(const Rcpp::List& prepared);

    int parameter_count() const override;
    void initialise_working() override;
    void update_working(const VectorXd& gamma) override;
    bool parameters_are_valid(const VectorXd& parameters) const override;
    double residual_variance_penalty() const override;

    void compute_group(const GroupData& random_effects_group,
                       const ResponseGroupData& response_group,
                       const ResponseWorkingGroup& response_working,
                       GroupWorkspace& workspace) const override;
};
