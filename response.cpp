
// [[Rcpp::depends(RcppEigen)]]

#include "response.hpp"

void HeteroNormalResponse::update_working(
    PreparedModel& model,
    const Eigen::VectorXd& gamma
)
{
  //std::cout << "Updating working model" << std::endl;
  //std::cout << "Number of groups: " << model.groups.size() << std::endl;
  //model.working.groups.resize(model.groups.size());
  for (int g = 0; g < (int)model.groups.size(); ++g) {
    //std::cout << "Group " << g << std::endl;
    const auto& group = model.groups[g];
    
    //std::cout << "  W: "
    //          << group.W.rows() << " x "
    //          << group.W.cols() << std::endl;
    
    //std::cout << "  gamma: "
     //         << gamma.size() << std::endl;

    //std::cout << "  eta size: "
    //          << model.working.groups[g].eta.size() << std::endl;

    //std::cout << "  Before multiply" << std::endl;

    model.working.groups[g].eta.noalias() =
        group.W * gamma;

    model.working.groups[g].inv_resid_var.array() =
        (-model.working.groups[g].eta.array()).exp();
  }
  //std::cout << "Updated working model" << std::endl;
}


inline void HeteroNormalResponse::compute_group(
    const GroupData& group,
    const WorkingGroup& working,
    GroupWorkspace& ws) const
{
    const auto& inv_resid_var = working.inv_resid_var;

    ws.RinvZ =
        group.Z.array().colwise() * inv_resid_var.array();

    ws.RinvX =
        group.X.array().colwise() * inv_resid_var.array();

    ws.Rinvy =
        group.y.array() * inv_resid_var.array();

    ws.logdetR = working.eta.sum();
}
