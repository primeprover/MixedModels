#pragma once

#include "model_structs.hpp"
#include "covariance_utils.hpp"
#include <limits>

struct RandomEffectsSystem {
  MatrixXd G_inv;
  double logdetG = 0.0;
};

inline RandomEffectsSystem build_random_effects_system(
    int q,
    const VectorXd& log_sds,
    const VectorXd& corrs_unconstrained)
{
  RandomEffectsSystem system;
  VectorXd theta = VectorXd::Zero(q + corrs_unconstrained.size());
  theta.head(q) = log_sds;
  theta.segment(q, corrs_unconstrained.size()) = corrs_unconstrained;
  MatrixXd L = build_cholesky(theta, q);

  MatrixXd G = L * L.transpose();
  G.diagonal().array() += 1e-8;

  Eigen::LLT<MatrixXd> llt_G(G);
  if (llt_G.info() != Eigen::Success) {
    system.G_inv = MatrixXd();
    system.logdetG = -std::numeric_limits<double>::infinity();
    return system;
  }

  system.G_inv = llt_G.solve(MatrixXd::Identity(q, q));
  system.logdetG = 2.0 * llt_G.matrixLLT().diagonal().array().log().sum();
  return system;
}

inline void random_effects_begin(
    const GroupData& group,
    const ResponseGroupWorkspace& response_workspace,
    RandomEffectsIntegrationWorkspace& ws)
{
    ws.ZtRinvZ.noalias() = group.Z.transpose() * response_workspace.RinvZ;
    ws.ZtRinvX.noalias() = group.Z.transpose() * response_workspace.RinvX;
    ws.ZtRinvy.noalias() = group.Z.transpose() * response_workspace.Rinvy;
}

inline void random_effects_finalize(
    const Eigen::LLT<MatrixXd>& llt,
    const ResponseGroupWorkspace& response_workspace,
    RandomEffectsIntegrationWorkspace& ws)
{
    ws.KinvZtRinvX = llt.solve(ws.ZtRinvX);
    ws.KinvZtRinvy = llt.solve(ws.ZtRinvy);
    ws.VinvX.noalias() = response_workspace.RinvX - response_workspace.RinvZ * ws.KinvZtRinvX;
    ws.Vinvy.noalias() = response_workspace.Rinvy - response_workspace.RinvZ * ws.KinvZtRinvy;
}

inline bool random_effects_compute(
    const GroupData& group,
    const MatrixXd& G_inv,
    const ResponseGroupWorkspace& response_workspace,
    RandomEffectsIntegrationWorkspace& ws)
{
    random_effects_begin(group, response_workspace, ws);

    ws.K = G_inv;
    ws.K.noalias() += ws.ZtRinvZ;

    ws.llt.compute(ws.K);
    if (ws.llt.info() != Eigen::Success)
        return false;

    ws.logdetK = 2.0 * ws.llt.matrixLLT().diagonal().array().log().sum();
    random_effects_finalize(ws.llt, response_workspace, ws);

    return true;
}
