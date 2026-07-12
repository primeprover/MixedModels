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

inline bool random_effects_compute(
    const GroupData& group,
    const MatrixXd& G_inv,
    GroupWorkspace& ws)
{
    ws.ZtRinvZ.noalias() = group.Z.transpose() * ws.RinvZ;
    ws.ZtRinvX.noalias() = group.Z.transpose() * ws.RinvX;
    ws.ZtRinvy.noalias() = group.Z.transpose() * ws.Rinvy;

    ws.K = G_inv;
    ws.K.noalias() += ws.ZtRinvZ;

    ws.llt.compute(ws.K);
    if (ws.llt.info() != Eigen::Success)
        return false;

    ws.logdetK = 2.0 * ws.llt.matrixLLT().diagonal().array().log().sum();
    ws.KinvZtRinvX = ws.llt.solve(ws.ZtRinvX);
    ws.KinvZtRinvy = ws.llt.solve(ws.ZtRinvy);

    ws.VinvX.noalias() = ws.RinvX - ws.RinvZ * ws.KinvZtRinvX;
    ws.Vinvy.noalias() = ws.Rinvy - ws.RinvZ * ws.KinvZtRinvy;

    return true;
}
