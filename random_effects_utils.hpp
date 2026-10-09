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
