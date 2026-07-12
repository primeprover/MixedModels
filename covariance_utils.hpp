#pragma once

#include "model_structs.hpp"

using namespace Eigen;

inline MatrixXd build_cholesky(const VectorXd& theta, int q)
{
  VectorXd log_sds = theta.head(q);
  VectorXd sds = log_sds.array().exp().matrix();

  int n_corr = q * (q - 1) / 2;
  VectorXd corrs_unconstrained = theta.segment(q, n_corr);
  VectorXd corrs = corrs_unconstrained.array().tanh().matrix();

  MatrixXd L_corr = MatrixXd::Zero(q, q);

  int k = 0;
  for (int i = 0; i < q; ++i) {
    double row_prod = 1.0;
    for (int j = 0; j < i; ++j) {
      L_corr(i, j) = corrs(k) * row_prod;
      row_prod *= std::sqrt(1.0 - corrs(k) * corrs(k));
      k++;
    }
    L_corr(i, i) = row_prod;
  }

  for (int i = 0; i < q; ++i) {
    L_corr.row(i) *= sds(i);
  }

  return L_corr;
}
