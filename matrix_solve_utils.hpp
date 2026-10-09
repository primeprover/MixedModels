// matrix_solve_utils.hpp - Robust Cholesky (LLT) factor/solve + log-determinant
// helpers. Pure, response-independent matrix operations shared by the
// fixed-effects solve (solve_fe) and the information-matrix determinant
// (logdet_chol) in reml_loglik.cpp.
#pragma once

#include <cmath>
#include "model_structs.hpp"   // Eigen (via RcppEigen); VectorXd / MatrixXd

inline bool robust_llt_solve(const MatrixXd& A, const VectorXd& b,
                             VectorXd& x, double& logdetA)
{
  const int n = A.rows();
  double maxdiag = 0.0;
  for (int i = 0; i < n; ++i) {
    maxdiag = std::max(maxdiag, std::fabs(A(i, i)));
  }
  const double scale = std::max<double>(1.0, maxdiag);
  const double jitter[7] = { 0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2 };
  for (int k = 0; k < 7; ++k) {
    MatrixXd A2 = A;
    if (jitter[k] > 0.0) {
      VectorXd d = A2.diagonal();
      d += (jitter[k] * scale) * VectorXd::Ones(n);
      A2.diagonal() = d;
    }
    Eigen::LLT<MatrixXd> l(A2);
    if (l.info() == Eigen::Success) {
      x = l.solve(b);
      logdetA = 2.0 * l.matrixLLT().diagonal().array().log().sum();
      return true;
    }
  }
  return false;
}

inline VectorXd solve_fe(const MatrixXd& XtVinvX, const VectorXd& XtVinvy, bool& success)
{
  VectorXd x;
  double ld;
  success = robust_llt_solve(XtVinvX, XtVinvy, x, ld);
  return x;
}

inline double logdet_chol(const MatrixXd& X, bool& success)
{
  const int n = X.rows();
  const VectorXd zero = VectorXd::Zero(n);
  VectorXd dummy;
  double ld;
  success = robust_llt_solve(X, zero, dummy, ld);
  return ld;
}
