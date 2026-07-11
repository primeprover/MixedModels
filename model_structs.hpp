#pragma once

#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Eigen;


struct WorkingGroup {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VectorXd eta;          // linear predictor for log(residual variance)
  VectorXd inv_resid_var; // 1 / residual variances
};

struct WorkingModel {
  std::vector<WorkingGroup> groups; // one per group
};

// Group data structure: precomputed matrices for a single group
struct GroupData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MatrixXd X;      // design matrix for fixed effects
  MatrixXd Z;      // design matrix for random effects
  VectorXd y;      // response vector
  int n;           // number of observations
  MatrixXd ZtZ;    // Z'Z (precomputed)
  VectorXd ZtY;    // Z'y (precomputed)
  MatrixXd ZtX;    // Z'X (precomputed)
  MatrixXd W;      // design matrix for residual variance (if heteroscedastic)
};

struct PreparedModel {

  // static scalar parameters
  int p;
  int q;
  int n_corr;
  bool heteroscedastic;

  double lo_sigma;
  double hi_sigma;

  // static data
  VectorXd y;
  MatrixXd W; // design matrix for residual variance (if heteroscedastic)

  // grouped structure
  std::vector<GroupData> groups;

  // working model - Mutable state that is updated during optimization
  WorkingModel working;
};

struct GroupWorkspace {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Response model outputs
    MatrixXd RinvZ;
    MatrixXd RinvX;
    VectorXd Rinvy;

    // Random effects intermediates
    MatrixXd ZtRinvZ;
    MatrixXd ZtRinvX;
    VectorXd ZtRinvy;

    MatrixXd K;

    MatrixXd KinvZtRinvX;
    VectorXd KinvZtRinvy;

    // Final outputs
    MatrixXd VinvX;
    VectorXd Vinvy;

    double logdetR = 0.0;
    double logdetK = 0.0;

    Eigen::LLT<MatrixXd> llt;
};
