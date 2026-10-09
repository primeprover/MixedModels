#pragma once

#include <Rcpp.h>
#include <RcppEigen.h>
#include <memory>
#include <vector>
using namespace Eigen;

class ResponseModel;

// Data shared by all responses in a model.  Response-specific X/y/W data live
// on the ResponseModel instead.
struct GroupData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MatrixXd Z;      // design matrix for random effects
  int n;           // number of observations
};


struct PreparedModel {
  std::vector<int> global_random_effect_sizes;
  std::vector<int> global_random_effect_offsets;

  // static scalar parameters
  int q;
  int n_corr;

  double lo_sigma;
  double hi_sigma;
  double residual_variance_penalty_weight;

  // Shared random-effects structure.
  std::vector<GroupData> groups;

  // Each response owns its data and mutable working state.  The current R
  // interface creates one response, but this supports independent outcomes.
  std::vector<std::unique_ptr<ResponseModel>> responses;

  ~PreparedModel();
};

struct ResponseGroupWorkspace {
  std::vector<int> random_effect_indices;
  bool observed = true;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MatrixXd RinvZ;
  MatrixXd RinvX;
  VectorXd Rinvy;

  double logdetR = 0.0;
};