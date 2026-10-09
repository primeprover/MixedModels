// contribution_accumulator.hpp - Multi-response conditional-contribution kernel.
//
// The single group-parallel routine that computes the cross-response-coupled
// fixed-effects information (Schur complement + Woodbury) for a PreparedModel.
// Kept out of reml_loglik.cpp so the R-facing entry points stay the main file.
#pragma once

#include "response_base.hpp"   // PreparedModel, ConditionalContribution, ResponseModel

bool accumulate_conditional_contributions(
    const PreparedModel& model,
    const MatrixXd& G_inv_global,
    ConditionalContribution& contribution);
