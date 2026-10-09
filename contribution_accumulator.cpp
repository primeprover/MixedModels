// contribution_accumulator.cpp - Multi-response conditional-contribution kernel.
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]
#include "contribution_accumulator.hpp"
#include "response.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

bool accumulate_conditional_contributions(
    const PreparedModel& model,
    const MatrixXd& G_inv_global,
    ConditionalContribution& contribution)
{
  if (model.responses.empty()) {
    return false;
  }

  int P_total = 0;
  for (const auto& resp : model.responses) {
    P_total += resp->fixed_effect_dimension();
  }
  if (P_total <= 0) {
    return false;
  }

  contribution = ConditionalContribution(P_total);
  const auto& groups = model.groups;
  bool bad = false;

  Eigen::setNbThreads(1);
  #pragma omp parallel
  {
    ConditionalContribution local_contribution(P_total);
    std::vector<ResponseGroupWorkspace> response_workspaces(model.responses.size());
    std::vector<MatrixXd> M_r(model.responses.size());
    std::vector<VectorXd> B_r(model.responses.size());
    bool bad_local = false;

    MatrixXd lXtRX(P_total, P_total);
    lXtRX.setZero();
    VectorXd lXtRy(P_total);
    lXtRy.setZero();
    MatrixXd lCdiag(P_total, P_total);
    lCdiag.setZero();
    VectorXd lc_vec(P_total);
    lc_vec.setZero();
    MatrixXd lCross(P_total, P_total);
    lCross.setZero();

    #pragma omp for schedule(dynamic, 1)
    for (int ig = 0; ig < (int)groups.size(); ++ig) {
      const auto& group = groups[ig];
      
      MatrixXd sZtRZ(model.q, model.q);
      sZtRZ.setZero();
      for (int ir = 0; ir < (int)model.responses.size(); ++ir) {
        const ResponseModel& resp = *model.responses[ir];
        const ResponseGroupData& rg = resp.groups[ig];
        if (!rg.observed) {
          continue;
        }
        resp.compute_group(group, rg, ig, response_workspaces[ir]);
        const int Q = (int)rg.random_effect_indices.size();
        MatrixXd Zr(group.Z.rows(), Q);
        for (int k = 0; k < Q; ++k) {
          const int idx = rg.random_effect_indices[k] - 1;
          if (idx >= 0 && idx < group.Z.cols()) {
            Zr.col(k) = group.Z.col(idx);
          }
        }
        MatrixXd blk = Zr.transpose() * response_workspaces[ir].RinvZ;
        for (int i = 0; i < Q; ++i) {
          const int gi = rg.random_effect_indices[i] - 1;
          for (int j = 0; j <= i; ++j) {
            const int gj = rg.random_effect_indices[j] - 1;
            if (gi >= 0 && gj >= 0 && gi < model.q && gj < model.q) {
              sZtRZ(gi, gj) += blk(i, j);
              if (i != j) {
                sZtRZ(gj, gi) += blk(i, j);
              }
            }
          }
        }
      }

      MatrixXd K = G_inv_global + sZtRZ;
      Eigen::LLT<MatrixXd> llt(K);
      if (llt.info() != Eigen::Success) {
        bad_local = true;
        continue;
      }
      double logdetK = 2.0 * llt.matrixLLT().diagonal().array().log().sum();
      MatrixXd Kinv = llt.solve(MatrixXd::Identity(model.q, model.q));

      int nor = 0;
      for (int ir = 0; ir < (int)model.responses.size(); ++ir) {
        const ResponseModel& resp = *model.responses[ir];
        const ResponseGroupData& rg = resp.groups[ig];
        if (!rg.observed) {
          continue;
        }
        ++nor;
        const int Q = (int)rg.random_effect_indices.size();
        MatrixXd Zr(group.Z.rows(), Q);
        for (int k = 0; k < Q; ++k) {
          const int idx = rg.random_effect_indices[k] - 1;
          if (idx >= 0 && idx < group.Z.cols()) {
            Zr.col(k) = group.Z.col(idx);
          }
        }
        M_r[ir].noalias() = rg.X.transpose() * response_workspaces[ir].RinvZ;
        B_r[ir].noalias() = Zr.transpose() * response_workspaces[ir].Rinvy;
        local_contribution.logdetV += response_workspaces[ir].logdetR;
        local_contribution.quad += rg.y.dot(response_workspaces[ir].Rinvy);
      }
      local_contribution.logdetV += logdetK;

      // Assemble Mt: each response contributes its Mr at the correct global rows
      MatrixXd Mt(model.q, P_total);
      Mt.setZero();
      VectorXd Bt_vec(model.q);
      Bt_vec.setZero();
      for (int ir = 0; ir < (int)model.responses.size(); ++ir) {
        const ResponseModel& resp = *model.responses[ir];
        const ResponseGroupData& rg = resp.groups[ig];
        if (!rg.observed) {
          continue;
        }
        const int Q = (int)rg.random_effect_indices.size();
        const int pr = resp.fixed_effect_dimension();
        for (int k = 0; k < Q; ++k) {
          const int gi = rg.random_effect_indices[k] - 1;
          if (gi >= 0 && gi < model.q) {
            Mt.row(gi).segment(resp.fixed_effect_offset(), pr) =
                M_r[ir].col(k).transpose();
          }
        }
        for (int k = 0; k < Q; ++k) {
          const int gi = rg.random_effect_indices[k] - 1;
          if (gi >= 0 && gi < model.q) {
            Bt_vec(gi) = B_r[ir](k);
          }
        }
      }
      MatrixXd gf = llt.solve(Mt);
      VectorXd zf = llt.solve(Bt_vec);
      MatrixXd Cg = Mt.transpose() * gf;
      VectorXd cg = Mt.transpose() * zf;

      // Accumulate diagonal blocks of X^T R^{-1} X
      for (int ir = 0; ir < (int)model.responses.size(); ++ir) {
        const ResponseModel& resp = *model.responses[ir];
        const ResponseGroupData& rg = resp.groups[ig];
        if (!rg.observed) {
          continue;
        }
        const int pr = resp.fixed_effect_dimension();
        const int off = resp.fixed_effect_offset();
        lXtRX.block(off, off, pr, pr).noalias() +=
            rg.X.transpose() * response_workspaces[ir].RinvX;
        lXtRy.segment(off, pr).noalias() +=
            rg.X.transpose() * response_workspaces[ir].Rinvy;
      }
      // Accumulate diagonal Woodbury correction
      for (int ir = 0; ir < (int)model.responses.size(); ++ir) {
        const ResponseModel& resp = *model.responses[ir];
        const int pr = resp.fixed_effect_dimension();
        const int off = resp.fixed_effect_offset();
        lCdiag.block(off, off, pr, pr).noalias() += Cg.block(off, off, pr, pr);
      }
      lc_vec.noalias() += cg;

      // Cross-response Woodbury corrections
      // For conditionally independent responses, X_a^T R^{-1} X_b = 0
      // So off-diagonal block = -M_a^T K^{-1} M_b
      int off_a = 0;
      for (int ia = 0; ia < (int)model.responses.size(); ++ia) {
        const ResponseModel& resa = *model.responses[ia];
        const ResponseGroupData& rga = resa.groups[ig];
        if (!rga.observed) {
          continue;
        }
        const int pra = resa.fixed_effect_dimension();
        MatrixXd Mta = MatrixXd::Zero(model.q, pra);
        for (int k = 0; k < (int)rga.random_effect_indices.size(); ++k) {
          const int gi = rga.random_effect_indices[k] - 1;
          if (gi < 0 || gi >= model.q) {
            continue;
          }
          Mta.row(gi).segment(0, pra) = M_r[ia].col(k).transpose();
        }
        for (int ib = ia + 1; ib < (int)model.responses.size(); ++ib) {
          const ResponseModel& resb = *model.responses[ib];
          const ResponseGroupData& rgb = resb.groups[ig];
          if (!rgb.observed) {
            continue;
          }
          const int prb = resb.fixed_effect_dimension();
          MatrixXd Mtb = MatrixXd::Zero(model.q, prb);
          for (int k = 0; k < (int)rgb.random_effect_indices.size(); ++k) {
            const int gi = rgb.random_effect_indices[k] - 1;
            if (gi < 0 || gi >= model.q) {
              continue;
            }
            Mtb.row(gi).segment(0, prb) = M_r[ib].col(k).transpose();
          }
          MatrixXd cross = Mta.transpose() * Kinv * Mtb;
          lCross.block(off_a, off_a + pra, pra, prb).noalias() += cross;
          lCross.block(off_a + pra, off_a, prb, pra).noalias() += cross.transpose();
        }
        off_a += pra;
      }

      // Bug 3: Full quadratic form
      if (nor > 1) {
        double bkb = 0.0;
        for (int ir = 0; ir < (int)model.responses.size(); ++ir) {
          if (!model.responses[ir]->groups[ig].observed) {
            continue;
          }
          const VectorXd& B1 = B_r[ir];
          if (B1.size() == 0) {
            continue;
          }
          for (int is_ = 0; is_ < (int)model.responses.size(); ++is_) {
            if (!model.responses[is_]->groups[ig].observed) {
              continue;
            }
            const VectorXd& B2 = B_r[is_];
            if (B2.size() == 0) {
              continue;
            }
            double t = 0.0;
            for (int ki = 0; ki < (int)B1.size(); ++ki) {
              const int gi = model.responses[ir]->groups[ig].random_effect_indices[ki] - 1;
              if (gi < 0 || gi >= model.q) {
                continue;
              }
              for (int kj = 0; kj < (int)B2.size(); ++kj) {
                const int gj = model.responses[is_]->groups[ig].random_effect_indices[kj] - 1;
                if (gj < 0 || gj >= model.q) {
                  continue;
                }
                t += B1(ki) * Kinv(gi, gj) * B2(kj);
              }
            }
            if (ir == is_) {
              bkb += t;
            } else if (ir < is_) {
              bkb += 2.0 * t;
            }
          }
        }
        local_contribution.quad -= bkb;
      } else if (nor == 1) {
        for (int ir = 0; ir < (int)model.responses.size(); ++ir) {
          if (!model.responses[ir]->groups[ig].observed) {
            continue;
          }
          const VectorXd& B = B_r[ir];
          if (B.size() == 0) {
            continue;
          }
          double t = 0.0;
          for (int ki = 0; ki < (int)B.size(); ++ki) {
            const int gi = model.responses[ir]->groups[ig].random_effect_indices[ki] - 1;
            if (gi < 0 || gi >= model.q) {
              continue;
            }
            for (int kj = 0; kj < (int)B.size(); ++kj) {
              const int gj = model.responses[ir]->groups[ig].random_effect_indices[kj] - 1;
              if (gj < 0 || gj >= model.q) {
                continue;
              }
              t += B(ki) * Kinv(gi, gj) * B(kj);
            }
          }
          local_contribution.quad -= t;
          break;
        }
      }
    }

    #pragma omp critical
    {
      contribution.logdetV += local_contribution.logdetV;
      if (bad_local) {
        bad = true;
      }
      contribution.quad += local_contribution.quad;
      // Diagonal blocks: X^T R^{-1} X - Cg(diag)
      for (size_t ir = 0; ir < model.responses.size(); ++ir) {
        const auto& resp = *model.responses[ir];
        const int pr = resp.fixed_effect_dimension();
        const int off = resp.fixed_effect_offset();
        contribution.XtVinvX.block(off, off, pr, pr).noalias() +=
            lXtRX.block(off, off, pr, pr) - lCdiag.block(off, off, pr, pr);
      }
      // Off-diagonal blocks: -M_a^T K^{-1} M_b = -lCross
      for (size_t ia = 0; ia < model.responses.size(); ++ia) {
        const auto& resa = *model.responses[ia];
        const int pra = resa.fixed_effect_dimension();
        const int offa = resa.fixed_effect_offset();
        for (size_t ib = ia + 1; ib < model.responses.size(); ++ib) {
          const auto& resb = *model.responses[ib];
          const int prb = resb.fixed_effect_dimension();
          const int offb = resb.fixed_effect_offset();
          contribution.XtVinvX.block(offa, offb, pra, prb).noalias() +=
              -lCross.block(offa, offb, pra, prb);
          contribution.XtVinvX.block(offb, offa, prb, pra).noalias() +=
              -lCross.block(offb, offa, prb, pra);
        }
      }
      contribution.XtVinvy.noalias() += (lXtRy - lc_vec);
    }
  }
  return !bad;
}
