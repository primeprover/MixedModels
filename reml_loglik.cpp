// reml_loglik.cpp - Fixed with full cross-response coupling
#include "reml_loglik.hpp"
#include "response.hpp"
#include "homo_normal_response.hpp"
#include "hetero_normal_response.hpp"
#include "response_factory.hpp"
#include "covariance_utils.hpp"
#include "random_effects_utils.hpp"
#include <cmath>
#include <vector>
#include <atomic>
#include <chrono>
#include <memory>
#include <limits>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

namespace {
std::vector<GroupData> extract_groups_from_payload(const Rcpp::List& prepared) {
  const Rcpp::List gl = prepared.containsElementNamed("shared_group_list")
      ? prepared["shared_group_list"] : prepared.containsElementNamed("group_list")
          ? prepared["group_list"] : Rcpp::List();
  const int n = gl.size(); std::vector<GroupData> groups(n);
  for (int i = 0; i < n; ++i) {
    const Rcpp::List gi = gl[i];
    groups[i].Z = Rcpp::as<MatrixXd>(gi["Z"]);
    groups[i].n = Rcpp::as<int>(gi["n"]);
    groups[i].ZtZ = Rcpp::as<MatrixXd>(gi["ZtZ"]);
  }
  return groups;
}
}
std::vector<GroupData> extract_groups(const Rcpp::List& prepared) { return extract_groups_from_payload(prepared); }
PreparedModel::~PreparedModel() = default;

namespace {
// Robustly factor a symmetric matrix expected to be positive semi-definite and
// solve A x = b, returning the log determinant. Tries Cholesky first, then
// applies increasing relative jitter. This handles the single-response
// hetero long model, whose fixed-effects information matrix can be singular
// to round-off when the random-effect variance is large and the fixed-effect
// design lies in the random-effect span (X = Z). Failing gracefully (instead
// of a hard crash) lets the optimizer steer around the ill-conditioned region.
bool robust_llt_solve(const MatrixXd& A, const VectorXd& b,
                      VectorXd& x, double& logdetA) {
  const int n = A.rows();
  double maxdiag = 0.0;
  for (int i = 0; i < n; ++i) maxdiag = std::max(maxdiag, std::fabs(A(i, i)));
  const double scale = std::max<double>(1.0, maxdiag);
  const double jitter[7] = {0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2};
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

VectorXd solve_fe(const MatrixXd& XtVinvX, const VectorXd& XtVinvy, bool& success) {
  VectorXd x; double ld;
  success = robust_llt_solve(XtVinvX, XtVinvy, x, ld);
  return x;
}
double logdet_chol(const MatrixXd& X, bool& success) {
  const int n = X.rows();
  const VectorXd zero = VectorXd::Zero(n);
  VectorXd dummy; double ld;
  success = robust_llt_solve(X, zero, dummy, ld);
  return ld;
}
}

// [[Rcpp::export]]
Rcpp::XPtr<PreparedModel> init_model_cpp(const Rcpp::List& prepared) {
  auto model = new PreparedModel();
  model->q=prepared["q"]; model->n_corr=prepared["n_corr"];
  model->lo_sigma=prepared["lo_sigma"]; model->hi_sigma=prepared["hi_sigma"];
  model->residual_variance_penalty_weight = prepared.containsElementNamed("residual_variance_penalty_weight")
      ? Rcpp::as<double>(prepared["residual_variance_penalty_weight"]) : 0.0;
  if (model->residual_variance_penalty_weight < 0.0) Rcpp::stop("penalty must be non-negative.");
  model->groups = extract_groups(prepared);
  if (prepared.containsElementNamed("response_spec")) {
    const Rcpp::List rs = prepared["response_spec"];
    if (rs.containsElementNamed("random_map")) {
      const Rcpp::IntegerVector rm = rs["random_map"];
      model->global_random_effect_sizes.assign(rm.begin(), rm.end());
      model->global_random_effect_offsets.assign(rm.size(), 0);
    }
  }
  if (prepared.containsElementNamed("responses")) {
    const Rcpp::List res = prepared["responses"];
    int po=0, fo=0;
    for (int i=0;i<res.size();++i) {
      auto r = make_response(res[i]);
      r->initialise_from_prepared(res[i]);
      r->set_parameter_offset(po); r->set_fixed_effect_offset(fo);
      r->initialise_working();
      po+=r->parameter_count(); fo+=r->fixed_effect_dimension();
      model->responses.push_back(std::move(r));
    }
  } else {
    auto r = make_response(prepared);
    r->initialise_from_prepared(prepared); r->initialise_working();
    model->responses.push_back(std::move(r));
  }
  return Rcpp::XPtr<PreparedModel>(model, true);
}

double compute_penalty(const PreparedModel& model, const VectorXd& log_sds, const ResponseModel& response) {
  double p=0.0;
  for (int i=0;i<model.q;++i) {
    p+=std::max(0.0,log_sds(i)-model.hi_sigma)*std::max(0.0,log_sds(i)-model.hi_sigma)
      +std::max(0.0,model.lo_sigma-log_sds(i))*std::max(0.0,model.lo_sigma-log_sds(i))
      +log_sds(i)*log_sds(i)*10000.0;
  }
  return p*1e-6 + model.residual_variance_penalty_weight*response.residual_variance_penalty();
}

bool accumulate_groups_response(const PreparedModel& model, const ResponseModel& response,
    const MatrixXd& G_inv, MatrixXd& XtVinvX, VectorXd& XtVinvy, double& logdetV, double& quad) {
  Eigen::setNbThreads(1);
  const auto& groups=model.groups; const int p=response.fixed_effect_dimension();
  ConditionalContribution contribution(p); std::atomic<bool> bad(false);
  { ConditionalContribution lc(p); ResponseGroupWorkspace rw; RandomEffectsIntegrationWorkspace iw;
    for (int ig=0;ig<(int)groups.size();++ig) {
      const auto& g=groups[ig]; const auto& rg=response.groups[ig];
      if (!rg.observed) continue;
      response.compute_group(g, rg, ig, rw);
      if (!random_effects_compute(g, G_inv, rw, iw)) { bad.store(true,std::memory_order_relaxed); continue; }
      lc.add_group_contribution(rg,rw,iw,response.fixed_effect_offset(),response.fixed_effect_dimension());
    }
    { contribution.merge_from(lc); }
  }
  XtVinvX=contribution.XtVinvX; XtVinvy=contribution.XtVinvy;
  logdetV=contribution.logdetV; quad=contribution.quad;
  return !bad.load();
}

// ============================================================
// MULTI-RESPONSE: Full cross-response coupling via Woodbury identity
// ============================================================
bool accumulate_conditional_contributions(const PreparedModel& model,
    const MatrixXd& G_inv_global, ConditionalContribution& contribution) {
  if (model.responses.empty()) return false;
  int P_total=0;
  for (const auto& r:model.responses) P_total+=r->fixed_effect_dimension();
  if (P_total<=0) return false;
  contribution = ConditionalContribution(P_total);
  const auto& groups=model.groups; bool bad=false;
  MatrixXd G_global;
  if (G_inv_global.size() > 0) {
    try { G_global = G_inv_global.inverse(); } catch (...) { G_global = MatrixXd(); }
  }
  Eigen::setNbThreads(1);
  #pragma omp parallel
  {
    ConditionalContribution lc(P_total);
    std::vector<ResponseGroupWorkspace> rws(model.responses.size());
    std::vector<MatrixXd> Mr(model.responses.size());
    std::vector<VectorXd> Br(model.responses.size());
    bool bad_local = false;
    MatrixXd lXtRX(P_total,P_total); lXtRX.setZero();
    VectorXd lXtRy(P_total); lXtRy.setZero();
    MatrixXd lCdiag(P_total,P_total); lCdiag.setZero();
    VectorXd lc_vec(P_total); lc_vec.setZero();
    MatrixXd lCross(P_total,P_total); lCross.setZero();
    #pragma omp for schedule(dynamic, 1)
    for (int ig=0;ig<(int)groups.size();++ig) {
      const auto& g=groups[ig];
      {
        // OBS_SPACE DISABLED: single-response ("long") models now fall through to the
        // general Schur-complement path below (identical to the multi-response path).
        // The old obs_space branch factorized V = diag(inv_resid_var) + Z*R_cov*Z^T
        // using the random-effect COVARIANCE. During optimization the heteroscedastic
        // residual variance can -> 0 (inv_resid_var -> 0), making V ~ Z*R_cov*Z^T
        // rank-deficient (n < q) so the Cholesky factorization fails with
        // "Could not factor information matrix". The general path instead factors
        // K = G_inv (precision) + Z' Rinv Z, which is a sum of a PD and a PSD
        // matrix and is always numerically stable. This is exactly the path the
        // mathematically-equivalent joint model (size==2, same n<q) takes successfully.
        bool obs_space = false;
        if (obs_space) {
          const auto& resp = *model.responses[0];
          const ResponseGroupData& rg = resp.groups[ig];
          if (!rg.observed) continue;
          resp.compute_group(g, rg, ig, rws[0]);
          const MatrixXd& X = rg.X;
          const VectorXd& y = rg.y;
          const int n = (int)y.size();
          const MatrixXd& RinvX = rws[0].RinvX;
          VectorXd inv_var(n);
          for (int i=0;i<n;++i) {
            bool found=false;
            for (int j=0;j<(int)X.cols();++j) {
              if (std::fabs(X(i,j)) > 1e-14) { inv_var(i) = RinvX(i,j)/X(i,j); found=true; break; }
            }
            if (!found) inv_var(i) = 1.0;
          }
          MatrixXd V = MatrixXd::Zero(n,n);
          for (int i=0;i<n;++i) V(i,i) = inv_var(i);
          V.noalias() += g.Z * G_global * g.Z.transpose();
          Eigen::LLT<MatrixXd> lltV(V);
          if (lltV.info() != Eigen::Success) { bad_local = true; continue; }
          MatrixXd Vinv = lltV.solve(MatrixXd::Identity(n,n));
          lc.logdetV += 2.0*lltV.matrixLLT().diagonal().array().log().sum();
          MatrixXd VinvX = Vinv * X;
          VectorXd Vinvy = Vinv * y;
          lc.quad += y.dot(Vinvy);
          lXtRX.noalias() += X.transpose() * VinvX;
          lXtRy.noalias() += X.transpose() * Vinvy;
          continue;
        }
      }
      MatrixXd sZtRZ(model.q,model.q); sZtRZ.setZero();
      for (int ir=0;ir<(int)model.responses.size();++ir) {
        const ResponseModel& resp=*model.responses[ir];
        const ResponseGroupData& rg=resp.groups[ig];
        if (!rg.observed) continue;
        resp.compute_group(g,rg,ig,rws[ir]);
        int Q=rg.random_effect_indices.size();
        MatrixXd Zr(g.Z.rows(),Q);
        for (int k=0;k<Q;++k) { int idx=rg.random_effect_indices[k]-1;
          if (idx>=0&&idx<g.Z.cols()) Zr.col(k)=g.Z.col(idx); }
        MatrixXd blk=Zr.transpose()*rws[ir].RinvZ;
        for (int i=0;i<Q;++i) { int gi=rg.random_effect_indices[i]-1;
          for (int j=0;j<=i;++j) { int gj=rg.random_effect_indices[j]-1;
            if (gi>=0&&gj>=0&&gi<model.q&&gj<model.q) {
              sZtRZ(gi,gj)+=blk(i,j); if(i!=j) sZtRZ(gj,gi)+=blk(i,j); } } } }
      MatrixXd K=G_inv_global+sZtRZ; Eigen::LLT<MatrixXd> llt(K);
      if (llt.info()!=Eigen::Success) { bad_local=true; continue; }
      double logdetK=2.0*llt.matrixLLT().diagonal().array().log().sum();
      MatrixXd Kinv=llt.solve(MatrixXd::Identity(model.q,model.q));

      int nor=0;
      for (int ir=0;ir<(int)model.responses.size();++ir) {
        const ResponseModel& resp=*model.responses[ir];
        const ResponseGroupData& rg=resp.groups[ig];
        if (!rg.observed) continue;
        ++nor;
        int Q=rg.random_effect_indices.size();
        MatrixXd Zr(g.Z.rows(),Q);
        for (int k=0;k<Q;++k) { int idx=rg.random_effect_indices[k]-1;
          if (idx>=0&&idx<g.Z.cols()) Zr.col(k)=g.Z.col(idx); }
        Mr[ir].noalias()=rg.X.transpose()*rws[ir].RinvZ;
        Br[ir].noalias()=Zr.transpose()*rws[ir].Rinvy;
        lc.logdetV+=rws[ir].logdetR;
        lc.quad+=rg.y.dot(rws[ir].Rinvy);
      }
      lc.logdetV+=logdetK;

      // Assemble Mt: each response contributes its Mr at the correct global rows
      MatrixXd Mt(model.q,P_total); Mt.setZero();
      VectorXd Bt_vec(model.q); Bt_vec.setZero();
      for (int ir=0;ir<(int)model.responses.size();++ir) {
        const ResponseModel& resp=*model.responses[ir];
        const ResponseGroupData& rg=resp.groups[ig];
        if (!rg.observed) continue;
        int Q=rg.random_effect_indices.size(), pr=resp.fixed_effect_dimension();
        for (int k=0;k<Q;++k) { int gi=rg.random_effect_indices[k]-1;
          if (gi>=0&&gi<model.q) Mt.row(gi).segment(resp.fixed_effect_offset(),pr)=Mr[ir].col(k).transpose(); }
        for (int k=0;k<Q;++k) { int gi=rg.random_effect_indices[k]-1;
          if (gi>=0&&gi<model.q) Bt_vec(gi)=Br[ir](k); } }
      MatrixXd gf=llt.solve(Mt); VectorXd zf=llt.solve(Bt_vec);
      MatrixXd Cg=Mt.transpose()*gf; VectorXd cg=Mt.transpose()*zf;

      // Accumulate diagonal blocks of X^T R^{-1} X
      for (int ir=0;ir<(int)model.responses.size();++ir) {
        const ResponseModel& resp=*model.responses[ir];
        const ResponseGroupData& rg=resp.groups[ig];
        if (!rg.observed) continue;
        int pr=resp.fixed_effect_dimension(), off=resp.fixed_effect_offset();
        lXtRX.block(off,off,pr,pr).noalias()+=rg.X.transpose()*rws[ir].RinvX;
        lXtRy.segment(off,pr).noalias()+=rg.X.transpose()*rws[ir].Rinvy;
      }
      // Accumulate diagonal Woodbury correction
      for (int ir=0;ir<(int)model.responses.size();++ir) {
        const ResponseModel& resp=*model.responses[ir];
        int pr=resp.fixed_effect_dimension(), off=resp.fixed_effect_offset();
        lCdiag.block(off,off,pr,pr).noalias()+=Cg.block(off,off,pr,pr);
      }
      lc_vec.noalias()+=cg;

      // Cross-response Woodbury corrections
      // For conditionally independent responses, X_a^T R^{-1} X_b = 0
      // So off-diagonal block = -M_a^T K^{-1} M_b
      int off_a=0;
      for (int ia=0;ia<(int)model.responses.size();++ia) {
        const ResponseModel& resa=*model.responses[ia];
        const ResponseGroupData& rga=resa.groups[ig];
        if (!rga.observed) continue;
        int pra=resa.fixed_effect_dimension();
        MatrixXd Mta=MatrixXd::Zero(model.q,pra);
        for (int k=0;k<(int)rga.random_effect_indices.size();++k) {
          int gi=rga.random_effect_indices[k]-1;
          if(gi<0||gi>=model.q) continue;
          Mta.row(gi).segment(0,pra)=Mr[ia].col(k).transpose(); }
        for (int ib=ia+1;ib<(int)model.responses.size();++ib) {
          const ResponseModel& resb=*model.responses[ib];
          const ResponseGroupData& rgb=resb.groups[ig];
          if (!rgb.observed) continue;
          int prb=resb.fixed_effect_dimension();
          MatrixXd Mtb=MatrixXd::Zero(model.q,prb);
          for (int k=0;k<(int)rgb.random_effect_indices.size();++k) {
            int gi=rgb.random_effect_indices[k]-1;
            if(gi<0||gi>=model.q) continue;
            Mtb.row(gi).segment(0,prb)=Mr[ib].col(k).transpose(); }
          MatrixXd cross=Mta.transpose()*Kinv*Mtb;
          lCross.block(off_a,off_a+pra,pra,prb).noalias()+=cross;
          lCross.block(off_a+pra,off_a,prb,pra).noalias()+=cross.transpose();
        }
        off_a+=pra;
      }

      // Bug 3: Full quadratic form
      if (nor>1) {
        double bkb=0.0;
        for (int ir=0;ir<(int)model.responses.size();++ir) {
          if (!model.responses[ir]->groups[ig].observed) continue;
          const VectorXd& B1=Br[ir]; if(B1.size()==0) continue;
          for (int is=0;is<(int)model.responses.size();++is) {
            if (!model.responses[is]->groups[ig].observed) continue;
            const VectorXd& B2=Br[is]; if(B2.size()==0) continue;
            double t=0.0;
            for (int ki=0;ki<B1.size();++ki) { int gi=model.responses[ir]->groups[ig].random_effect_indices[ki]-1;
              if(gi<0||gi>=model.q) continue;
              for (int kj=0;kj<B2.size();++kj) { int gj=model.responses[is]->groups[ig].random_effect_indices[kj]-1;
                if(gj<0||gj>=model.q) continue;
                t+=B1(ki)*Kinv(gi,gj)*B2(kj); } }
            if(ir==is) bkb+=t; else if(ir<is) bkb+=2.0*t; } }
        lc.quad-=bkb;
      } else if (nor==1) {
        for (int ir=0;ir<(int)model.responses.size();++ir) {
          if (!model.responses[ir]->groups[ig].observed) continue;
          const VectorXd& B=Br[ir]; if(B.size()==0) continue;
          double t=0.0;
          for (int ki=0;ki<B.size();++ki) { int gi=model.responses[ir]->groups[ig].random_effect_indices[ki]-1;
            if(gi<0||gi>=model.q) continue;
            for (int kj=0;kj<B.size();++kj) { int gj=model.responses[ir]->groups[ig].random_effect_indices[kj]-1;
              if(gj<0||gj>=model.q) continue;
              t+=B(ki)*Kinv(gi,gj)*B(kj); } }
          lc.quad-=t; break; } }
    }
    #pragma omp critical
    {
      contribution.logdetV += lc.logdetV;
      if (bad_local) bad = true;
      contribution.quad += lc.quad;
      // Diagonal blocks: X^T R^{-1} X - Cg(diag)
      for (size_t ir=0;ir<model.responses.size();++ir) {
        const auto& resp=*model.responses[ir];
        int pr=resp.fixed_effect_dimension(), off=resp.fixed_effect_offset();
        contribution.XtVinvX.block(off,off,pr,pr).noalias() += (lXtRX.block(off,off,pr,pr) - lCdiag.block(off,off,pr,pr));
      }
      // Off-diagonal blocks: -M_a^T K^{-1} M_b = -lCross
      for (size_t ia=0;ia<model.responses.size();++ia) {
        const auto& resa=*model.responses[ia];
        int pra=resa.fixed_effect_dimension(), offa=resa.fixed_effect_offset();
        for (size_t ib=ia+1;ib<model.responses.size();++ib) {
          const auto& resb=*model.responses[ib];
          int prb=resb.fixed_effect_dimension(), offb=resb.fixed_effect_offset();
          contribution.XtVinvX.block(offa,offb,pra,prb).noalias() += (-lCross.block(offa,offb,pra,prb));
          contribution.XtVinvX.block(offb,offa,prb,pra).noalias() += (-lCross.block(offb,offa,prb,pra));
        }
      }
      contribution.XtVinvy.noalias() += (lXtRy - lc_vec);
    }
  }
  return !bad;
}

// ============================================================
// MAIN
// ============================================================
// [[Rcpp::export]]
Rcpp::List reml_loglik_cpp_impl(Rcpp::XPtr<PreparedModel> model_xptr,
    const Eigen::VectorXd& theta_raw, bool return_penalty, bool return_timing) {
  PreparedModel& model=*model_xptr;
  auto t0=std::chrono::high_resolution_clock::now();
  const int qg=model.q, nc=model.n_corr, ro=qg+nc;
  if (model.responses.empty()) Rcpp::stop("Need at least one response.");
  int p=0, tp=0;
  for (const auto& rp:model.responses) { const ResponseModel& r=*rp;
    tp+=r.parameter_count(); p+=r.fixed_effect_dimension(); }
  if (theta_raw.size()!=ro+tp) Rcpp::stop("theta size mismatch.");

  VectorXd log_sds=theta_raw.head(qg);
  VectorXd cuc=theta_raw.segment(qg,nc);
  for (const auto& rp:model.responses) { ResponseModel& r=*rp;
    r.update_working_from_theta(theta_raw,qg,nc,rp->parameter_offset());
    if (!r.parameters_are_valid_for_theta(theta_raw,qg,nc,rp->parameter_offset())) return bad_result(); }
  VectorXd corrs=cuc.array().tanh().matrix();
  if ((corrs.array().abs()>0.9999999).any()) return bad_result();

  RandomEffectsSystem rs=build_random_effects_system(qg,log_sds,cuc);
  if (rs.G_inv.size()==0) return bad_result();
  double penalty=0.0;
  for (const auto& rp:model.responses) penalty+=compute_penalty(model,log_sds,*rp);
  penalty=0.0;
  auto t1=std::chrono::high_resolution_clock::now();
  double t_build=std::chrono::duration<double>(t1-t0).count();

  ConditionalContribution contribution(p);
  if (!accumulate_conditional_contributions(model,rs.G_inv,contribution)) return bad_result();
  auto t2=std::chrono::high_resolution_clock::now();
  double t_groups=std::chrono::duration<double>(t2-t1).count();
  double logdetV=contribution.logdetV+static_cast<double>(model.groups.size())*rs.logdetG;

  bool fe_ok;
  VectorXd beta=solve_fe(contribution.XtVinvX,contribution.XtVinvy,fe_ok);
  if (!fe_ok) return bad_result();
  double qr=std::max(0.0,contribution.quad-beta.transpose()*contribution.XtVinvy);

  // BUG 1 FIX: Sum observations across ALL responses
  int total_n=0;
  for (const auto& rp:model.responses)
    for (const auto& gr:rp->groups) total_n+=static_cast<int>(gr.y.size());

  bool det_ok;
  double logdetX=logdet_chol(contribution.XtVinvX,det_ok);
  if (!det_ok) return bad_result();
  double ll=-0.5*(logdetV+logdetX+qr+(total_n-p)*std::log(2.0*M_PI));
  auto t3=std::chrono::high_resolution_clock::now();
  double t_finalize=std::chrono::duration<double>(t3-t2).count();

  Rcpp::List result;
  result["logLik_unpenalized"]=ll; result["logLik_penalized"]=ll-penalty; result["penalty"]=penalty;
  if (return_timing&&return_penalty) result["timings"]=Rcpp::List::create(
    Rcpp::Named("build_G")=t_build,Rcpp::Named("groups")=t_groups,
    Rcpp::Named("finalize")=t_finalize,Rcpp::Named("total")=t_build+t_groups+t_finalize);
  return result;
}

// [[Rcpp::export]]
VectorXd compute_beta_hat_cpp(const Eigen::VectorXd& theta, Rcpp::XPtr<PreparedModel> model) {
  PreparedModel& p=*model;
  if (p.responses.empty()) Rcpp::stop("Need at least one response.");
  const int ro=p.q+p.n_corr; int p_=0,tp=0;
  for (const auto& rp:p.responses) { const ResponseModel& r=*rp;
    tp+=r.parameter_count(); p_+=r.fixed_effect_dimension(); }
  if (theta.size()!=ro+tp) Rcpp::stop("theta size mismatch.");
  for (const auto& rp:p.responses) { ResponseModel& r=*rp;
    r.update_working_from_theta(theta,p.q,p.n_corr,rp->parameter_offset()); }
  MatrixXd L=build_cholesky(theta,p.q), G=L*L.transpose();
  G.diagonal().array()+=1e-8; Eigen::LLT<MatrixXd> lg(G);
  if (lg.info()!=Eigen::Success) Rcpp::stop("Could not factor G.");
  MatrixXd Gi=lg.solve(MatrixXd::Identity(p.q,p.q));
  ConditionalContribution contribution(p_);
  if (!accumulate_conditional_contributions(p,Gi,contribution))
    Rcpp::stop("Could not accumulate.");
  bool ok;
  VectorXd bh=solve_fe(contribution.XtVinvX,contribution.XtVinvy,ok);
  (void)ok;
  return bh;
}
