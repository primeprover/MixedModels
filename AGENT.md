# AGENT.md - MixedModels Project Notes

## Current Status
**DONE.** All bugs fixed and the C++ joint + single-response models are correct.
- **Multi-response joint model** matches the authoritative nlme REML reference exactly
  (joint = **-15958.07** == nlme reference).
- **Single-response "long" model** now matches the joint model: the root-cause bug
  (Bug 5, below) is fixed.
- **Threading race** (issue #2) is fixed while keeping multithreading: the accumulation
  is race-free (per-thread private accumulators + a serialized `#pragma omp critical`
  merge); 1- and 16-thread long fits agree to ~8.6e-09 (FP rounding).

## Bugs Fixed

### In `reml_loglik.cpp`
**Bug 1 — Incorrect observation count in REML normalization**
- **Location**: `reml_loglik_cpp_impl()`
- **Original**: Used only the first response's observation count
  (`total_observation_count(model.responses.front()->groups)`).
- **Fix**: Sum observation counts across **ALL** responses.
- **Impact**: REML constant `-0.5*(n-p)*log(2π)` now uses the correct total N.

**Bug 2 — Double-counting of logdetK in multi-response accumulation**
- **Location**: `accumulate_conditional_contributions()`
- **Original**: `logdetK` was added per-response via per-response `add_group_contribution()` calls.
- **Fix**: `logdetK` added **exactly once per group**, outside the per-response loop
  (`lc.logdetV += logdetK`).
- **Impact**: each response now contributes only its `logdetR`, not `logdetK`.

**Bug 3 — Dropped cross-response coupling in quadratic forms**
- **Location**: `accumulate_conditional_contributions()`
- **Original**: quadratic-form correction only used per-response diagonal Woodbury terms
  `B_r' K⁻¹_rr B_r`.
- **Fix**: full correction `B' K⁻¹ B = Σ_{r,s} B_r' K⁻¹_{rs} B_s` using the full `K⁻¹`.
- **Impact**: correctly accounts for off-diagonal cross-response correlations.

**Bug 4 — VinvX/Vinvy Woodbury correction used the wrong K⁻¹ sub-matrix**
- **Location**: `accumulate_conditional_contributions()` second pass
- **Original**: for each response r, embedded `M_r` (Q_resp × p_resp) into q-space with
  zeros elsewhere, solved `K γ = embedded M_r`, extracted Q_resp rows → computed
  `K⁻¹_{sub,:} M_r` (WRONG: only Q_resp rows of the full K⁻¹), dropping cross-response
  coupling when `Q_resp < q`.
- **Fix**:
  1. Assemble `M_total` (q × P_total) and `B_total` (q × 1) from **all** responses.
  2. Solve `K γ = M_total` and `K ζ = B_total` **once** using the cached Cholesky.
  3. For each response, extract its Q_resp × p_resp block of γ and Q_resp block of ζ.
  4. `VinvX = RinvX - RinvZ × gamma_block`, `Vinvy = Rinvy - RinvZ × zeta_block`.
- **Impact**: captures cross-response coupling through the **full** K⁻¹.

### In `MixedModel.R`
**Bug 5 — Missing `random_effect_indices` in the heteroscedastic (long) path**
  **(ROOT CAUSE of the long-model mismatch)**
- **Location**: `prepare_mixed_data()` heteroscedastic branch
- **Original**: hetero groups were built **without** `random_effect_indices`, so the C++
  `extract_hetero_groups()` left `ResponseGroupData::random_effect_indices` empty (`Q=0`).
  With `Q=0`, `accumulate_conditional_contributions()` never added the `Z'R^{-1}Z`
  Schur-complement term to `K`, i.e. the random effects were **entirely dropped** from the
  precision matrix.
- **Fix**: add `random_effect_indices = as.integer(random_map)` to the hetero group list
  (also correct for the multi-response path via remapping).
- **Impact**: the long model's `K = G^{-1} + Z'R^{-1}Z` now matches the joint model's,
  so the two are mathematically equivalent. Pre-fix long = -17951.89 vs joint -15958.07
  (a gap of ~2000 log-likelihood units, from dropped random effects).

### Threading race (issue #2)
- **Symptom**: `set_threads_cpp(n>1)` → "Error: Could not factor information matrix."
  (`accumulate_conditional_contributions` accumulated `XtVinvX`; multi-threaded the
  result could be clobbered / near-singular.)
- **Fix**: multithreading restored (per user — serial-only was unsuitable) and made
  race-free in `accumulate_conditional_contributions`: per-thread private accumulators
  (lc, lXtRX, lXtRy, lCdiag, lc_vec, lCross) declared **inside** the `#pragma omp parallel`
  region (private by scope, no `private()` clause), a `#pragma omp for schedule(dynamic,1)`
  over groups, and a single `#pragma omp critical` that merges each thread's `lc` into the
  shared `contribution`. `ConditionalContribution` (response_base.hpp) got a default ctor so
  the per-thread private copy is valid. 1- vs 16-thread long fits agree to ~8.6e-09 (FP rounding).

### Long-model ("longitudinal") factorization failure — root cause
- **Symptom**: single-response hetero ("long") model threw "Could not factor information matrix."
  during optimization (the information/Schur-complement matrix would not factorize).
- **Root cause**: the long model took a special `obs_space` branch (`responses.size()==1 && n < q`)
  in `accumulate_conditional_contributions` that factorized `V = diag(inv_resid_var) + Z R Z^T`
  using the random-effect **covariance** R. During the optimizer the heteroscedastic residual
  variance can go to 0, so `inv_resid_var` ~ 0 and `V ~ Z R Z^T`, which is rank-deficient
  (rank <= n < q) and fails Cholesky. The 2-response joint model skips this branch and instead
  factorizes the precision-based `K = G^{-1} + Z' Rinv Z`, which is always positive-definite.
- **Fix**: disable the branch (`bool obs_space = false`), so the long model uses the same robust
  Schur-complement path as the joint model. After the Bug 5 fix (`random_effect_indices`
  present, `Q = 6` per group), that path is well-formed.
- **Impact**: the long model's information matrix now factors at every optimizer theta; it fits
  and converges to **-15958.07** == nlme reference, with 1 thread or 16 threads.

## Verification Status
- **Code compiles** cleanly.
- **Joint model** = **-15958.07** == nlme reference (diag_fix.R -> diag_final_log.txt).
- **Pre-fix joint** = -15944.5 (buggy, zero off-diagonal coupling).
- **Pre-fix long** = -17951.89 (random effects dropped).
- **Post-fix long** == joint == nlme reference = **-15958.07**, converged — **VERIFIED** (no
  factorization error, `mt_fit.R`; 1- vs 16-thread logLik differ ~8.6e-09 (race-free).
- Separate SBP = -8688.837; Separate HbA1c = -7292.565; Separate sum = -15981.4.
- Joint beats the separate-sum by ~23 (expected cross-response correlation gain).

## Key Files
- `MixedModel.R` - Main R interface (Bug 5 fix in `prepare_mixed_data` hetero branch)
- `reml_loglik.cpp` - C++ likelihood (Bugs 1-4 fixed; OpenMP race-free: per-thread accumulators + critical merge; obs_space branch disabled)
- `homo_normal_response.cpp` / `hetero_normal_response.cpp` - response definitions
- `response_base.hpp` - `ResponseGroupData` (has `random_effect_indices`), `ConditionalContribution`
- `mt_fit.R` - long model 1-thread vs 16-thread race check (VERIFIED, 8.6e-09 diff)
- `diag_fix.R` - comprehensive verification (same-theta checks)
- `truth_theta_23.rds` - simulation truth parameters
- `test_multi_resp_equivalence.R` - equivalence test

## Verification commands
```
C:\PROGRA~1\R\R-43~1.1\bin\Rscript.exe diag_fix.R
C:\PROGRA~1\R\R-43~1.1\bin\Rscript.exe verify_1thread.R
```
(Rscript not on PATH; use the full path, or run in a persistent terminal to avoid the 30 s
`run_command` timeout. The Rcpp DLL rebuild is slow on this machine (g++ `-g3` + Eigen);
R buffers stdout to a redirected log until the process exits.)
