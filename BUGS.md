# BUGS.md: Multi-Response Log-Likelihood Discrepancies

This document details three critical bugs in the C++ multi-response log-likelihood computation (`reml_loglik_cpp_impl` and `accumulate_conditional_contributions`). These bugs cause the joint (wide) multi-response model to compute an incorrect log-likelihood compared to the mathematically equivalent single-response (long) model. 

An agentic AI tasked with fixing these must ensure that the joint model's likelihood is invariant to the data layout (wide vs. long) and correctly implements the full multivariate normal likelihood with shared random effects.

---

## Bug 1: Incorrect Observation Count (`n`) in REML Normalization

### Context
`reml_loglik_cpp_impl` (Final log-likelihood assembly)

### Description
The REML log-likelihood includes a normalization constant: `-0.5 * (n - p) * log(2 * M_PI)`. 
Currently, the code calculates `n` using only the observations from the **first** response:
```cpp
const int n = total_observation_count(model.responses.front()->groups);
```
For a joint model with $R$ responses, this sets $n = N$ (the number of patients/rows for response 1), rather than the true total number of scalar observations $n = R \times N$.

### Mathematical Impact
The joint model's log-likelihood is artificially shifted by `-0.5 * (R - 1) * N * log(2 * M_PI)`. For a 2-response model with 2,500 patients, this introduces a massive, constant error of roughly `+2,297` to the log-likelihood.

### Required Fix
1. Locate the calculation of `n` in `reml_loglik_cpp_impl`.
2. Change it to sum the observation counts across **all** responses in the model, not just `responses.front()`.
3. Ensure `p` (total fixed effects) remains the sum of fixed effects across all responses (which is likely already correct, but verify).

---

## Bug 2: Double-Counting of `logdetK` in Multi-Response Accumulation

### Context
`accumulate_conditional_contributions` -> `add_group_contribution` (or the loop iterating over `model.responses`).

### Description
In a joint model, the random effects are shared across responses. Therefore, the $q \times q$ precision matrix $K = G^{-1} + Z'R^{-1}Z$ is computed **once per group** and is shared across all responses. 
However, inside the per-response loop, the code adds the log-determinant of $K$ to the total `logdetV` for *every* response:
```cpp
// Inside the loop over model.responses:
logdetV += response_workspace.logdetR + integration_workspace.logdetK; 
```
For a 2-response model, `logdetK` is added twice per group.

### Mathematical Impact
The determinant of the marginal covariance $V_g$ for a group should be $\det(R_g) \det(G) \det(K_g)$. By adding `logdetK` twice, the code computes $\log|V_g| = \log|R_g| + 2\log|K_g| + \log|G|$. This artificially inflates the complexity penalty in the REML likelihood, biasing the optimizer to shrink random-effect variances and correlations toward zero.

### Required Fix
1. Ensure `integration_workspace.logdetK` is added to the group's `logdetV` exactly **once** per group.
2. This can be achieved by either:
   * Moving the `logdetK` addition outside the per-response loop (adding it at the group level).
   * Adding a condition inside the loop so it is only added on the first response iteration (e.g., `if (ir == 0)`).

---

## Bug 3: Dropped Cross-Response Coupling in Quadratic Forms

### Context
`accumulate_conditional_contributions` -> Computation of `XtVinvX`, `XtVinvy`, and `quad` ($y'V^{-1}y$).

### Description
The true marginal covariance $V = R + ZGZ'$ is **not block-diagonal** across responses because the off-diagonal blocks of $G$ (cross-response correlations) couple the responses. Consequently, $V^{-1}$ is not block-diagonal.

Currently, the code computes the Woodbury terms (`VinvX`, `Vinvy`) independently per response using only that response's block of $Z$. It then computes `infoblock = Xr' VinvXr` and places it in the diagonal block of `XtVinvX`. 
This implicitly assumes $V^{-1}$ is block-diagonal, completely dropping the off-diagonal cross-response terms.

### Mathematical Impact
Using the Woodbury identity, the true information matrix is:
$$X'V^{-1}X = X'R^{-1}X - (X'R^{-1}Z) K^{-1} (Z'R^{-1}X)$$
Let $M = X'R^{-1}Z$. Because $R$ and $X$ are block-diagonal across responses, $M$ is block-diagonal. However, $K^{-1}$ is a **full** $q \times q$ matrix. 
The current code only computes the diagonal blocks of $M K^{-1} M'$. It drops the off-diagonal blocks (e.g., $M_1 K^{-1}_{12} M_2'$), which represent the cross-response coupling. This makes the joint model's likelihood mathematically inconsistent (it uses the full $K$ for the determinant, but a block-diagonal approximation for the quadratic forms).

### Required Fix
This is the most complex fix. The agent must compute the full quadratic forms using the full $q \times q$ $K^{-1}$ without explicitly forming the massive $N_{total} \times N_{total}$ $V^{-1}$ matrix.

**Step-by-step implementation guide:**
1. **Compute block-diagonal components per response:** For each response $r$, compute $M_r = X_r' R_r^{-1} Z_r$ and $B_r = Z_r' R_r^{-1} y_r$. (The code likely already does this internally).
2. **Assemble the full block-diagonal matrices:** Construct the full $p \times q$ matrix $M$ and the full $q \times 1$ vector $B$ by placing $M_r$ and $B_r$ into their respective global index offsets. (All off-diagonal blocks of $M$ and $B$ between different responses are strictly zero).
3. **Compute the full Woodbury corrections:** 
   * For `XtVinvX`: Compute the full correction matrix $C = M K^{-1} M^T$. Subtract $C$ from the block-diagonal $X'R^{-1}X$ to get the full `XtVinvX`.
   * For `XtVinvy`: Compute the full correction vector $c = M K^{-1} B$. Subtract $c$ from the block-diagonal $X'R^{-1}y$ to get the full `XtVinvy`.
   * For `quad` ($y'V^{-1}y$): Compute the scalar correction $q_{corr} = B^T K^{-1} B$. Subtract $q_{corr}$ from the total $y'R^{-1}y$ to get the full `quad`.
4. **Crucial Constraint:** Do not loop over responses to compute the final `XtVinvX`, `XtVinvy`, and `quad`. These final assemblies **must** happen after the per-response $M_r$ and $B_r$ are gathered, using the full $q \times q$ $K^{-1}$ matrix to capture the cross-response correlations. Do not construct the full $N_{total} \times N_{total}$ $V^{-1}$ matrix; use the block-diagonal structure of $M$ and the full $q \times q$ $K^{-1}$ to compute the corrections efficiently.

---

## Testing Instructions for the AI

After implementing the fixes, verify correctness by:
1. Fitting both the joint (wide) model and the long (single-response) model to the same data
2. Evaluating both log-likelihoods at identical parameter values (theta)
3. Confirming that `logLik(joint_model)` exactly matches `logLik(long_model)` within numerical precision
4. Ensuring the optimizer converges to the same parameter estimates for both formulations

The fixes must preserve computational efficiency—do not create large temporary matrices or break the memory-efficient per-group processing pattern.
There is a test script `test_multi_resp_equivalence.R` that can be used to validate the fixes. It compares the log-likelihoods of the joint and long models and prints the difference, which should be near zero after the bugs are fixed. R should be able to compile and run the test script. No need to look for a c++ compiler; the Rcpp package handles compilation automatically.
