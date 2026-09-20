# Mixed Model C++ Engine: Architecture & Computational Flow

## 1. Parameter Layout (`theta`)
Optimization operates on a single unconstrained vector with the following layout:
- `[0, q_global)`          : Log-SDs of global random effects (length $Q$)
- `[q, q + n_corr)`       : Unconstrained correlations (length $Q(Q-1)/2$, mapped via $\tanh()$)
- `[offset, end)`         : Response-specific residual variance parameters
  - Homoscedastic: 1 parameter per response ($\log \sigma_r^2$)
  - Heteroscedastic: $p_{W}$ parameters per response (log-variance design coefficients)

## 2. Data Structures & Separation of Concerns
| Component | Contents | Theta-Dependent? | Updated When? |
|-----------|----------|------------------|---------------|
| `PreparedModel::groups` | Global $Z$, $n$, $Z^T Z$ per group | ❌ No | Once at init |
| `ResponseModel::groups` | Response-specific $X, y$, indices, observed flag | ❌ No | Once at init |
| `ResponseModel::working_*` | $R^{-1}$ terms, $\eta$, scaled matrices | ✅ Yes | Every likelihood call via `update_working_from_theta()` |
| `ConditionalContribution` | Accumulators for $X^T V^{-1} X$, $X^T V^{-1} y$, $\log|V|$, quadratic form | ❌ No | Reset per eval, merged across threads |

## 3. Computational Flow (Per Likelihood Evaluation)
1. **Parse theta**: Extract global log-SDs, correlations, response parameters.
2. **Build G**: Construct $G = L L^T$ from Cholesky, compute $G^{-1}$ and $\log|G|$.
3. **Update Working State**: Each response computes its current $R_r^{-1}$ scaling factors.
4. **Group Loop (OpenMP)**:
   - Per response: Compute scaled matrices $R_r^{-1}Z_r$, $R_r^{-1}X_r$, $R_r^{-1}y_r$.
   - Scatter $\sum Z_r^T R_r^{-1} Z_r$ into global $Q \times Q$ space.
   - Form $K = G^{-1} + Z^T R^{-1} Z$, compute LLT factorization.
   - Woodbury update: Solve for $V^{-1}X$, $V^{-1}y$, accumulate determinants & quadratic forms.
5. **Finalize**: 
   - Add $\log|G|$ to determinant sum (once per group).
   - Solve fixed effects: $\hat{\beta} = (X^T V^{-1} X)^{-1} X^T V^{-1} y$.
   - Compute quadratic residual & REML log-likelihood.
   - Apply boundary penalties on SDs & residual variances.

## 4. Indexing Conventions
- `q_global` ($Q$): Full random-effects dimension across all responses (e.g., 6)
- `q_resp` ($q_r$): Columns belonging to one response (e.g., 3)
- R passes 1-based indices in `random_effect_indices`; C++ subtracts 1 before column extraction.
- Fixed-effect offsets (`fixed_effect_offset_`) prevent cross-response $X^T V^{-1} X$ leakage.

## 5. Key Mathematical Identities Used
- **Determinant Decomposition**: $\log|V_g| = \sum_r \log|R_{rg}| + \log|K_g| + \log|G|$
- **Woodbury Matrix Identity**: $V^{-1} = R^{-1} - R^{-1} Z (G^{-1} + Z^T R^{-1} Z)^{-1} Z^T R^{-1}$
- **REML Log-Likelihood**: Marginalizes fixed effects via $(n-p)\log(2\pi)$ correction and quadratic residual deflation.
