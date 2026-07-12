# MixedModels TODO

Renumbered on 2026-07-11 into implementation order.  From now on, keep item
numbers stable: mark items complete rather than deleting them, and append new
work within the most suitable section.

## Established response boundary

1. [x] `PreparedModel` owns a vector of response objects.
2. [x] Each response owns response-specific per-group data and its `working`
   state.
3. [x] Normal response construction and R-list extraction live in the response
   implementation; `init_model_cpp()` selects the current normal response type.
4. [x] Homoscedastic and heteroscedastic normal responses use the same
   response-based group accumulator.

## Next: normal response separation and regression protection

5. [ ] Split `response.cpp`/`response.hpp` into a small shared response base,
   separate `HomoNormalResponse` and `HeteroNormalResponse` headers and
   implementations, and a response-factory implementation.
6. [ ] Restore an independent scalar-variance homo-normal fast path.  It must
   not allocate or update per-observation `eta`/`inv_resid_var` vectors merely
   to match the heteroscedastic interface.
7. [ ] Keep the homo- and heteroscedastic-normal group calculations separate
   enough that either remains a useful regression check for the other.
8. [ ] Add repeatable small-data regression tests for homo- and
   heteroscedastic-normal likelihoods, beta estimates, parameter-count
   validation, and the homo-normal fast path.
9. [ ] Add a construction test that verifies response data and working buffers
   are owned by the stored response object.

## Response specification and construction

10. [ ] Replace the `prepared["heteroscedastic"]` selection in
    `init_model_cpp()` with a formal R response specification.  Initially
    default to `homo_normal`; retain the current `var_pred` shortcut that
    selects `hetero_normal` when no explicit type is supplied.
11. [ ] Move response-type parsing and construction to a response factory so
    `init_model_cpp()` creates a response from a specification without an
    expanding response-type `if`/`switch` block.
12. [ ] Define an R-side response specification object: type, outcome/fixed
    effect inputs, random-effects mapping, and response-specific options.
    Keep R-to-C++ conversion at the response factory/constructor boundary.
13. [ ] Move response parameter initial values, bounds, names, validation, and
    penalties out of `MixedModel.R` and into a response-aware specification/API.
    The current C++ `residual_variance_penalty_weight` is an interim hook based
    on induced log residual variances, so equivalent homo/hetero models receive
    the same penalty.

## Conditional contributions and Gaussian random-effects kernel

14. [ ] Introduce a response-independent *conditional contribution* contract.
    For each response and group it represents information conditional on the
    group's random effects; it must be possible to combine contributions from
    independent responses before integrating over shared random effects.
15. [ ] Add a response-independent merge operation for conditional
    contributions from all outcomes in one group, then pass the merged result
    to the random-effects integration kernel.  Do not marginalise each
    response separately when they share random effects.
16. [ ] Split `GroupWorkspace` into a normal/Gaussian workspace and a
    random-effects integration workspace.  Rename the existing structure to
    make its `R^-1`, log-determinant, Cholesky, and `V^-1` assumptions explicit.
17. [ ] Extract the current exact normal-response random-effects calculation
    into a `GaussianRandomEffectsKernel`, used by both normal responses but
    not imposed as the general response interface.
18. [ ] Add tests for merging conditional contributions from two independent
    responses sharing random effects, before enabling multi-response fitting.

## Sparse EHR-oriented multi-response models

19. [ ] Define a response-to-global-random-effect map.  Each response uses a
    local random-effects design `Z_r` and an ordered vector of indexes into the
    model-wide random-effects vector `b`; shared indexes represent the same
    random effect across responses.
20. [ ] Implement the normal contribution merge using those maps: scatter each
    response's local `Z_r' R_r^-1 Z_r`, `Z_r' R_r^-1 X_r`, and
    `Z_r' R_r^-1 y_r` contributions into global-random-effect coordinates
    before factorising the shared random-effects system.
21. [ ] Support response-specific random-slope and interaction columns.  The R
    interface must construct each response's local `Z_r` from its own random
    effects formula/design and map each local column to a global random-effect
    index; no assumption that fixed-effect `X_r` or random-effect `Z_r` terms
    are identical across responses.
22. [ ] Support groups with no observations for a response.  Such a
    response/group pair contributes nothing to the conditional likelihood;
    its random effects remain informed by observed responses through shared
    effects and correlations in the global random-effects covariance matrix.
23. [ ] Make response-group data sparse: store only observed response/group
    pairs and iterate those pairs during contribution construction, while the
    random-effects kernel still iterates the model-wide groups.
24. [ ] Replace the single-response guards in `reml_loglik_cpp_impl()` and
    `compute_beta_hat_cpp()` after the merged-contribution contract is tested.
25. [ ] Replace response-0 assumptions for `p`, observation count, and
    fixed-effect accumulation with an explicit multi-response parameter layout.
26. [ ] Add tests with two responses, partially overlapping groups, a shared
    random intercept, response-specific random slopes, and correlated but
    distinct random effects.  Verify that a missing response/group pair adds
    no likelihood contribution and that observed responses still update the
    posterior information for its correlated random effects.
27. [ ] Keep correlated multivariate residuals out of `ResponseCollection`.
    Independent outcomes may share random effects; a correlated multivariate
    response is a distinct response model.
28. [ ] Document the missing-data assumption for absent EHR outcomes: absence
    is conditionally ignorable given the fitted joint model.  If the
    observation process is informative, model it explicitly as an additional
    response rather than treating correlation alone as a missingness model.

## General response likelihood and optimisation

29. [ ] Define a minimal response interface that supports an exact conditional
    log-likelihood evaluation without requiring gradients or Hessians.  New
    response types should be implementable and verifiable through this path
    first.
30. [ ] Add optional score/gradient and Hessian interfaces.  Optimisers may use
    them when provided, but responses must not be required to implement them.
31. [ ] Define the integration contract for non-normal random-effects models:
    exact integration where available, Laplace approximation, adaptive
    quadrature, or another response/kernel-specific method.
32. [ ] Add a Laplace integration kernel that consumes an exact conditional
    likelihood plus optional derivatives; define a derivative-free fallback
    for initial response implementations.
33. [ ] Design EM support as a separate optimisation strategy over a common
    model/likelihood interface, rather than embedding EM-specific state in
    response classes.
34. [ ] Separate likelihood evaluation from optimisation selection so REML,
    full likelihood, gradient/Hessian-based methods, EM, Laplace, and future
    optimisers can be selected without changing response implementations.
35. [ ] Specify the fully parametric survival-response contract, including
    baseline-hazard parameterisation, censoring/truncation inputs, and its
    conditional likelihood.  Semiparametric Cox PH is explicitly out of scope
    until this interface is stable.
36. [ ] For every new response type, first compare its exact conditional
    log-likelihood against a simple/reference implementation before enabling
    derivative, Laplace, or EM paths.

## Coupled and joint responses

37. [ ] Distinguish independent responses sharing random effects from coupled
    responses whose conditional likelihood depends on another response's
    latent predictor, random effects, or event process.  Do not represent the
    latter by silently inserting fitted values as ordinary fixed covariates.
38. [ ] Design an explicit `JointModel`/compound-response contract for a
    limited initial set of coupled models.  A leading candidate is a normal
    longitudinal marker jointly modelled with a parametric survival outcome
    whose hazard depends on the marker's latent trajectory.
39. [ ] Represent response dependencies as an explicit, validated graph or
    compound model specification.  Define how shared latent random effects,
    response-specific parameters, and response-to-response links are
    evaluated; reject unsupported cycles rather than producing ambiguous
    fitted-value feedback.
40. [ ] Keep prediction uncertainty and feedback inside the joint likelihood.
    If a fitted/predicted response is used by another outcome, integrate or
    approximate over its latent uncertainty rather than treating a plug-in
    prediction as known data.
41. [ ] Route coupled non-normal models away from the exact Gaussian REML
    accumulator to an appropriate joint-likelihood/integration strategy (for
    example Laplace or quadrature).  Retain simple conditional likelihood
    evaluation as the first implementation and verification step.
42. [ ] Consider an explicit observation-process response for informative
    outcome measurement or dropout.  State its missing-data assumptions and
    test it separately from outcome-response correlation.
