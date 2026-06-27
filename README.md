# polycal

## Polycal

Polynomial calibration curves with uncertainty-aware model selection.

`polycal` provides a small API for constructing calibration problems,
fitting polynomial calibration curves, validating that those curves are
suitable for calibration use, and evaluating the resulting calibration
functions in both directions.

The crate is designed around calibration workflows where a set of known
stimulus values is paired with measured response values. It fits Chebyshev
polynomial calibration curves using the lower-level `poly_series` crate.

### Core workflow

A calibration workflow usually has four steps:

1. Build a [`Problem`] from calibration data.
2. Choose an uncertainty model.
3. Fit one degree, or scan several polynomial degrees.
4. Evaluate the fitted curve from stimulus to response, or invert it from
   response to stimulus.

```rust
use polycal::{Problem, ScoringStrategy};

let stimulus: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let response = vec![1.0, 3.0, 5.0, 7.0, 9.0];

let fit = Problem::builder()
    .with_data(stimulus, response)
    .infer_domain()
    .score_by(ScoringStrategy::Aicc)
    .without_goodness_of_fit()
    .build()?
    .solve_degree(1)?;

let y = fit.response(2.5)?;
let x = fit.stimulus(6.0)?;

assert!((y - 6.0).abs() < 1e-10);
assert!((x - 2.5).abs() < 1e-10);
```

### Calibration problems

A [`Problem`] stores:

- stimulus values,
- response values,
- the calibration domain,
- an uncertainty model,
- an optional calibration constraint,
- a goodness-of-fit policy, and
- a scoring strategy for model selection.

Problems are constructed with [`Problem::builder`]. The builder validates
array lengths, finite values, uncertainty dimensions and calibration domain
before producing a problem.

### Uncertainty models

`polycal` supports several uncertainty descriptions through [`Uncertainty`]:

- no supplied uncertainty,
- independent response standard uncertainties,
- a full response covariance matrix,
- independent stimulus and response standard uncertainties,
- full stimulus and response covariance matrices.

In this release, fitting with uncertainty in the response variable is
implemented. Fitting with uncertainty in the stimulus variable is recognised
by the API but not yet implemented; those paths return an unsupported-method
error.

### Constraints

Calibration curves may be constrained using [`Constraint`].

A constrained model is written as:

```
f(x) = a(x) + m(x) q(x)
```

where `q(x)` is the free fitted polynomial, `a(x)` is an additive component,
and `m(x)` is a multiplicative component.

This can encode common calibration requirements such as forcing the curve to
pass through the origin.

### Model selection

[`Problem::solve_up_to_degree`] fits candidate polynomial degrees and ranks
accepted candidates using a [`ScoringStrategy`].

Candidate curves are rejected if they are not monotonic, because a calibration
curve must provide a unique inverse mapping from response to stimulus.

Optional goodness-of-fit validation can also reject candidates whose χ²
statistic lies outside the configured confidence interval.

### Evaluation

A fitted [`Fit`] can be evaluated in both directions:

- [`Fit::response`] maps stimulus to response.
- [`Fit::stimulus`] maps response to stimulus.

Fallible evaluation is used so that out-of-domain values, non-monotonic
curves and ambiguous inverse solutions are reported as errors rather than
silent numerical failures.

Uncertainty propagation is available through:

- [`Fit::response_with_uncertainty`],
- [`Fit::stimulus_estimate_first_order`].

These methods propagate supplied input uncertainty and fitted coefficient
covariance using local linearisation.

### Feature flags

Root finding, fitting and generalized least-squares calculations require a
BLAS/LAPACK backend through. The recommented pattern is to
enable one backend feature on this crate.

Example:

```toml
[dependencies]
polycal = { version = "...", features = ["openblas-system"] }
```

### Scope of this release

This release focuses on polynomial calibration with uncertainty in the
response variable. Errors-in-variables fitting, total least squares and
Monte Carlo uncertainty propagation are planned extension points rather than
part of the stable v0.2 interface.

[`Constraint`]: crate::fit::Constraint
[`Fit`]: crate::fit::Fit
[`Fit::response`]: crate::fit::Fit::response
[`Fit::response_with_uncertainty`]: crate::fit::Fit::response_with_uncertainty
[`Fit::stimulus`]: crate::fit::Fit::stimulus
[`Fit::stimulus_estimate_first_order`]: crate::fit::Fit::stimulus_estimate_first_order
[`Problem`]: crate::Problem
[`Problem::builder`]: crate::Problem::builder
[`Problem::solve_up_to_degree`]: crate::Problem::solve_up_to_degree
[`ScoringStrategy`]: crate::ScoringStrategy
[`Uncertainty`]: crate::problem::Uncertainty

License: MIT
