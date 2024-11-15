[![codecov](https://codecov.io/gh/sensoriumtl/polycal/graph/badge.svg?token=6FRPE3DXWO)](https://codecov.io/gh/sensoriumtl/polycal)

# Polycal
---
Methods for determining, verifying and using polynomial calibration curves. The methods used conform as closely as possible to ISO/TS 28038.

## Usage

To use the crate we first build a `Problem`, using known calibration data. We then then solve for the best fit solution:
```rust
use ndarray::Array1;
use polycal::ProblemBuilder;

a = 1.;
b = 2.;
stimulus: Array1<f64> = Array1::range(0., 10., 0.5);
num_data_points = stimulus.len();
response: Array1<f64> = stimulus
    .iter()
    .map(|x| a + b * x)
    .collect();
let dependent_uncertainty: Array1<f64> = response
    .iter()
    .map(|x| x / 1000.0)
    .collect();

let problem = ProblemBuilder::new(stimulus.view(), response.view())
    .unwrap()
    .with_dependent_variance(dependent_uncertainty.view())
    .unwrap()
    .build();

let maximum_degree = 5;

let best_fit = problem.solve(maximum_degree).unwrap();

for (expected, actual) in response.into_iter().zip(stimulus.into_iter().map(|x|
    best_fit.certain_response(x).unwrap())).skip(1).take(num_data_points-2) {
        assert!((expected - actual).abs() < 1e-5);;
}
```

We can either reconstruct unknown response from known stimulus values:
```rust
use polycal::{AbsUncertainty, Uncertainty};

let known_stimulus = AbsUncertainty::new(1.0, 0.01);
let estimated_response = best_fit.response(known_stimulus);
```
or calculate unknown stimulus from a known response
```rust
use polycal::{AbsUncertainty, Uncertainty};

let known_stimulus = AbsUncertainty::new(1.0, 0.01);
let initial_guess = None;
let max_iter = Some(100);
let estimated_stimulus = best_fit.stimulus(
    known_response,
    initial_guess,
    max_iter
);
let estimated_stimulus = best_fit.stimulus(known_response);
```
