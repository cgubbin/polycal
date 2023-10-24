# Polycal
---
Methods for determining, verifying and using polynomial calibration curves. The methods used conform as closely as possible to ISO/TS 28038.

## Usage

To use the crate we first build a `Problem`, using known calibration data. We then then solve for the best fit solution:
```rust
use polycal::ProblemBuilder;

let problem = ProblemBuilder::new(x_data, y_data)
    .with_independent_uncertainty(y_uncertainty)
    .build();

let best_fit = problem.solve()?;
```

We can either reconstruct unknown response from known stimulus values:
```rust
use polycal::Unsure;

let known_stimulus = Unsure { estimate: 1.0, standard_uncertainty: 0.01 };
let estimated_response = best_fit.response(known_stimulus);
```
or calculate unknown stimulus from a known response
```rust
use polycal::Unsure;

let known_response = Unsure { estimate: 1.0, standard_uncertainty: 0.01 };
let estimated_stimulus = best_fit.stimulus(known_response);
```
