//! polycal is a library for fitting, and using, polynomial calibration functions.
//!
//! It's goal is to offer a simple, application agnostic, interface allowing for easy integration
//! into code bases in many fields.
//!
//! The implementation follows the ISO/TS 28038 standard where possible.
//!
//! To use the library, we first create a [`Problem`] using some known calibration data. Calibration
//! data is composed of independent, or stimulus variables. These describe the inputs to the
//! measurement model. For each stimulus there is an associated response, or dependent variable. A
//! calibration problem can be constructed from known stimulus and response data. Currently the
//! inputs must be provided as [`ndarray::ArrayView1`].
//!
//! ```
//! use ndarray::Array1;
//! use polycal::ProblemBuilder;
//!
//! let a = 1.;
//! let b = 2.;
//! let stimulus: Array1<f64> = Array1::range(0., 10., 0.5);
//! let num_data_points = stimulus.len();
//! let response: Array1<f64> = stimulus
//!     .iter()
//!     .map(|x| a + b * x)
//!     .collect();
//!
//! let problem = ProblemBuilder::new(stimulus.view(), response.view())
//!     .build();
//!
//! let maximum_degree = 5;
//!
//! let best_fit = problem.solve(maximum_degree).unwrap();
//!
//! for (expected, actual) in response.into_iter().zip(stimulus.into_iter().map(|x|
//! best_fit.certain_response(x).unwrap())).skip(1).take(num_data_points-2) {
//!     assert!((expected - actual).abs() < 1e-5);;
//!     }
//! ```
//!
//! We can account for uncertainties in the *response* variables at present. These can be attached
//! to a [`ProblemBuilder`] as:
//!
//! ```
//! use ndarray::Array1;
//! use polycal::ProblemBuilder;
//!
//! let a = 1.;
//! let b = 2.;
//! let stimulus: Array1<f64> = Array1::range(0., 10., 0.5);
//! let num_data_points = stimulus.len();
//! let response: Array1<f64> = stimulus
//!     .iter()
//!     .map(|x| a + b * x)
//!     .collect();
//! let independent_uncertainty: Array1<f64> = response
//!     .iter()
//!     .map(|x| x / 1000.0)
//!     .collect();
//!
//! let problem = ProblemBuilder::new(stimulus.view(), response.view())
//!     .with_independent_uncertainty(independent_uncertainty.view())
//!     .build();
//! ```
//! Note that all methods with [`panic`] if the provided stimulus, response and uncertainties
//! contain unequal numbers of elements.
//!
//! ## Reconstruction
//!
//! Given a [`Fit`] we can reconstruct unknown response from known stimulus values. This uses the
//! calculated polynomial series directly.
//! ```ignore
//! use polycal::Unsure;
//!
//! let known_stimulus = Unsure { estimate: 1.0, standard_uncertainty: 0.01 };
//! let estimated_response = best_fit.response(known_stimulus);
//! ```
//! Alternatively we can calculate unknown stimulus values from known response values. This
//! numerically minimises the residual of the fit. An initial guess and maximum iteration count can
//! be provided.
//! ```ignore
//! use polycal::Unsure;
//!
//! let known_response = Unsure { estimate: 1.0, standard_uncertainty: 0.01 };
//! let initial_guess = None;
//! let max_iter = Some(100);
//! let estimated_stimulus = best_fit.stimulus(
//!     known_response,
//!     initial_guess,
//!     max_iter
//!     );
//! ```
//!
//! # TODO
//! - Currently polycal can only account for errors in the independent variables. In future a total
//!     least squares algorithm should be implemented to allow use of errors in both independent and
//!     dependent variables.
//!
//!
//!
#![allow(dead_code)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

mod builder;
mod calculate;
mod chebyshev;
mod error;
mod problem;
mod solvers;
mod utils;

extern crate blas_src;

pub type PolyCalResult<T, E> = ::std::result::Result<T, PolyCalError<E>>;

pub use builder::{ProblemBuilder, Set, Unset};
pub use calculate::Fit;
pub use chebyshev::{ChebyshevBuilder, PolynomialSeries, Series};
pub use error::PolyCalError;
pub use problem::{Constraint, Problem, ScoringStrategy};

pub use argmin::core::ArgminFloat;

pub use argmin_math::{
    ArgminAdd, ArgminConj, ArgminDot, ArgminL2Norm, ArgminMul, ArgminSub, ArgminZeroLike,
};
