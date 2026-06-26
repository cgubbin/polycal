//! Fluent builder for calibration problems.
//!
//! A [`Problem`] describes a calibration dataset together with its uncertainty
//! model, fitting constraints and model-selection strategy.
//!
//! Problems are constructed using [`ProblemBuilder`], which validates the input
//! before producing a [`Problem`].
//!
//! # Examples
//!
//! An ordinary least-squares calibration:
//!
//! ```
//! # use polycal::{Problem, ScoringStrategy};
//! # let x = [0.0, 1.0, 2.0];
//! # let y = [1.0, 2.0, 3.0];
//! let problem = Problem::builder()
//!     .with_observations(x, y)
//!     .infer_domain()
//!     .score_by(ScoringStrategy::Aicc)
//!     .build()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! A weighted calibration:
//!
//! ```
//! # use polycal::{Problem, ScoringStrategy};
//! # let x = [0.0, 1.0, 2.0];
//! # let y = [1.0, 2.0, 3.0];
//! # let uy = [0.05, 0.05, 0.10];
//! let problem = Problem::builder()
//!     .with_observations(x, y)
//!     .with_y_uncertainty(uy)
//!     .infer_domain()
//!     .build()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Once constructed, a [`Problem`] may be solved for one or more polynomial
//! degrees using the calibration routines provided elsewhere in the crate.

use crate::{Constraint, Problem, ScoringStrategy, Uncertainty};

use ndarray::{Array1, Array2};
use std::ops::Range;

/// Builder for calibration problems.
///
/// The builder collects the calibration observations together with the
/// associated uncertainty model and fitting options before validating the
/// complete problem.
#[derive(Debug)]
pub struct ProblemBuilder<E> {
    x: Option<Array1<E>>,
    y: Option<Array1<E>>,
    uncertainty: Uncertainty<E>,
    domain: Option<Range<E>>,
    strategy: ScoringStrategy,
    constraint: Option<Constraint<E>>,
}

impl<E> ProblemBuilder<E> {
    /// Create a new builder
    pub fn new() -> Self {
        todo!()
    }

    /// Supply the calibration data.
    pub fn with_data(self, x: impl Into<Array1<E>>, y: impl Into<Array1<E>>) -> Self;

    /// Treat the observations as having negligible uncertainty.
    pub fn with_no_uncertainty(self) -> Self;

    /// Supply independent uncertainties on the dependent variable.
    pub fn with_y_uncertainty(self, sigma_y: impl Into<Array1<E>>) -> Self;

    /// Supply a covariance matrix for the dependent variable.
    pub fn with_y_covariance(self, covariance: impl Into<Array2<E>>) -> Self;

    /// Supply independent uncertainties on both variables.
    pub fn with_xy_uncertainty(
        self,
        sigma_x: impl Into<Array1<E>>,
        sigma_y: impl Into<Array1<E>>,
    ) -> Self;

    /// Supply covariance matrices for both variables.
    pub fn with_xy_covariance(
        self,
        covariance_x: impl Into<Array2<E>>,
        covariance_y: impl Into<Array2<E>>,
    ) -> Self;

    /// Specify the physical calibration domain.
    pub fn on_domain(self, domain: Range<E>) -> Self;

    /// Infer the physical domain from the supplied observations.
    pub fn infer_domain(self) -> Self;

    /// Specify the model-selection strategy.
    pub fn score_by(self, strategy: ScoringStrategy) -> Self;

    /// Apply a calibration constraint.
    pub fn constrain(self, constraint: Constraint<E>) -> Self;

    /// Validate the supplied information and construct the problem.
    pub fn build(self) -> Result<Problem<E>, ProblemError<E>>;
}
