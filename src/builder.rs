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
//! # let x = vec![0.0, 1.0, 2.0];
//! # let y = vec![1.0, 2.0, 3.0];
//! let problem = Problem::builder()
//!     .with_data(x, y)
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
//! # let x = vec![0.0, 1.0, 2.0];
//! # let y = vec![1.0, 2.0, 3.0];
//! # let uy = vec![0.05, 0.05, 0.10];
//! let problem = Problem::builder()
//!     .with_data(x, y)
//!     .with_y_uncertainty(uy)
//!     .infer_domain()
//!     .build()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Once constructed, a [`Problem`] may be solved for one or more polynomial
//! degrees using the calibration routines provided elsewhere in the crate.

use crate::{
    Problem, fit::Constraint, problem::GoodnessOfFit, problem::Uncertainty, score::ScoringStrategy,
};

use confi::ConfidenceLevel;
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
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
    infer_domain: bool,
    domain: Option<Range<E>>,
    strategy: ScoringStrategy,
    constraint: Option<Constraint<E>>,
    goodness_of_fit: GoodnessOfFit<E>,
}

impl<E: Float + FromPrimitive> Default for ProblemBuilder<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E> ProblemBuilder<E> {
    /// Create a new problem builder.
    #[must_use]
    pub fn new() -> Self
    where
        E: Float + FromPrimitive,
    {
        Self {
            x: None,
            y: None,
            uncertainty: Uncertainty::None,
            domain: None,
            infer_domain: false,
            strategy: ScoringStrategy::Aicc,
            constraint: None,
            goodness_of_fit: GoodnessOfFit::ChiSquare {
                confidence: ConfidenceLevel::new(
                    E::from_f64(0.95).expect("0.95 should be representable"),
                )
                .expect("0.95 confidence should be representable"),
            },
        }
    }

    /// Supply calibration data.
    ///
    /// `x` contains the physical independent/reference values and `y` containts the physical
    /// dependent/response values
    #[must_use]
    pub fn with_data(mut self, x: impl Into<Array1<E>>, y: impl Into<Array1<E>>) -> Self {
        self.x = Some(x.into());
        self.y = Some(y.into());
        self
    }

    /// Supply calibration data.
    ///
    /// `x` contains the physical independent/reference values and `y` containts the physical
    /// dependent/response values
    #[must_use]
    pub fn with_observations(mut self, x: impl Into<Array1<E>>, y: impl Into<Array1<E>>) -> Self {
        self.with_data(x, y)
    }

    /// Treat the observations as having negligible uncertainty.
    #[must_use]
    pub fn with_no_uncertainty(mut self) -> Self {
        self.uncertainty = Uncertainty::None;
        self
    }

    /// Supply independent standard uncertainties on the dependent variable.
    #[must_use]
    pub fn with_y_uncertainty(mut self, sigma_y: impl Into<Array1<E>>) -> Self {
        self.uncertainty = Uncertainty::YDiagonal { uy: sigma_y.into() };
        self
    }

    /// Supply a covariance matrix for the dependent variable.
    #[must_use]
    pub fn with_y_covariance(mut self, covariance: impl Into<Array2<E>>) -> Self {
        self.uncertainty = Uncertainty::YCovariance {
            vy: covariance.into(),
        };
        self
    }

    /// Supply independent uncertainties on both variables.
    #[must_use]
    pub fn with_xy_uncertainty(
        mut self,
        sigma_x: impl Into<Array1<E>>,
        sigma_y: impl Into<Array1<E>>,
    ) -> Self {
        self.uncertainty = Uncertainty::XYDiagonal {
            ux: sigma_x.into(),
            uy: sigma_y.into(),
        };
        self
    }

    /// Supply covariance matrices for both variables.
    #[must_use]
    pub fn with_xy_covariance(
        mut self,
        covariance_x: impl Into<Array2<E>>,
        covariance_y: impl Into<Array2<E>>,
    ) -> Self {
        self.uncertainty = Uncertainty::XYCovariance {
            vx: covariance_x.into(),
            vy: covariance_y.into(),
        };
        self
    }

    /// Specify the physical calibration domain.
    #[must_use]
    pub fn on_domain(mut self, domain: Range<E>) -> Self {
        self.domain = Some(domain);
        self.infer_domain = false;
        self
    }

    /// Infer the physical domain from the supplied observations.
    #[must_use]
    pub fn infer_domain(mut self) -> Self {
        self.domain = None;
        self.infer_domain = true;
        self
    }

    /// Specify the model-selection strategy.
    #[must_use]
    pub fn score_by(mut self, strategy: ScoringStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Apply a calibration constraint.
    #[must_use]
    pub fn constrain(mut self, constraint: Constraint<E>) -> Self {
        self.constraint = Some(constraint);
        self
    }

    /// Disable chi-square validation of candidate fits
    #[must_use]
    pub fn without_goodness_of_fit(mut self) -> Self {
        self.goodness_of_fit = GoodnessOfFit::Disabled;
        self
    }

    /// Require chi-square validation of candidate fits
    #[must_use]
    pub fn require_goodness_of_fit(mut self, confidence: ConfidenceLevel<E>) -> Self {
        self.goodness_of_fit = GoodnessOfFit::ChiSquare { confidence };
        self
    }

    /// Validate the supplied information and construct the problem.
    pub fn build(self) -> Result<Problem<E>, ProblemBuilderError>
    where
        E: Float,
    {
        let x = self.x.ok_or(ProblemBuilderError::MissingData)?;
        let y = self.y.ok_or(ProblemBuilderError::MissingData)?;

        validate_data(&x, &y)?;

        let domain = if self.infer_domain {
            infer_domain_from_x(&x)?
        } else {
            self.domain.ok_or(ProblemBuilderError::MissingDomain)?
        };

        validate_domain(&domain)?;
        validate_uncertainty(&self.uncertainty, x.len())?;

        let response_domain = infer_domain_from_x(&y)?;
        validate_domain(&response_domain)?;

        Ok(Problem {
            x,
            y,
            uncertainty: self.uncertainty,
            domain,
            response_domain,
            strategy: self.strategy,
            constraint: self.constraint,
            goodness_of_fit: self.goodness_of_fit,
        })
    }
}

impl<E: Float + FromPrimitive> Problem<E> {
    /// Begin constructing a calibration problem
    #[must_use]
    pub fn builder() -> ProblemBuilder<E> {
        ProblemBuilder::new()
    }
}

/// Errors returned while constructing a [`Problem`]
#[derive(Debug, thiserror::Error)]
pub enum ProblemBuilderError {
    /// No calibration data were supplied
    #[error("calibration data must be supplied")]
    MissingData,

    /// No domain was supplied, and domain inference was not requested
    #[error("calibration domain must be supplied or inferred")]
    MissingDomain,

    /// Calibration arrays were empty
    #[error("calibration data must contain at least one point")]
    EmptyData,

    /// The independent and dependent arrays have different lengths.
    #[error("x and y data must have equal length")]
    LengthMismatch,

    /// Data contained NaN or infinite values.
    #[error("calibration data must be finite")]
    InvalidData,

    /// Domain was non-finite or had non-positive width.
    #[error("domain must be finite and have positive width")]
    InvalidDomain,

    /// Uncertainty array length did not match the data length.
    #[error("uncertainty vector length must match data length")]
    InvalidUncertaintyLength,

    /// Covariance matrix shape did not match the data length.
    #[error("covariance matrix must be square with dimension equal to data length")]
    InvalidCovarianceShape,

    /// Uncertainties must be finite and non-negative.
    #[error("uncertainties must be finite and non-negative")]
    InvalidUncertaintyValues,
}

fn validate_data<E>(x: &Array1<E>, y: &Array1<E>) -> Result<(), ProblemBuilderError>
where
    E: Float,
{
    if x.is_empty() {
        return Err(ProblemBuilderError::EmptyData);
    }

    if x.len() != y.len() {
        return Err(ProblemBuilderError::LengthMismatch);
    }

    if x.iter().any(|value| !value.is_finite()) || y.iter().any(|value| !value.is_finite()) {
        return Err(ProblemBuilderError::InvalidData);
    }

    Ok(())
}

fn infer_domain_from_x<E>(x: &Array1<E>) -> Result<Range<E>, ProblemBuilderError>
where
    E: Float,
{
    if x.is_empty() {
        return Err(ProblemBuilderError::EmptyData);
    }

    let mut lower = x[0];
    let mut upper = x[0];

    for &value in x.iter().skip(1) {
        lower = lower.min(value);
        upper = upper.max(value);
    }

    let domain = lower..upper;
    validate_domain(&domain)?;

    Ok(domain)
}

fn validate_domain<E>(domain: &Range<E>) -> Result<(), ProblemBuilderError>
where
    E: Float,
{
    if domain.start.is_finite() && domain.end.is_finite() && domain.start < domain.end {
        Ok(())
    } else {
        Err(ProblemBuilderError::InvalidDomain)
    }
}

fn validate_uncertainty<E>(
    uncertainty: &Uncertainty<E>,
    n: usize,
) -> Result<(), ProblemBuilderError>
where
    E: Float,
{
    match uncertainty {
        Uncertainty::None => Ok(()),

        Uncertainty::YDiagonal { uy } => validate_uncertainty_vector(uy, n),

        Uncertainty::YCovariance { vy } => validate_covariance_matrix(vy, n),

        Uncertainty::XYDiagonal { ux, uy } => {
            validate_uncertainty_vector(ux, n)?;
            validate_uncertainty_vector(uy, n)
        }

        Uncertainty::XYCovariance { vx, vy } => {
            validate_covariance_matrix(vx, n)?;
            validate_covariance_matrix(vy, n)
        }
    }
}

fn validate_uncertainty_vector<E>(values: &Array1<E>, n: usize) -> Result<(), ProblemBuilderError>
where
    E: Float,
{
    if values.len() != n {
        return Err(ProblemBuilderError::InvalidUncertaintyLength);
    }

    if values
        .iter()
        .any(|value| !value.is_finite() || *value < E::zero())
    {
        return Err(ProblemBuilderError::InvalidUncertaintyValues);
    }

    Ok(())
}

fn validate_covariance_matrix<E>(
    covariance: &Array2<E>,
    n: usize,
) -> Result<(), ProblemBuilderError>
where
    E: Float,
{
    if covariance.nrows() != n || covariance.ncols() != n {
        return Err(ProblemBuilderError::InvalidCovarianceShape);
    }

    if covariance.iter().any(|value| !value.is_finite()) {
        return Err(ProblemBuilderError::InvalidUncertaintyValues);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    fn build_basic_problem() -> Problem<f64> {
        Problem::builder()
            .with_data(vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0])
            .infer_domain()
            .build()
            .unwrap()
    }

    #[test]
    fn new_builder_has_expected_defaults() {
        let builder = ProblemBuilder::<f64>::new();

        assert!(builder.x.is_none());
        assert!(builder.y.is_none());
        assert!(matches!(builder.uncertainty, Uncertainty::None));
        assert!(builder.domain.is_none());
        assert!(!builder.infer_domain);
        assert!(builder.constraint.is_none());
        assert!(matches!(builder.strategy, ScoringStrategy::Aicc));
    }

    #[test]
    fn problem_builder_default_matches_new() {
        let builder = ProblemBuilder::<f64>::default();

        assert!(builder.x.is_none());
        assert!(builder.y.is_none());
        assert!(matches!(builder.uncertainty, Uncertainty::None));
        assert!(builder.domain.is_none());
        assert!(!builder.infer_domain);
        assert!(builder.constraint.is_none());
        assert!(matches!(builder.strategy, ScoringStrategy::Aicc));
    }

    #[test]
    fn problem_builder_entrypoint_returns_empty_builder() {
        let builder = Problem::<f64>::builder();

        assert!(builder.x.is_none());
        assert!(builder.y.is_none());
        assert!(matches!(builder.uncertainty, Uncertainty::None));
    }

    #[test]
    fn with_data_stores_x_and_y() {
        let builder = ProblemBuilder::new().with_data(vec![0.0, 1.0], vec![2.0, 3.0]);

        assert_eq!(builder.x.unwrap(), arr1(&[0.0, 1.0]));
        assert_eq!(builder.y.unwrap(), arr1(&[2.0, 3.0]));
    }

    #[test]
    fn with_observations_is_alias_for_with_data() {
        let builder = ProblemBuilder::new().with_observations(vec![0.0, 1.0], vec![2.0, 3.0]);

        assert_eq!(builder.x.unwrap(), arr1(&[0.0, 1.0]));
        assert_eq!(builder.y.unwrap(), arr1(&[2.0, 3.0]));
    }

    #[test]
    fn build_rejects_missing_data() {
        let err = ProblemBuilder::<f64>::new()
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::MissingData));
    }

    #[test]
    fn build_rejects_missing_domain() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::MissingDomain));
    }

    #[test]
    fn build_rejects_empty_data() {
        let err = ProblemBuilder::<f64>::new()
            .with_data(vec![], vec![])
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::EmptyData));
    }

    #[test]
    fn build_rejects_length_mismatch() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0])
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::LengthMismatch));
    }

    #[test]
    fn build_rejects_nan_x() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, f64::NAN], vec![1.0, 2.0])
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidData));
    }

    #[test]
    fn build_rejects_nan_y() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, f64::NAN])
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidData));
    }

    #[test]
    fn build_rejects_infinite_x() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, f64::INFINITY], vec![1.0, 2.0])
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidData));
    }

    #[test]
    fn build_rejects_infinite_y() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, f64::NEG_INFINITY])
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidData));
    }

    #[test]
    fn build_infers_domain_from_x_values() {
        let problem = ProblemBuilder::new()
            .with_data(vec![10.0, 5.0, 20.0], vec![1.0, 2.0, 3.0])
            .infer_domain()
            .build()
            .unwrap();

        assert_eq!(problem.domain, 5.0..20.0);
    }

    #[test]
    fn build_rejects_inferred_zero_width_domain() {
        let err = ProblemBuilder::new()
            .with_data(vec![1.0, 1.0], vec![2.0, 3.0])
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidDomain));
    }

    #[test]
    fn build_uses_explicit_domain() {
        let problem = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .on_domain(-10.0..10.0)
            .build()
            .unwrap();

        assert_eq!(problem.domain, -10.0..10.0);
    }

    #[test]
    fn explicit_domain_disables_domain_inference() {
        let problem = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .infer_domain()
            .on_domain(-10.0..10.0)
            .build()
            .unwrap();

        assert_eq!(problem.domain, -10.0..10.0);
    }

    #[test]
    fn infer_domain_overrides_previous_explicit_domain() {
        let problem = ProblemBuilder::new()
            .with_data(vec![0.0, 5.0, 10.0], vec![1.0, 2.0, 3.0])
            .on_domain(-10.0..10.0)
            .infer_domain()
            .build()
            .unwrap();

        assert_eq!(problem.domain, 0.0..10.0);
    }

    #[test]
    fn build_rejects_reversed_explicit_domain() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .on_domain(1.0..0.0)
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidDomain));
    }

    #[test]
    fn build_rejects_zero_width_explicit_domain() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .on_domain(1.0..1.0)
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidDomain));
    }

    #[test]
    fn build_rejects_non_finite_explicit_domain() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .on_domain(f64::NEG_INFINITY..1.0)
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidDomain));
    }

    #[test]
    fn with_no_uncertainty_sets_uncertainty_to_none() {
        let problem = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_y_uncertainty(vec![0.1, 0.2])
            .with_no_uncertainty()
            .infer_domain()
            .build()
            .unwrap();

        assert!(matches!(problem.uncertainty, Uncertainty::None));
    }

    #[test]
    fn build_accepts_y_uncertainty() {
        let problem = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_y_uncertainty(vec![0.1, 0.2])
            .infer_domain()
            .build()
            .unwrap();

        match problem.uncertainty {
            Uncertainty::YDiagonal { uy } => assert_eq!(uy, arr1(&[0.1, 0.2])),
            _ => panic!("expected YDiagonal uncertainty"),
        }
    }

    #[test]
    fn build_rejects_bad_y_uncertainty_length() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_y_uncertainty(vec![0.1])
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidUncertaintyLength));
    }

    #[test]
    fn build_rejects_negative_y_uncertainty() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_y_uncertainty(vec![0.1, -0.2])
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidUncertaintyValues));
    }

    #[test]
    fn build_rejects_nan_y_uncertainty() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_y_uncertainty(vec![0.1, f64::NAN])
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidUncertaintyValues));
    }

    #[test]
    fn build_accepts_y_covariance() {
        let covariance = arr2(&[[1.0, 0.1], [0.1, 2.0]]);

        let problem = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_y_covariance(covariance.clone())
            .infer_domain()
            .build()
            .unwrap();

        match problem.uncertainty {
            Uncertainty::YCovariance { vy } => assert_eq!(vy, covariance),
            _ => panic!("expected YCovariance uncertainty"),
        }
    }

    #[test]
    fn build_rejects_bad_y_covariance_shape() {
        let covariance = Array2::zeros((2, 3));

        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_y_covariance(covariance)
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidCovarianceShape));
    }

    #[test]
    fn build_rejects_non_finite_y_covariance() {
        let covariance = arr2(&[[1.0, f64::NAN], [0.1, 2.0]]);

        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_y_covariance(covariance)
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidUncertaintyValues));
    }

    #[test]
    fn build_accepts_xy_uncertainty() {
        let problem = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_xy_uncertainty(vec![0.01, 0.02], vec![0.1, 0.2])
            .infer_domain()
            .build()
            .unwrap();

        match problem.uncertainty {
            Uncertainty::XYDiagonal { ux, uy } => {
                assert_eq!(ux, arr1(&[0.01, 0.02]));
                assert_eq!(uy, arr1(&[0.1, 0.2]));
            }
            _ => panic!("expected XYDiagonal uncertainty"),
        }
    }

    #[test]
    fn build_rejects_bad_xy_uncertainty_length() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_xy_uncertainty(vec![0.01], vec![0.1, 0.2])
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidUncertaintyLength));
    }

    #[test]
    fn build_rejects_negative_x_uncertainty() {
        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_xy_uncertainty(vec![-0.01, 0.02], vec![0.1, 0.2])
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidUncertaintyValues));
    }

    #[test]
    fn build_accepts_xy_covariance() {
        let vx = arr2(&[[1.0, 0.1], [0.1, 2.0]]);
        let vy = arr2(&[[3.0, 0.2], [0.2, 4.0]]);

        let problem = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_xy_covariance(vx.clone(), vy.clone())
            .infer_domain()
            .build()
            .unwrap();

        match problem.uncertainty {
            Uncertainty::XYCovariance { vx: px, vy: py } => {
                assert_eq!(px, vx);
                assert_eq!(py, vy);
            }
            _ => panic!("expected XYCovariance uncertainty"),
        }
    }

    #[test]
    fn build_rejects_bad_xy_covariance_shape() {
        let vx = Array2::zeros((2, 2));
        let vy = Array2::zeros((2, 3));

        let err = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_xy_covariance(vx, vy)
            .infer_domain()
            .build()
            .unwrap_err();

        assert!(matches!(err, ProblemBuilderError::InvalidCovarianceShape));
    }

    #[test]
    fn score_by_sets_strategy() {
        let problem = ProblemBuilder::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .infer_domain()
            .score_by(ScoringStrategy::Bic)
            .build()
            .unwrap();

        assert!(matches!(problem.strategy, ScoringStrategy::Bic));
    }

    #[test]
    fn build_basic_problem_works() {
        let problem = build_basic_problem();

        assert_eq!(problem.x, arr1(&[0.0, 1.0, 2.0]));
        assert_eq!(problem.y, arr1(&[1.0, 2.0, 3.0]));
        assert_eq!(problem.domain, 0.0..2.0);
        assert!(matches!(problem.uncertainty, Uncertainty::None));
        assert!(problem.constraint.is_none());
    }
}
