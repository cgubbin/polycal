//! Fitted calibration curves.
//!
//! This module defines the calibration-level result type [`Fit`].
//!
//! A [`Fit`] is not just the raw polynomial returned by the underlying
//! polynomial-series fitting routine. It represents the final calibration
//! curve, including any calibration constraint that was applied during fitting.
//!
//! Internally, constrained fits are represented as
//!
//! ```text
//! y = a(x) + m(x) q(x)
//! ```
//!
//! where:
//!
//! - `a(x)` is the additive constraint,
//! - `m(x)` is the multiplicative constraint,
//! - `q(x)` is the fitted free polynomial.
//!
//! The stored [`Fit::calibration_curve`] is the fully composed curve
//! `a(x) + m(x) q(x)`, while [`Fit::free_polynomial`] returns `q(x)`.
//!
//! Most users should evaluate the calibration curve, not the free polynomial

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use poly_series::{ChebyshevSeries, PolynomialSeries};
use std::ops::Range;

/// Constraint applied to a fitted calibration curve.
///
/// A constraint transforms a free polynomial `q(x)` into the final calibration
/// curve
///
/// ```text
/// f(x) = a(x) + m(x) q(x)
/// ```
///
/// where `a(x)` is the additive component and `m(x)` is the multiplicative
/// component.
///
/// This representation can enforce structural properties of the calibration
/// curve. For example, choosing `a(x) = 0` and `m(x) = x` forces the final
/// curve to pass through the origin regardless of the fitted free polynomial.
///
/// The covariance stored on [`Fit`] refers to the coefficients of the free
/// polynomial, not necessarily to the coefficients of the fully composed curve.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Constraint<E> {
    /// Additive component `a(x)` of the constraint.
    pub(crate) additive: ChebyshevSeries<E>,

    /// Multiplicative component `m(x)` of the constraint.
    pub(crate) multiplicative: ChebyshevSeries<E>,
}

impl<E> Constraint<E>
where
    E: Float + FromPrimitive,
{
    /// Construct a constraint from explicit additive and multiplicative components.
    #[must_use]
    pub fn new(additive: ChebyshevSeries<E>, multiplicative: ChebyshevSeries<E>) -> Self {
        Self {
            additive,
            multiplicative,
        }
    }

    /// Construct a constraint that leaves the fitted polynomial unchanged.
    ///
    /// This gives
    ///
    /// ```text
    /// f(x) = q(x)
    /// ```
    #[must_use]
    pub fn identity(domain: Range<E>) -> Self {
        Self {
            additive: ChebyshevSeries::new(vec![E::zero()], domain.clone()).unwrap(),
            multiplicative: ChebyshevSeries::new(vec![E::one()], domain).unwrap(),
        }
    }

    /// Construct a constraint forcing the final calibration curve through the origin.
    ///
    /// The final model is
    ///
    /// ```text
    /// f(x) = x q(x)
    /// ```
    ///
    /// so `f(0) = 0` regardless of the fitted free polynomial `q`.
    ///
    /// The supplied domain must include the physical coordinate `0`.
    #[must_use]
    pub fn passing_through_origin(domain: Range<E>) -> Self {
        let zero = E::zero();

        // On an arbitrary physical domain, x is represented as a Chebyshev
        // series in scaled coordinate t:
        //
        // x(t) = centre + half_width * t
        let two = E::one() + E::one();
        let centre = (domain.start + domain.end) / two;
        let half_width = (domain.end - domain.start) / two;

        Self {
            additive: ChebyshevSeries::new(vec![zero], domain.clone()).unwrap(),
            multiplicative: ChebyshevSeries::new(vec![centre, half_width], domain).unwrap(),
        }
    }

    /// Construct a constraint forcing the final curve through `(x0, y0)`.
    ///
    /// The final model is
    ///
    /// ```text
    /// f(x) = y0 + (x - x0) q(x)
    /// ```
    ///
    /// so `f(x0) = y0` for any free polynomial `q`.
    #[must_use]
    pub fn passing_through(x0: E, y0: E, domain: Range<E>) -> Self {
        let two = E::one() + E::one();
        let centre = (domain.start + domain.end) / two;
        let half_width = (domain.end - domain.start) / two;

        // x - x0 = (centre - x0) + half_width * T1(t)
        Self {
            additive: ChebyshevSeries::new(vec![y0], domain.clone()).unwrap(),
            multiplicative: ChebyshevSeries::new(vec![centre - x0, half_width], domain).unwrap(),
        }
    }
}

impl<E> Constraint<E> {
    /// Apply the constraint to a free polynomial.
    ///
    /// Returns the composed calibration curve
    ///
    /// ```text
    /// a(x) + m(x) q(x)
    /// ```
    ///
    /// where `q(x)` is `series`.
    #[must_use]
    pub fn apply(&self, series: &ChebyshevSeries<E>) -> ChebyshevSeries<E>
    where
        E: Float + FromPrimitive,
    {
        self.additive.clone() + self.multiplicative.clone() * series.clone()
    }
}

/// Result of fitting a calibration curve.
///
/// `Fit` is the main result type returned by calibration solving. It stores both
/// the final calibration curve and the free polynomial used to construct it.
///
/// For unconstrained fits, [`Fit::calibration_curve`] and
/// [`Fit::free_polynomial`] represent the same polynomial.
///
/// For constrained fits, [`Fit::free_polynomial`] is the fitted polynomial
/// `q(x)`, while [`Fit::calibration_curve`] is the composed curve
/// `a(x) + m(x) q(x)`.
#[derive(Clone, Debug)]
pub struct Fit<E> {
    /// Fully composed calibration curve.
    pub(super) curve: ChebyshevSeries<E>,

    /// Raw fitted free polynomial.
    pub(super) free_polynomial: ChebyshevSeries<E>,

    /// Covariance matrix for the free polynomial coefficients, if available.
    pub(super) covariance: Option<Array2<E>>,

    /// Fitted response values at the original stimulus values.
    #[allow(dead_code)]
    pub(super) fitted_values: Array1<E>,

    /// Residuals at the original calibration points.
    ///
    /// Residuals are defined as
    ///
    /// ```text
    /// y_i - f(x_i)
    /// ```
    ///
    /// where `f` is the final calibration curve.
    #[allow(dead_code)]
    pub(super) residuals: Array1<E>,

    /// Constraint used to construct the final calibration curve.
    pub(super) constraint: Option<Constraint<E>>,

    /// Observed response range.
    pub(super) response_domain: Range<E>,

    /// Numerical fitting method used to obtain this fit.
    #[allow(dead_code)]
    pub(super) method: FitMethod,
}

impl<E> Fit<E> {
    /// Return the raw fitted free polynomial.
    ///
    /// For unconstrained fits this is the same as the calibration curve.
    /// For constrained fits this is the polynomial `q(x)` in
    /// `a(x) + m(x) q(x)`.
    #[must_use]
    pub fn free_polynomial(&self) -> &ChebyshevSeries<E> {
        &self.free_polynomial
    }

    /// Return the final calibration curve.
    ///
    /// This is the curve that should normally be used for calibration
    /// evaluation, monotonicity checks, inverse lookup and residual
    /// calculations.
    #[must_use]
    pub fn calibration_curve(&self) -> &ChebyshevSeries<E> {
        &self.curve
    }

    /// Evaluate the final calibration curve at stimulus `x`.
    ///
    /// This method does not check whether `x` lies inside the calibration
    /// domain. Use the fallible response/evaluation API if domain validation is
    /// required.
    pub fn evaluate(&self, x: E) -> E
    where
        E: Float + FromPrimitive,
    {
        self.curve.evaluate(x)
    }

    /// Return the fitted values at the original calibration points.
    #[must_use]
    pub fn fitted_values(&self) -> &Array1<E> {
        &self.fitted_values
    }

    /// Return residuals at the original calibration points.
    #[must_use]
    pub fn residuals(&self) -> &Array1<E> {
        &self.residuals
    }

    /// Return the coefficient covariance matrix, if available.
    ///
    /// The covariance matrix refers to the coefficients of
    /// [`Fit::free_polynomial`]. For constrained fits, uncertainty propagation
    /// through the constraint must account for the multiplicative component.
    #[must_use]
    pub fn covariance(&self) -> Option<&Array2<E>> {
        self.covariance.as_ref()
    }

    /// Return the observed response range.
    #[must_use]
    pub fn response_domain(&self) -> Range<E>
    where
        E: Clone,
    {
        self.response_domain.clone()
    }

    /// Return the fitting method used to construct the fit.
    #[must_use]
    pub fn method(&self) -> &FitMethod {
        &self.method
    }

    /// Return the constraint used by this fit, if any.
    #[must_use]
    pub fn constraint(&self) -> Option<&Constraint<E>> {
        self.constraint.as_ref()
    }
}

/// Numerical method used to construct a calibration fit.
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Debug, PartialEq)]
pub enum FitMethod {
    /// Ordinary least squares.
    ///
    /// Used when no measurement uncertainty is supplied.
    OrdinaryLeastSquares,

    /// Weighted least squares with independent response uncertainties.
    ///
    /// Used when each response observation has an independent standard
    /// uncertainty.
    WeightedLeastSquares,

    /// Generalized least squares with a full response covariance matrix.
    ///
    /// Used when the response observations have correlated uncertainties.
    GeneralizedLeastSquares,

    /// Total least squares / errors-in-variables fitting.
    ///
    /// Reserved for fits that account for uncertainty in both stimulus and
    /// response values.
    TotalLeastSquares,
}

impl<E> Fit<E> {
    /// Construct a calibration fit from a polynomial-series fit report.
    ///
    /// This adapter is used for ordinary, weighted and generalized least
    /// squares paths implemented through `poly_series`.
    ///
    /// The raw series in the report becomes the free polynomial. If a
    /// constraint is supplied, the final calibration curve is constructed by
    /// applying the constraint to that free polynomial.
    pub(crate) fn from_series_report(
        report: poly_series::FitReport<E, ChebyshevSeries<E>>,
        constraint: Option<Constraint<E>>,
        response_domain: Range<E>,
        method: FitMethod,
    ) -> Self
    where
        E: Float + FromPrimitive,
    {
        let curve = match &constraint {
            None => report.series.clone(),
            Some(c) => c.apply(&report.series),
        };

        Self {
            curve,
            free_polynomial: report.series,
            covariance: report.covariance.map(|values| {
                covariance_from_vecs(values)
                    .expect("this passes in testing so should not fail in practice")
            }),
            fitted_values: Array1::from_vec(report.fitted_values),
            residuals: Array1::from_vec(report.residuals),
            constraint,
            response_domain,
            method,
        }
    }
}

/// Convert a nested vector representation into an `ndarray` covariance matrix.
fn covariance_from_vecs<E>(values: Vec<Vec<E>>) -> Result<Array2<E>, ndarray::ShapeError> {
    let nrows = values.len();
    let ncols = values.first().map_or(0, Vec::len);

    let flat = values.into_iter().flatten().collect::<Vec<_>>();

    Array2::from_shape_vec((nrows, ncols), flat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use poly_series::{FitReport, PolynomialSeries};

    const EPS: f64 = 1.0e-12;

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= EPS,
            "expected {lhs} ≈ {rhs}, difference = {}",
            (lhs - rhs).abs()
        );
    }

    fn series(coefficients: Vec<f64>, domain: Range<f64>) -> ChebyshevSeries<f64> {
        ChebyshevSeries::new(coefficients, domain).unwrap()
    }

    fn report(series: ChebyshevSeries<f64>) -> FitReport<f64, ChebyshevSeries<f64>> {
        FitReport {
            series: series.clone(),
            coefficients: series.coefficients().to_vec(),
            covariance: None,
            fitted_values: vec![1.0, 2.0, 3.0],
            residuals: vec![0.1, -0.2, 0.3],
            degrees_of_freedom: 1,
            residual_sum_of_squares: 0.14,
            residual_variance: Some(0.14),
        }
    }

    #[test]
    fn covariance_from_vecs_constructs_matrix() {
        let covariance = covariance_from_vecs(vec![vec![1.0, 0.1], vec![0.1, 2.0]]).unwrap();

        assert_eq!(covariance, arr2(&[[1.0, 0.1], [0.1, 2.0]]));
    }

    #[test]
    fn covariance_from_vecs_rejects_ragged_rows() {
        let err = covariance_from_vecs(vec![vec![1.0, 0.1], vec![0.1]]);

        assert!(err.is_err());
    }

    #[test]
    fn constraint_apply_combines_additive_multiplicative_and_free_series() {
        let additive = series(vec![1.0], -1.0..1.0);
        let multiplicative = series(vec![0.0, 1.0], -1.0..1.0);
        let free = series(vec![2.0], -1.0..1.0);

        let constraint = Constraint {
            additive,
            multiplicative,
        };

        let curve = constraint.apply(&free);

        // curve(t) = 1 + t * 2
        assert_close(curve.evaluate(-1.0), -1.0);
        assert_close(curve.evaluate(0.0), 1.0);
        assert_close(curve.evaluate(1.0), 3.0);
    }

    #[test]
    fn constraint_apply_can_enforce_passing_through_origin_shape() {
        let additive = series(vec![0.0], -1.0..1.0);
        let multiplicative = series(vec![0.0, 1.0], -1.0..1.0);
        let free = series(vec![5.0, 2.0], -1.0..1.0);

        let constraint = Constraint {
            additive,
            multiplicative,
        };

        let curve = constraint.apply(&free);

        assert_close(curve.evaluate(0.0), 0.0);
    }

    #[test]
    fn from_series_report_without_constraint_uses_report_series_as_curve() {
        let free = series(vec![1.0, 2.0], -1.0..1.0);
        let report = report(free.clone());

        let fit = Fit::from_series_report(report, None, 0.0..10.0, FitMethod::OrdinaryLeastSquares);

        for x in [-1.0, 0.0, 1.0] {
            assert_close(fit.curve.evaluate(x), free.evaluate(x));
            assert_close(fit.free_polynomial.evaluate(x), free.evaluate(x));
        }

        assert!(fit.constraint.is_none());
        assert_eq!(fit.response_domain, 0.0..10.0);
        assert!(matches!(fit.method, FitMethod::OrdinaryLeastSquares));
    }

    #[test]
    fn from_series_report_with_constraint_uses_constrained_curve() {
        let free = series(vec![2.0], -1.0..1.0);

        let constraint = Constraint {
            additive: series(vec![1.0], -1.0..1.0),
            multiplicative: series(vec![0.0, 1.0], -1.0..1.0),
        };

        let fit = Fit::from_series_report(
            report(free.clone()),
            Some(constraint),
            0.0..10.0,
            FitMethod::WeightedLeastSquares,
        );

        // curve(t) = 1 + 2t
        assert_close(fit.curve.evaluate(-1.0), -1.0);
        assert_close(fit.curve.evaluate(0.0), 1.0);
        assert_close(fit.curve.evaluate(1.0), 3.0);

        // free polynomial is still the raw fitted polynomial.
        assert_close(fit.free_polynomial.evaluate(-1.0), 2.0);
        assert_close(fit.free_polynomial.evaluate(0.0), 2.0);
        assert_close(fit.free_polynomial.evaluate(1.0), 2.0);

        assert!(fit.constraint.is_some());
        assert!(matches!(fit.method, FitMethod::WeightedLeastSquares));
    }

    #[test]
    fn from_series_report_converts_fitted_values_to_array() {
        let free = series(vec![1.0], -1.0..1.0);

        let fit = Fit::from_series_report(
            report(free),
            None,
            0.0..10.0,
            FitMethod::OrdinaryLeastSquares,
        );

        assert_eq!(fit.fitted_values, arr1(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn from_series_report_converts_residuals_to_array() {
        let free = series(vec![1.0], -1.0..1.0);

        let fit = Fit::from_series_report(
            report(free),
            None,
            0.0..10.0,
            FitMethod::OrdinaryLeastSquares,
        );

        assert_eq!(fit.residuals, arr1(&[0.1, -0.2, 0.3]));
    }

    #[test]
    fn from_series_report_converts_covariance_when_present() {
        let free = series(vec![1.0, 2.0], -1.0..1.0);
        let mut report = report(free);

        report.covariance = Some(vec![vec![1.0, 0.25], vec![0.25, 4.0]]);

        let fit = Fit::from_series_report(report, None, 0.0..10.0, FitMethod::OrdinaryLeastSquares);

        assert_eq!(fit.covariance.unwrap(), arr2(&[[1.0, 0.25], [0.25, 4.0]]));
    }

    #[test]
    fn from_series_report_keeps_covariance_none_when_absent() {
        let free = series(vec![1.0], -1.0..1.0);

        let fit = Fit::from_series_report(
            report(free),
            None,
            0.0..10.0,
            FitMethod::OrdinaryLeastSquares,
        );

        assert!(fit.covariance.is_none());
    }

    #[test]
    fn from_series_report_preserves_response_domain() {
        let free = series(vec![1.0], -1.0..1.0);

        let fit = Fit::from_series_report(
            report(free),
            None,
            5.0..15.0,
            FitMethod::GeneralizedLeastSquares,
        );

        assert_eq!(fit.response_domain, 5.0..15.0);
        assert!(matches!(fit.method, FitMethod::GeneralizedLeastSquares));
    }

    #[test]
    fn all_fit_methods_are_constructible() {
        let methods = [
            FitMethod::OrdinaryLeastSquares,
            FitMethod::WeightedLeastSquares,
            FitMethod::GeneralizedLeastSquares,
            FitMethod::TotalLeastSquares,
        ];

        assert_eq!(methods.len(), 4);
    }
}
