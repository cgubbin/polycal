use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use poly_series::{ChebyshevSeries, PolynomialSeries};
use std::ops::Range;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// A constraint.
///
/// Given a constraint we use the problem y = `p_n(x`, a) * multiplicative(x) + additive(x). A
/// carefully constructed constraint can ensure the response variable and it's derivatives obeys
/// certain pre-conditions such as passing through the origin.
pub struct Constraint<E> {
    /// Additive component of the constraint
    pub(crate) additive: ChebyshevSeries<E>,
    /// Multiplicative component of the constraint
    pub(crate) multiplicative: ChebyshevSeries<E>,
}

impl<E> Constraint<E> {
    pub fn apply(&self, series: &ChebyshevSeries<E>) -> ChebyshevSeries<E>
    where
        E: Float + FromPrimitive,
    {
        self.additive.clone() + self.multiplicative.clone() * series.clone()
    }
}

#[derive(Clone, Debug)]
pub struct Fit<E> {
    pub(crate) curve: ChebyshevSeries<E>,
    pub(crate) free_polynomial: ChebyshevSeries<E>,
    pub(crate) covariance: Option<Array2<E>>,
    pub(crate) fitted_values: Array1<E>,
    pub(crate) residuals: Array1<E>,
    pub(crate) constraint: Option<Constraint<E>>,
    pub(crate) response_domain: Range<E>,
    pub(crate) method: FitMethod,
}

impl<E> Fit<E> {
    pub fn free_polynomial(&self) -> &ChebyshevSeries<E> {
        &self.free_polynomial
    }

    pub fn calibration_curve(&self) -> &ChebyshevSeries<E> {
        &self.curve
    }

    pub fn evaluate(&self, x: E) -> E
    where
        E: Float + FromPrimitive,
    {
        self.curve.evaluate(x)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FitMethod {
    OrdinaryLeastSquares,
    WeightedLeastSquares,
    GeneralizedLeastSquares,
    TotalLeastSquares,
}

impl<E> Fit<E> {
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
            covariance: report
                .covariance
                .map(|values| covariance_from_vecs(values).unwrap()),
            fitted_values: Array1::from_vec(report.fitted_values),
            residuals: Array1::from_vec(report.residuals),
            constraint,
            response_domain,
            method,
        }
    }
}

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
