use crate::fit::Fit;

use num_traits::{Float, FromPrimitive};
use poly_series::{ChebyshevSeries, PolynomialRoots, PolynomialSeries};
use std::ops::Range;

#[derive(Debug, thiserror::Error)]
pub enum EvaluationError<E> {
    #[error("stimulus {value:?} is outside calibration domain {domain:?}")]
    StimulusOutsideDomain { value: E, domain: Range<E> },

    #[error("response {value:?} is outside response domain {domain:?}")]
    ResponseOutsideDomain { value: E, domain: Range<E> },

    #[error("calibration curve is not monotonic")]
    NonMonotonic,

    #[error("invalid value in uncertainty")]
    InvalidUncertainty,

    #[error("no inverse solution found for response {value:?}")]
    NoInverseSolution { value: E },

    #[error("multiple inverse solutions found for response {value:?}: {roots:?}")]
    MultipleInverseSolutions { value: E, roots: Vec<E> },

    #[error("root finding failed")]
    RootFinding(#[from] poly_series::ChebyshevError),

    #[error("local sensitivity is too close to zero at stimulus {stimulus:?}: slope {slope:?}")]
    NearZeroSensitivity { stimulus: E, slope: E },
}

#[derive(Clone, Debug)]
pub struct Estimate<E> {
    value: E,
    standard_uncertainty: Option<E>,
}

impl<E> Fit<E>
where
    E: Float + FromPrimitive + ndarray_linalg::Scalar<Real = E> + ndarray_linalg::Lapack,
{
    pub fn response(&self, stimulus: E) -> Result<E, EvaluationError<E>> {
        if !contains_closed(&self.curve.domain(), stimulus) {
            return Err(EvaluationError::StimulusOutsideDomain {
                value: stimulus,
                domain: self.curve.domain(),
            });
        }

        Ok(self.curve.evaluate(stimulus))
    }

    pub fn stimulus(&self, response: E) -> Result<E, EvaluationError<E>> {
        if !contains_closed(&self.response_domain, response) {
            return Err(EvaluationError::ResponseOutsideDomain {
                value: response,
                domain: self.response_domain.clone(),
            });
        }

        if !self.curve.is_monotonic()? {
            return Err(EvaluationError::NonMonotonic);
        }

        let shifted =
            self.curve.clone() - ChebyshevSeries::new(vec![response], self.curve.domain())?;

        let roots = shifted.roots_in_domain()?;

        match roots.as_slice() {
            [] => Err(EvaluationError::NoInverseSolution { value: response }),
            [root] => Ok(*root),
            _ => Err(EvaluationError::MultipleInverseSolutions {
                value: response,
                roots,
            }),
        }
    }

    pub fn response_with_uncertainty(
        &self,
        stimulus: E,
        stimulus_uncertainty: E,
    ) -> Result<Estimate<E>, EvaluationError<E>> {
        let value = self.response(stimulus)?;

        validate_standard_uncertainty(stimulus_uncertainty)?;

        let slope = self.curve.first_derivative().evaluate(stimulus);

        let variance_from_input = slope * slope * stimulus_uncertainty * stimulus_uncertainty;

        let variance_from_coefficients = self
            .variance_from_coefficients(stimulus)
            .unwrap_or_else(E::zero);

        Ok(Estimate {
            value,
            standard_uncertainty: Some(Float::sqrt(
                variance_from_input + variance_from_coefficients,
            )),
        })
    }

    pub fn stimulus_estimate_first_order(
        &self,
        response: E,
        response_uncertainty: E,
    ) -> Result<Estimate<E>, EvaluationError<E>>
    where
        E: ndarray_linalg::Scalar<Real = E> + ndarray_linalg::Lapack,
    {
        validate_standard_uncertainty(response_uncertainty)?;

        let stimulus = self.stimulus(response)?;
        let slope = self.curve.first_derivative().evaluate(stimulus);

        let min_slope = Float::sqrt(E::epsilon());

        if Float::abs(slope) <= min_slope {
            return Err(EvaluationError::NearZeroSensitivity { stimulus, slope });
        }

        let variance_from_response = response_uncertainty * response_uncertainty / (slope * slope);

        let variance_from_coefficients = self
            .variance_from_coefficients(stimulus)
            .unwrap_or_else(E::zero)
            / (slope * slope);

        Ok(Estimate {
            value: stimulus,
            standard_uncertainty: Some(Float::sqrt(
                variance_from_response + variance_from_coefficients,
            )),
        })
    }

    fn variance_from_coefficients(&self, stimulus: E) -> Option<E> {
        let covariance = self.covariance.as_ref()?;
        let jacobian = self.coefficient_jacobian(stimulus);

        let mut variance = E::zero();

        for i in 0..jacobian.len() {
            for j in 0..jacobian.len() {
                variance = variance + jacobian[i] * covariance[[i, j]] * jacobian[j];
            }
        }

        Some(variance)
    }

    fn coefficient_jacobian(&self, stimulus: E) -> Vec<E> {
        let degree = self.free_polynomial.degree();
        let basis = chebyshev_basis_at(stimulus, degree, &self.free_polynomial.domain());

        let multiplier = self
            .constraint
            .as_ref()
            .map_or_else(E::one, |c| c.multiplicative.evaluate(stimulus));

        basis.into_iter().map(|value| multiplier * value).collect()
    }

    pub fn response_estimate(&self, stimulus: E) -> Result<Estimate<E>, EvaluationError<E>> {
        Ok(Estimate {
            value: self.response(stimulus)?,
            standard_uncertainty: None,
        })
    }

    pub fn response_unchecked(&self, stimulus: E) -> E {
        self.curve.evaluate(stimulus)
    }
}

fn validate_standard_uncertainty<E>(uncertainty: E) -> Result<(), EvaluationError<E>>
where
    E: Float,
{
    if uncertainty.is_finite() && uncertainty >= E::zero() {
        Ok(())
    } else {
        Err(EvaluationError::InvalidUncertainty)
    }
}

fn contains_closed<E>(domain: &Range<E>, value: E) -> bool
where
    E: PartialOrd,
{
    domain.start <= value && value <= domain.end
}

fn chebyshev_basis_at<E>(x: E, degree: usize, domain: &Range<E>) -> Vec<E>
where
    E: Float + FromPrimitive,
{
    let t = poly_series::scaling::to_scaled(x, domain);

    let mut values = vec![E::zero(); degree + 1];
    values[0] = E::one();

    if degree == 0 {
        return values;
    }

    values[1] = t;

    let two = E::one() + E::one();

    for n in 2..=degree {
        values[n] = two * t * values[n - 1] - values[n - 2];
    }

    values
}

#[cfg(test)]
mod evaluation_tests {
    use super::*;
    use crate::fit::FitMethod;

    use ndarray::arr1;
    use poly_series::ChebyshevSeries;

    const EPS: f64 = 1.0e-10;

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= EPS,
            "expected {lhs} ≈ {rhs}, difference = {}",
            (lhs - rhs).abs()
        );
    }

    fn linear_fit() -> Fit<f64> {
        // y = x on domain [0, 10].
        //
        // Chebyshev T1(t) = t, and t = x / 5 - 1.
        // So y = x = 5t + 5.
        let curve = ChebyshevSeries::new(vec![5.0, 5.0], 0.0..10.0).unwrap();

        Fit {
            curve: curve.clone(),
            free_polynomial: curve,
            covariance: None,
            fitted_values: arr1(&[]),
            residuals: arr1(&[]),
            constraint: None,
            response_domain: 0.0..10.0,
            method: FitMethod::OrdinaryLeastSquares,
        }
    }

    fn decreasing_fit() -> Fit<f64> {
        // y = 10 - x on domain [0, 10].
        //
        // t = x / 5 - 1, so 10 - x = 5 - 5t.
        let curve = ChebyshevSeries::new(vec![5.0, -5.0], 0.0..10.0).unwrap();

        Fit {
            curve: curve.clone(),
            free_polynomial: curve,
            covariance: None,
            fitted_values: arr1(&[]),
            residuals: arr1(&[]),
            constraint: None,
            response_domain: 0.0..10.0,
            method: FitMethod::OrdinaryLeastSquares,
        }
    }

    fn non_monotonic_fit() -> Fit<f64> {
        // T2 on [-1, 1] is non-monotonic.
        let curve = ChebyshevSeries::new(vec![0.0, 0.0, 1.0], -1.0..1.0).unwrap();

        Fit {
            curve: curve.clone(),
            free_polynomial: curve,
            covariance: None,
            fitted_values: arr1(&[]),
            residuals: arr1(&[]),
            constraint: None,
            response_domain: -1.0..1.0,
            method: FitMethod::OrdinaryLeastSquares,
        }
    }

    #[test]
    fn response_evaluates_curve_inside_domain() {
        let fit = linear_fit();

        assert_close(fit.response(0.0).unwrap(), 0.0);
        assert_close(fit.response(2.5).unwrap(), 2.5);
        assert_close(fit.response(10.0).unwrap(), 10.0);
    }

    #[test]
    fn response_rejects_stimulus_below_domain() {
        let fit = linear_fit();

        let err = fit.response(-0.1).unwrap_err();

        assert!(matches!(
            err,
            EvaluationError::StimulusOutsideDomain { value, .. }
                if value == -0.1
        ));
    }

    #[test]
    fn response_rejects_stimulus_above_domain() {
        let fit = linear_fit();

        let err = fit.response(10.1).unwrap_err();

        assert!(matches!(
            err,
            EvaluationError::StimulusOutsideDomain { value, .. }
                if value == 10.1
        ));
    }

    #[test]
    fn response_accepts_domain_endpoints() {
        let fit = linear_fit();

        assert_close(fit.response(0.0).unwrap(), 0.0);
        assert_close(fit.response(10.0).unwrap(), 10.0);
    }

    #[test]
    fn stimulus_inverts_increasing_curve() {
        let fit = linear_fit();

        assert_close(fit.stimulus(0.0).unwrap(), 0.0);
        assert_close(fit.stimulus(2.5).unwrap(), 2.5);
        assert_close(fit.stimulus(10.0).unwrap(), 10.0);
    }

    #[test]
    fn stimulus_inverts_decreasing_curve() {
        let fit = decreasing_fit();

        assert_close(fit.stimulus(10.0).unwrap(), 0.0);
        assert_close(fit.stimulus(7.5).unwrap(), 2.5);
        assert_close(fit.stimulus(0.0).unwrap(), 10.0);
    }

    #[test]
    fn stimulus_rejects_response_below_response_domain() {
        let fit = linear_fit();

        let err = fit.stimulus(-0.1).unwrap_err();

        assert!(matches!(
            err,
            EvaluationError::ResponseOutsideDomain { value, .. }
                if value == -0.1
        ));
    }

    #[test]
    fn stimulus_rejects_response_above_response_domain() {
        let fit = linear_fit();

        let err = fit.stimulus(10.1).unwrap_err();

        assert!(matches!(
            err,
            EvaluationError::ResponseOutsideDomain { value, .. }
                if value == 10.1
        ));
    }

    #[test]
    fn stimulus_accepts_response_domain_endpoints() {
        let fit = linear_fit();

        assert_close(fit.stimulus(0.0).unwrap(), 0.0);
        assert_close(fit.stimulus(10.0).unwrap(), 10.0);
    }

    #[test]
    fn stimulus_rejects_non_monotonic_curve() {
        let fit = non_monotonic_fit();

        let err = fit.stimulus(0.5).unwrap_err();

        assert!(matches!(err, EvaluationError::NonMonotonic));
    }

    #[test]
    fn stimulus_reports_no_inverse_solution_when_response_domain_is_too_broad() {
        let curve = ChebyshevSeries::new(vec![5.0, 5.0], 0.0..10.0).unwrap();

        let fit = Fit {
            curve: curve.clone(),
            free_polynomial: curve,
            covariance: None,
            fitted_values: arr1(&[]),
            residuals: arr1(&[]),
            constraint: None,
            response_domain: -100.0..100.0,
            method: FitMethod::OrdinaryLeastSquares,
        };

        let err = fit.stimulus(50.0).unwrap_err();

        assert!(matches!(
            err,
            EvaluationError::NoInverseSolution { value }
                if value == 50.0
        ));
    }

    #[test]
    fn response_unchecked_evaluates_without_domain_check() {
        let fit = linear_fit();

        assert_close(fit.response_unchecked(12.0), 12.0);
    }
}

#[cfg(test)]
mod first_order_uncertainty_tests {
    use super::*;
    use crate::fit::{Constraint, FitMethod};

    use ndarray::{arr1, arr2};
    use poly_series::ChebyshevSeries;

    const EPS: f64 = 1.0e-10;

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= EPS,
            "expected {lhs} ≈ {rhs}, difference = {}",
            (lhs - rhs).abs()
        );
    }

    fn linear_fit_without_covariance() -> Fit<f64> {
        // y = 2x on [0, 10].
        //
        // t = x / 5 - 1, so 2x = 10t + 10.
        let curve = ChebyshevSeries::new(vec![10.0, 10.0], 0.0..10.0).unwrap();

        Fit {
            curve: curve.clone(),
            free_polynomial: curve,
            covariance: None,
            fitted_values: arr1(&[]),
            residuals: arr1(&[]),
            constraint: None,
            response_domain: 0.0..20.0,
            method: FitMethod::OrdinaryLeastSquares,
        }
    }

    fn linear_fit_with_covariance() -> Fit<f64> {
        let curve = ChebyshevSeries::new(vec![10.0, 10.0], 0.0..10.0).unwrap();

        Fit {
            curve: curve.clone(),
            free_polynomial: curve,
            covariance: Some(arr2(&[[4.0, 0.0], [0.0, 9.0]])),
            fitted_values: arr1(&[]),
            residuals: arr1(&[]),
            constraint: None,
            response_domain: 0.0..20.0,
            method: FitMethod::OrdinaryLeastSquares,
        }
    }

    fn constrained_linear_fit_with_covariance() -> Fit<f64> {
        // free q(x) = 2.
        let free = ChebyshevSeries::new(vec![2.0], -1.0..1.0).unwrap();

        // curve(x) = additive + multiplicative * q = 0 + x * 2 = 2x.
        let constraint = Constraint {
            additive: ChebyshevSeries::new(vec![0.0], -1.0..1.0).unwrap(),
            multiplicative: ChebyshevSeries::new(vec![0.0, 1.0], -1.0..1.0).unwrap(),
        };

        let curve = constraint.apply(&free);

        Fit {
            curve,
            free_polynomial: free,
            covariance: Some(arr2(&[[9.0]])),
            fitted_values: arr1(&[]),
            residuals: arr1(&[]),
            constraint: Some(constraint),
            response_domain: -2.0..2.0,
            method: FitMethod::OrdinaryLeastSquares,
        }
    }

    #[test]
    fn response_with_uncertainty_propagates_stimulus_uncertainty() {
        let fit = linear_fit_without_covariance();

        let estimate = fit.response_with_uncertainty(5.0, 0.25).unwrap();

        assert_close(estimate.value, 10.0);

        // y = 2x, dy/dx = 2, ux = 0.25 -> uy = 0.5.
        assert_close(estimate.standard_uncertainty.unwrap(), 0.5);
    }

    #[test]
    fn response_with_uncertainty_includes_coefficient_covariance() {
        let fit = linear_fit_with_covariance();

        let estimate = fit.response_with_uncertainty(5.0, 0.0).unwrap();

        assert_close(estimate.value, 10.0);

        // At x = 5, t = 0, basis = [1, 0].
        // Cov = diag([4, 9]), so coefficient variance = 4.
        assert_close(estimate.standard_uncertainty.unwrap(), 2.0);
    }

    #[test]
    fn response_with_uncertainty_combines_input_and_coefficient_variance() {
        let fit = linear_fit_with_covariance();

        let estimate = fit.response_with_uncertainty(5.0, 0.25).unwrap();

        assert_close(estimate.value, 10.0);

        // coefficient variance = 4.
        // input variance contribution = (2 * 0.25)^2 = 0.25.
        // total std = sqrt(4.25).
        assert_close(estimate.standard_uncertainty.unwrap(), 4.25_f64.sqrt());
    }

    #[test]
    fn response_with_uncertainty_propagates_through_constraint() {
        let fit = constrained_linear_fit_with_covariance();

        let estimate = fit.response_with_uncertainty(0.5, 0.0).unwrap();

        assert_close(estimate.value, 1.0);

        // free q has one coefficient with variance 9.
        // constrained model is f(x) = multiplicative(x) * q.
        // multiplicative(0.5) = 0.5.
        // J = 0.5 * [T0] = [0.5].
        // variance = 0.5² * 9 = 2.25, std = 1.5.
        assert_close(estimate.standard_uncertainty.unwrap(), 1.5);
    }

    #[test]
    fn response_with_uncertainty_rejects_negative_uncertainty() {
        let fit = linear_fit_without_covariance();

        let err = fit.response_with_uncertainty(5.0, -0.1).unwrap_err();

        assert!(matches!(err, EvaluationError::InvalidUncertainty));
    }

    #[test]
    fn response_with_uncertainty_rejects_nan_uncertainty() {
        let fit = linear_fit_without_covariance();

        let err = fit.response_with_uncertainty(5.0, f64::NAN).unwrap_err();

        assert!(matches!(err, EvaluationError::InvalidUncertainty));
    }

    #[test]
    fn response_with_uncertainty_rejects_stimulus_outside_domain() {
        let fit = linear_fit_without_covariance();

        let err = fit.response_with_uncertainty(11.0, 0.1).unwrap_err();

        assert!(matches!(
            err,
            EvaluationError::StimulusOutsideDomain { value, .. }
                if value == 11.0
        ));
    }

    #[test]
    fn stimulus_estimate_first_order_propagates_response_uncertainty() {
        let fit = linear_fit_without_covariance();

        let estimate = fit.stimulus_estimate_first_order(10.0, 0.5).unwrap();

        assert_close(estimate.value, 5.0);

        // x = y / 2, so ux = uy / 2 = 0.25.
        assert_close(estimate.standard_uncertainty.unwrap(), 0.25);
    }

    #[test]
    fn stimulus_estimate_first_order_includes_coefficient_covariance() {
        let fit = linear_fit_with_covariance();

        let estimate = fit.stimulus_estimate_first_order(10.0, 0.0).unwrap();

        assert_close(estimate.value, 5.0);

        // At x = 5, coefficient std in y is 2.
        // dy/dx = 2, so coefficient-induced ux = 1.
        assert_close(estimate.standard_uncertainty.unwrap(), 1.0);
    }

    #[test]
    fn stimulus_estimate_first_order_combines_response_and_coefficient_variance() {
        let fit = linear_fit_with_covariance();

        let estimate = fit.stimulus_estimate_first_order(10.0, 0.5).unwrap();

        assert_close(estimate.value, 5.0);

        // response variance contribution: 0.5² / 2² = 0.0625.
        // coefficient variance contribution: 4 / 2² = 1.
        // total std = sqrt(1.0625).
        assert_close(estimate.standard_uncertainty.unwrap(), 1.0625_f64.sqrt());
    }

    #[test]
    fn stimulus_estimate_first_order_propagates_constraint_coefficient_covariance() {
        let fit = constrained_linear_fit_with_covariance();

        let estimate = fit.stimulus_estimate_first_order(1.0, 0.0).unwrap();

        assert_close(estimate.value, 0.5);

        // As above, coefficient std in y at x=0.5 is 1.5.
        // dy/dx = 2, so ux = 0.75.
        assert_close(estimate.standard_uncertainty.unwrap(), 0.75);
    }

    #[test]
    fn stimulus_estimate_first_order_rejects_negative_uncertainty() {
        let fit = linear_fit_without_covariance();

        let err = fit.stimulus_estimate_first_order(10.0, -0.1).unwrap_err();

        assert!(matches!(err, EvaluationError::InvalidUncertainty));
    }

    #[test]
    fn stimulus_estimate_first_order_rejects_response_outside_domain() {
        let fit = linear_fit_without_covariance();

        let err = fit.stimulus_estimate_first_order(21.0, 0.1).unwrap_err();

        assert!(matches!(
            err,
            EvaluationError::ResponseOutsideDomain { value, .. }
                if value == 21.0
        ));
    }

    #[test]
    fn stimulus_estimate_first_order_rejects_near_zero_sensitivity() {
        // Constant curve y = 1 on [0, 10].
        let curve = ChebyshevSeries::new(vec![1.0], 0.0..10.0).unwrap();

        let fit = Fit {
            curve: curve.clone(),
            free_polynomial: curve,
            covariance: None,
            fitted_values: arr1(&[]),
            residuals: arr1(&[]),
            constraint: None,
            response_domain: 1.0..1.0,
            method: FitMethod::OrdinaryLeastSquares,
        };

        let err = fit.stimulus_estimate_first_order(1.0, 0.1).unwrap_err();

        // Depending on your stimulus() implementation, this may become
        // NoInverseSolution before reaching NearZeroSensitivity. If so, remove
        // this test or use a curve with a tiny but nonzero slope.
        assert!(matches!(
            err,
            EvaluationError::NearZeroSensitivity { .. }
                | EvaluationError::NoInverseSolution { .. }
                | EvaluationError::MultipleInverseSolutions { .. }
        ));
    }
}
