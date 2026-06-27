//! Calibration solving and candidate selection.
//!
//! This module turns a validated [`Problem`] into a fitted calibration curve.
//!
//! Solving is performed degree-by-degree. For each candidate polynomial degree,
//! the solver:
//!
//! 1. fits a calibration curve,
//! 2. rejects non-monotonic curves,
//! 3. computes the χ² goodness-of-fit statistic,
//! 4. applies the configured goodness-of-fit validation, and
//! 5. scores the accepted candidate using the configured [`ScoringStrategy`].
//!
//! The best accepted candidate is then selected by minimum score.
//!
//! Monotonicity is mandatory: a non-monotonic curve is not considered a valid
//! calibration curve because it cannot provide a unique inverse mapping from
//! response to stimulus.
//!
//! Goodness-of-fit validation is controlled by [`GoodnessOfFit`]. Model ranking
//! is controlled separately by [`ScoringStrategy`].

mod constrained;
mod unconstrained;

use crate::{Problem, fit::Fit, problem::GoodnessOfFit, problem::Uncertainty};

use ndarray::{Array1, Array2};
use ndarray_linalg::{Lapack, Scalar, Solve};
use num_traits::{Float, FromPrimitive};
use poly_series::{ChebyshevError, ChebyshevSeries, PolynomialRoots, PolynomialSeries};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Error returned while solving a calibration problem.
#[derive(thiserror::Error, Debug)]
pub enum SolveError<E> {
    /// A single requested candidate degree was rejected.
    #[error("candidate fit rejected: {0:?}")]
    CandidateRejected(#[from] CandidateRejection<E>),

    /// No candidate degree produced an acceptable calibration curve.
    #[error("no acceptable fits found: {rejections:?}")]
    NoAcceptableFit {
        /// Rejection reasons for all attempted candidate degrees.
        rejections: Vec<CandidateRejection<E>>,
    },
}

/// Reason a candidate calibration curve was rejected.
#[derive(thiserror::Error, Debug)]
pub enum CandidateRejection<E> {
    /// Numerical fitting failed.
    #[error("candidate fit failed for degree {degree}: {source:?}")]
    FitFailed {
        /// Polynomial degree that was attempted.
        degree: usize,
        /// Underlying fitting error.
        source: FitError<E>,
    },

    /// The fitted calibration curve was not monotonic.
    #[error("candidate fit was not monotonic for degree {degree}")]
    NonMonotonic {
        /// Polynomial degree that was attempted.
        degree: usize,
    },

    /// The monotonicity check itself failed.
    #[error("monotonicity check failed for degree {degree}: {source:?}")]
    MonotonicityCheckFailed {
        /// Polynomial degree that was attempted.
        degree: usize,
        /// Underlying root-finding error.
        source: ChebyshevError,
    },

    /// The candidate failed the configured χ² goodness-of-fit validation.
    #[error("candidate failed goodness of fit test")]
    GoodnessOfFitRejected {
        /// Polynomial degree that was attempted.
        degree: usize,
        /// Observed χ² statistic.
        chi_square: E,
        /// Lower acceptable χ² bound.
        lower: E,
        /// Upper acceptable χ² bound.
        upper: E,
        /// Residual degrees of freedom.
        degrees_of_freedom: usize,
    },
}

/// Error returned while fitting a single candidate degree.
#[derive(thiserror::Error, Debug)]
pub enum FitError<E> {
    /// Error from the underlying Chebyshev polynomial implementation.
    #[error("polynomial fitting failed: {0}")]
    Chebyshev(#[from] ChebyshevError),

    /// The requested fitting method is not implemented for this uncertainty model.
    #[error("method unsupported: {reason}")]
    Unsupported {
        /// Explanation of why the fitting path is unsupported.
        reason: &'static str,
    },

    /// Supplied uncertainty values were invalid for the requested fitting method.
    #[error("uncertainty not valid for fit")]
    InvalidUncertainty,

    /// Linear algebra failure during fitting.
    #[error("linear algebra error: {0}")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),

    /// Placeholder for errors carrying attempted values.
    #[error("fit failed: {0:?}")]
    Numeric(E),
}

#[allow(dead_code)]
/// Internal representation of an accepted candidate fit.
///
/// Candidate fits are produced while scanning polynomial degrees. They carry
/// the fitted calibration curve together with the statistics used for
/// validation and model selection.
struct CandidateFit<E> {
    /// Polynomial degree used for this candidate.
    degree: usize,

    /// Fitted calibration result.
    fit: Fit<E>,

    /// χ² statistic for the final calibration curve.
    chi_square: E,

    /// Model-selection score. Lower is better.
    score: E,
}

impl<E> Problem<E>
where
    E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
{
    /// Solve the calibration problem by trying all degrees from `1` to `max_degree`.
    ///
    /// Each candidate is fitted, checked for monotonicity, optionally checked
    /// against the configured goodness-of-fit criterion, and then scored using
    /// the configured scoring strategy.
    ///
    /// Returns the accepted candidate with the lowest score.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::NoAcceptableFit`] if no candidate degree produces
    /// a valid calibration curve.
    pub fn solve_up_to_degree(&self, max_degree: usize) -> Result<Fit<E>, SolveError<E>> {
        let mut candidates = Vec::new();
        let mut rejections = Vec::new();

        for degree in 1..=max_degree {
            match self.fit_candidate(degree) {
                Ok(candidate) => candidates.push(candidate),
                Err(rejection) => rejections.push(rejection),
            }
        }

        if candidates.is_empty() {
            return Err(SolveError::NoAcceptableFit { rejections });
        }

        Ok(self.select_best(candidates).fit)
    }

    /// Solve the calibration problem using one fixed polynomial degree.
    ///
    /// Unlike [`Self::solve_up_to_degree`], this does not perform model
    /// selection. The single candidate must pass monotonicity and
    /// goodness-of-fit validation.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::CandidateRejected`] if the requested degree fails
    /// fitting or validation.
    pub fn solve_degree(&self, degree: usize) -> Result<Fit<E>, SolveError<E>> {
        Ok(self.fit_candidate(degree).map(|candidate| candidate.fit)?)
    }

    /// Solve the calibration problem using one fixed polynomial degree.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::CandidateRejected`] if the requested degree fails
    /// fitting or validation.
    fn fit_candidate(&self, degree: usize) -> Result<CandidateFit<E>, CandidateRejection<E>> {
        let fit = self
            .fit_degree(degree)
            .map_err(|source| CandidateRejection::FitFailed { degree, source })?;

        self.validate_monotonicity(degree, &fit)?;

        let chi_square = self.chi_square(&fit);

        self.validate_goodness_of_fit(degree, chi_square)?;

        let score = self.strategy.score(degree, self.len(), chi_square);

        Ok(CandidateFit {
            fit,
            degree,
            chi_square,
            score,
        })
    }

    /// Compute the χ² statistic for a fitted calibration curve.
    ///
    /// The residuals are always computed against the final calibration curve,
    /// including any applied constraint.
    fn chi_square(&self, fit: &Fit<E>) -> E {
        let curve = fit.calibration_curve();

        match &self.uncertainty {
            Uncertainty::None => {
                self.x
                    .iter()
                    .zip(self.y.iter())
                    .fold(E::zero(), |sum, (&x, &y)| {
                        let r = y - curve.evaluate(x);
                        sum + r * r
                    })
            }

            Uncertainty::YDiagonal { uy } => self.x.iter().zip(self.y.iter()).zip(uy.iter()).fold(
                E::zero(),
                |sum, ((&x, &y), &u)| {
                    let r = y - curve.evaluate(x);
                    sum + (r * r) / (u * u)
                },
            ),

            Uncertainty::YCovariance { vy } => {
                let residual = self.residual_vector(curve);
                quadratic_form_inverse_covariance(&residual, vy)
            }

            Uncertainty::XYDiagonal { .. } | Uncertainty::XYCovariance { .. } => {
                // TODO: For TLS/ODR, use the solver's objective/statistic if available.
                self.residual_vector(curve)
                    .iter()
                    .fold(E::zero(), |sum, &r| sum + r * r)
            }
        }
    }

    /// Validate that the final calibration curve is monotonic.
    ///
    /// Monotonicity is mandatory because calibration requires a unique inverse
    /// response-to-stimulus mapping.
    fn validate_monotonicity(
        &self,
        degree: usize,
        fit: &Fit<E>,
    ) -> Result<(), CandidateRejection<E>> {
        match fit.calibration_curve().is_monotonic() {
            Ok(true) => Ok(()),

            Ok(false) => Err(CandidateRejection::NonMonotonic { degree }),

            Err(source) => Err(CandidateRejection::MonotonicityCheckFailed { degree, source }),
        }
    }

    /// Apply the configured χ² goodness-of-fit validation.
    fn validate_goodness_of_fit(
        &self,
        degree: usize,
        chi_square: E,
    ) -> Result<(), CandidateRejection<E>>
    where
        E: Float + FromPrimitive,
    {
        let GoodnessOfFit::ChiSquare { confidence } = self.goodness_of_fit else {
            return Ok(());
        };

        let dof = self.degrees_of_freedom(degree);

        if dof == 0 {
            return Ok(());
        }

        let alpha = confidence.significance().into_inner();
        let lower_p = alpha / E::from_f64(2.0).unwrap();
        let upper_p = E::one() - lower_p;

        let lower = chi_square_quantile(lower_p, dof);
        let upper = chi_square_quantile(upper_p, dof);

        if lower <= chi_square && chi_square <= upper {
            Ok(())
        } else {
            Err(CandidateRejection::GoodnessOfFitRejected {
                degree,
                chi_square,
                lower,
                upper,
                degrees_of_freedom: dof,
            })
        }
    }

    /// Select the accepted candidate with the lowest model-selection score.
    fn select_best(&self, candidates: Vec<CandidateFit<E>>) -> CandidateFit<E> {
        debug_assert!(!candidates.is_empty());
        candidates
            .into_iter()
            .min_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap()
    }

    /// Return residuals `y_i - f(x_i)` for the final calibration curve.
    pub(crate) fn residual_vector(&self, curve: &ChebyshevSeries<E>) -> Array1<E> {
        self.x
            .iter()
            .zip(self.y.iter())
            .map(|(&x, &y)| y - curve.evaluate(x))
            .collect()
    }
}

fn chi_square_quantile<E>(probability: E, degrees_of_freedom: usize) -> E
where
    E: Float + FromPrimitive,
{
    let distribution = ChiSquared::new(degrees_of_freedom as f64).unwrap();
    E::from_f64(distribution.inverse_cdf(probability.to_f64().unwrap())).unwrap()
}

fn quadratic_form_inverse_covariance<E>(r: &Array1<E>, covariance: &Array2<E>) -> E
where
    E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
{
    let solved = covariance
        .clone()
        .solve_into(r.clone())
        .expect("validated covariance should solve");

    r.dot(&solved)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::score::ScoringStrategy;

    use confi::ConfidenceLevel;

    use ndarray::{Array1, arr1, arr2};
    use poly_series::{ChebyshevSeries, FitReport as SeriesFitReport};

    use crate::{Problem, problem::GoodnessOfFit};
    use crate::{
        fit::{Constraint, Fit, FitMethod},
        problem::Uncertainty,
    };

    const EPS: f64 = 1.0e-12;

    impl<E: Float + FromPrimitive> Problem<E> {
        pub(crate) fn fitted_values(&self, curve: &ChebyshevSeries<E>) -> Array1<E> {
            self.x.iter().map(|&x| curve.evaluate(x)).collect()
        }
    }

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= EPS,
            "expected {lhs} ≈ {rhs}, difference = {}",
            (lhs - rhs).abs()
        );
    }

    fn assert_array_close(lhs: &Array1<f64>, rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());

        for (&lhs, &rhs) in lhs.iter().zip(rhs.iter()) {
            assert_close(lhs, rhs);
        }
    }

    fn series(coefficients: Vec<f64>) -> ChebyshevSeries<f64> {
        ChebyshevSeries::new(coefficients, 0.0..2.0).unwrap()
    }

    fn fit_from_curve(curve: ChebyshevSeries<f64>) -> Fit<f64> {
        let report = SeriesFitReport {
            series: curve.clone(),
            coefficients: vec![],
            covariance: None,
            fitted_values: vec![],
            residuals: vec![],
            degrees_of_freedom: 0,
            residual_sum_of_squares: 0.0,
            residual_variance: None,
        };

        Fit::from_series_report(report, None, 0.0..10.0, FitMethod::OrdinaryLeastSquares)
    }

    fn problem_with_uncertainty(uncertainty: Uncertainty<f64>) -> Problem<f64> {
        Problem {
            x: arr1(&[0.0, 1.0, 2.0]),
            y: arr1(&[1.0, 4.0, 9.0]),
            uncertainty,
            domain: 0.0..2.0,
            response_domain: 1.0..9.0,
            strategy: ScoringStrategy::ChiSquare,
            constraint: None,
            goodness_of_fit: GoodnessOfFit::Disabled,
        }
    }

    #[test]
    fn residual_vector_uses_final_curve() {
        // p(x) = 1 + 4t, with domain [0, 2].
        //
        // x = 0 -> t = -1 -> p = -3
        // x = 1 -> t =  0 -> p =  1
        // x = 2 -> t =  1 -> p =  5
        let fit = fit_from_curve(series(vec![1.0, 4.0]));
        let problem = problem_with_uncertainty(Uncertainty::None);

        let residuals = problem.residual_vector(fit.calibration_curve());

        // y - p(x) = [1 - (-3), 4 - 1, 9 - 5]
        assert_array_close(&residuals, &[4.0, 3.0, 4.0]);
    }

    #[test]
    fn fitted_values_use_final_curve() {
        let fit = fit_from_curve(series(vec![1.0, 4.0]));
        let problem = problem_with_uncertainty(Uncertainty::None);

        let fitted = problem.fitted_values(fit.calibration_curve());

        assert_array_close(&fitted, &[-3.0, 1.0, 5.0]);
    }

    #[test]
    fn chi_square_without_uncertainty_is_sum_squared_residuals() {
        let fit = fit_from_curve(series(vec![1.0, 4.0]));
        let problem = problem_with_uncertainty(Uncertainty::None);

        let chi_square = problem.chi_square(&fit);

        // residuals = [4, 3, 4]
        assert_close(chi_square, 4.0 * 4.0 + 3.0 * 3.0 + 4.0 * 4.0);
    }

    #[test]
    fn chi_square_with_y_uncertainty_uses_variance() {
        let fit = fit_from_curve(series(vec![1.0, 4.0]));

        let problem = problem_with_uncertainty(Uncertainty::YDiagonal {
            uy: arr1(&[2.0, 3.0, 4.0]),
        });

        let chi_square = problem.chi_square(&fit);

        // residuals = [4, 3, 4]
        // chi² = 4²/2² + 3²/3² + 4²/4² = 4 + 1 + 1 = 6
        assert_close(chi_square, 6.0);
    }

    #[test]
    fn chi_square_with_y_covariance_uses_inverse_covariance() {
        let fit = fit_from_curve(series(vec![1.0, 4.0]));

        let problem = problem_with_uncertainty(Uncertainty::YCovariance {
            vy: arr2(&[[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 16.0]]),
        });

        let chi_square = problem.chi_square(&fit);

        // Equivalent to the diagonal uncertainty case above:
        // rᵀ V⁻¹ r = 4²/4 + 3²/9 + 4²/16 = 6
        assert_close(chi_square, 6.0);
    }

    #[test]
    fn residual_vector_uses_constrained_curve_not_free_polynomial() {
        let free = series(vec![2.0]); // constant 2

        let constraint = Constraint {
            additive: series(vec![1.0]),
            multiplicative: series(vec![0.0, 1.0]), // t
        };

        let report = SeriesFitReport {
            series: free,
            coefficients: vec![],
            covariance: None,
            fitted_values: vec![],
            residuals: vec![],
            degrees_of_freedom: 0,
            residual_sum_of_squares: 0.0,
            residual_variance: None,
        };

        let fit = Fit::from_series_report(
            report,
            Some(constraint),
            0.0..10.0,
            FitMethod::OrdinaryLeastSquares,
        );

        let problem = problem_with_uncertainty(Uncertainty::None);
        let residuals = problem.residual_vector(fit.calibration_curve());

        // constrained curve = 1 + 2t = [-1, 1, 3]
        // y = [1, 4, 9]
        assert_array_close(&residuals, &[2.0, 3.0, 6.0]);
    }

    #[test]
    fn chi_square_with_unit_uncertainty_matches_unweighted_sum_squares() {
        let fit = fit_from_curve(series(vec![1.0, 4.0]));

        let unweighted = problem_with_uncertainty(Uncertainty::None).chi_square(&fit);

        let weighted = problem_with_uncertainty(Uncertainty::YDiagonal {
            uy: arr1(&[1.0, 1.0, 1.0]),
        })
        .chi_square(&fit);

        assert_close(weighted, unweighted);
    }

    #[test]
    fn chi_square_with_scaled_uncertainty_decreases_weight() {
        let fit = fit_from_curve(series(vec![1.0, 4.0]));

        let small_uncertainty = problem_with_uncertainty(Uncertainty::YDiagonal {
            uy: arr1(&[1.0, 1.0, 1.0]),
        })
        .chi_square(&fit);

        let large_uncertainty = problem_with_uncertainty(Uncertainty::YDiagonal {
            uy: arr1(&[2.0, 2.0, 2.0]),
        })
        .chi_square(&fit);

        assert_close(large_uncertainty, small_uncertainty / 4.0);
    }

    #[test]
    fn fitted_values_and_residuals_are_consistent() {
        let fit = fit_from_curve(series(vec![1.0, 4.0]));
        let problem = problem_with_uncertainty(Uncertainty::None);

        let fitted = problem.fitted_values(fit.calibration_curve());
        let residuals = problem.residual_vector(fit.calibration_curve());

        for ((&y, &fit), &residual) in problem.y.iter().zip(fitted.iter()).zip(residuals.iter()) {
            assert_close(residual, y - fit);
        }
    }

    #[test]
    fn chi_square_with_correlated_covariance_uses_full_inverse() {
        let fit = fit_from_curve(series(vec![1.0, 4.0]));

        let problem = problem_with_uncertainty(Uncertainty::YCovariance {
            vy: arr2(&[[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 4.0]]),
        });

        let chi_square = problem.chi_square(&fit);

        assert_close(chi_square, 38.0 / 3.0);
    }

    fn problem(goodness_of_fit: GoodnessOfFit<f64>) -> Problem<f64> {
        Problem {
            x: arr1(&[-1.0, 0.0, 1.0, 2.0, 3.0]),
            y: arr1(&[-1.0, 0.0, 1.0, 2.0, 3.0]),
            uncertainty: Uncertainty::None,
            domain: -1.0..3.0,
            response_domain: -1.0..3.0,
            strategy: ScoringStrategy::ChiSquare,
            constraint: None,
            goodness_of_fit,
        }
    }

    #[test]
    fn monotonic_curve_is_accepted() {
        let problem = problem(GoodnessOfFit::Disabled);

        // p(t) = T1(t), monotonic on [-1, 1].
        let fit = fit_from_curve(series(vec![0.0, 1.0]));

        let result = problem.validate_monotonicity(1, &fit);

        assert!(result.is_ok());
    }

    #[test]
    fn non_monotonic_curve_is_rejected() {
        let problem = problem(GoodnessOfFit::Disabled);

        // p(t) = T2(t), derivative root inside domain.
        let fit = fit_from_curve(series(vec![0.0, 0.0, 1.0]));

        let result = problem.validate_monotonicity(2, &fit);

        assert!(matches!(
            result,
            Err(CandidateRejection::NonMonotonic { degree: 2 })
        ));
    }

    #[test]
    fn goodness_of_fit_disabled_accepts_any_chi_square() {
        let problem = problem(GoodnessOfFit::Disabled);

        let result = problem.validate_goodness_of_fit(1, 1.0e12);

        assert!(result.is_ok());
    }

    #[test]
    fn goodness_of_fit_rejects_outside_chi_square_interval() {
        let problem = problem(GoodnessOfFit::ChiSquare {
            confidence: ConfidenceLevel::new(0.95).unwrap(),
        });

        // n = 5, degree = 1 -> dof = 3.
        // This is far outside the upper 95% central chi-square interval.
        let result = problem.validate_goodness_of_fit(1, 1.0e6);

        assert!(matches!(
            result,
            Err(CandidateRejection::GoodnessOfFitRejected {
                degree: 1,
                chi_square,
                degrees_of_freedom: 3,
                ..
            }) if chi_square == 1.0e6
        ));
    }

    #[test]
    fn goodness_of_fit_accepts_inside_chi_square_interval() {
        let problem = problem(GoodnessOfFit::ChiSquare {
            confidence: ConfidenceLevel::new(0.95).unwrap(),
        });

        // n = 5, degree = 1 -> dof = 3.
        // χ² = 3 is near the centre of the χ² distribution with 3 dof.
        let result = problem.validate_goodness_of_fit(1, 3.0);

        assert!(result.is_ok());
    }
}
