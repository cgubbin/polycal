mod constrained;
mod unconstrained;

use crate::{
    Problem,
    fit::{Constraint, Fit, FitMethod},
    problem::GoodnessOfFit,
    problem::Uncertainty,
};

use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, Lapack, Scalar, Solve, SolveTriangularInto, UPLO};
use num_traits::{Float, FromPrimitive};
use poly_series::{
    ChebyshevError, ChebyshevSeries, FitPolynomialSeries, PolynomialRoots, PolynomialSeries,
};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::ops::Range;

#[derive(thiserror::Error, Debug)]
pub enum SolveError<E> {
    #[error("candidate fit rejected: {0:?}")]
    CandidateRejected(#[from] CandidateRejection<E>),

    #[error("no acceptable fits found: {rejections:?}")]
    NoAcceptableFit {
        rejections: Vec<CandidateRejection<E>>,
    },
}

#[derive(Debug)]
enum CandidateRejection<E> {
    FitFailed {
        degree: usize,
        source: FitError<E>,
    },
    NonMonotonic {
        degree: usize,
    },

    MonotonicityCheckFailed {
        degree: usize,
        source: ChebyshevError,
    },
    GoodnessOfFitRejected {
        degree: usize,
        chi_square: E,
        lower: E,
        upper: E,
        degrees_of_freedom: usize,
    },
}

#[derive(thiserror::Error, Debug)]
pub enum FitError<E> {
    #[error("candidate fit rejected: {0:?}")]
    Chebyshev(#[from] ChebyshevError),

    #[error("no acceptable fits found: {rejections:?}")]
    NoAcceptableFit { rejections: Vec<E> },

    #[error("method unsupported for reason {reason}")]
    Unsupported { reason: &'static str },

    #[error("uncertainty not valid for fit")]
    InvalidUncertainty,

    #[error("linalg error: {0}")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),
}

struct CandidateFit<E> {
    degree: usize,
    fit: Fit<E>,
    chi_square: E,
    score: E,
}

impl<E> Problem<E>
where
    E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
{
    pub fn solve_up_to_degree(&self, max_degree: usize) -> Result<Fit<E>, SolveError<E>> {
        let mut candidates = Vec::new();
        let mut rejections = Vec::new();

        for degree in 1..=max_degree {
            match self.generate_candidate(degree) {
                Ok(candidate) => candidates.push(candidate),
                Err(rejection) => rejections.push(rejection),
            }
        }

        if candidates.is_empty() {
            return Err(SolveError::NoAcceptableFit { rejections });
        }

        Ok(self.select_best(candidates).fit)
    }

    pub fn solve_degree(&self, degree: usize) -> Result<Fit<E>, SolveError<E>> {
        Ok(self
            .generate_candidate(degree)
            .map(|candidate| candidate.fit)?)
    }

    fn generate_candidate(&self, degree: usize) -> Result<CandidateFit<E>, CandidateRejection<E>> {
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
                let residual = self.residual_vector(&curve);
                quadratic_form_inverse_covariance(&residual, vy)
            }

            Uncertainty::XYDiagonal { .. } | Uncertainty::XYCovariance { .. } => {
                // For TLS/ODR, use the solver's objective/statistic if available.
                todo!()
            }
        }
    }

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

    // Select the best candidate from a list
    //
    // No checks are done for non-empty lists, the caller should check
    fn select_best(&self, candidates: Vec<CandidateFit<E>>) -> CandidateFit<E> {
        debug_assert!(!candidates.is_empty());
        candidates
            .into_iter()
            .min_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap()
    }

    pub(crate) fn residual_vector(&self, curve: &ChebyshevSeries<E>) -> Array1<E> {
        self.x
            .iter()
            .zip(self.y.iter())
            .map(|(&x, &y)| y - curve.evaluate(x))
            .collect()
    }

    pub(crate) fn fitted_values(&self, curve: &ChebyshevSeries<E>) -> Array1<E> {
        self.x.iter().map(|&x| curve.evaluate(x)).collect()
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
