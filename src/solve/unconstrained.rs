use crate::{
    Problem,
    fit::{Fit, FitMethod},
    problem::Uncertainty,
};

use super::FitError;

use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, Lapack, Scalar, SolveTriangularInto, UPLO};
use num_traits::{Float, FromPrimitive};
use poly_series::ChebyshevSeries;

impl<E> Problem<E>
where
    E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
{
    pub(super) fn fit_degree(&self, degree: usize) -> Result<Fit<E>, FitError<E>> {
        if let Some(constraint) = self.constraint.as_ref() {
            return self.fit_degree_constrained(degree, constraint);
        }

        match &self.uncertainty {
            Uncertainty::None => self.fit_degree_ols(degree),

            Uncertainty::YDiagonal { uy } => self.fit_degree_wls_diagonal(degree, uy),

            Uncertainty::YCovariance { vy } => self.fit_degree_gls_covariance(degree, vy),

            Uncertainty::XYDiagonal { ux, uy } => self.fit_degree_odr_diagonal(degree, ux, uy),

            Uncertainty::XYCovariance { vx, vy } => self.fit_degree_odr_covariance(degree, vx, vy),
        }
    }

    pub(super) fn fit_degree_ols(&self, degree: usize) -> Result<Fit<E>, FitError<E>> {
        let report = ChebyshevSeries::fit_report_on_domain(
            self.x
                .as_slice()
                .expect("validated x array should be contiguous"),
            self.y
                .as_slice()
                .expect("validated y array should be contiguous"),
            degree,
            self.domain.clone(),
        )?;

        Ok(Fit::from_series_report(
            report,
            None,
            self.response_domain.clone(),
            FitMethod::OrdinaryLeastSquares,
        ))
    }

    pub(super) fn fit_degree_wls_diagonal(
        &self,
        degree: usize,
        uy: &Array1<E>,
    ) -> Result<Fit<E>, FitError<E>> {
        let weights = y_uncertainty_to_weights(uy)?;

        let report = ChebyshevSeries::fit_weighted_report_on_domain(
            self.x
                .as_slice()
                .expect("validated x array should be contiguous"),
            self.y
                .as_slice()
                .expect("validated y array should be contiguous"),
            weights.as_slice().expect("weights should be contiguous"),
            degree,
            self.domain.clone(),
        )?;

        Ok(Fit::from_series_report(
            report,
            None,
            self.response_domain.clone(),
            FitMethod::WeightedLeastSquares,
        ))
    }

    pub(super) fn fit_degree_gls_covariance(
        &self,
        degree: usize,
        vy: &Array2<E>,
    ) -> Result<Fit<E>, FitError<E>> {
        let x = self
            .x
            .as_slice()
            .expect("validated x array should be contiguous");

        let y = self
            .y
            .as_slice()
            .expect("validated y array should be contiguous");

        let design = ChebyshevSeries::design_matrix(x, degree, &self.domain);
        let rhs = Array1::from_vec(y.to_vec());

        // V = L Lᵀ
        let lower = vy.clone().cholesky(UPLO::Lower)?;

        // Minimise ||L⁻¹(Ac - y)||².
        let whitened_design =
            lower.solve_triangular_into(UPLO::Lower, ndarray_linalg::Diag::NonUnit, design)?;

        let whitened_rhs =
            lower.solve_triangular_into(UPLO::Lower, ndarray_linalg::Diag::NonUnit, rhs)?;

        let report = ChebyshevSeries::fit_report_from_design_on_domain(
            x,
            y,
            degree,
            self.domain.clone(),
            whitened_design,
            whitened_rhs,
        )?;

        Ok(Fit::from_series_report(
            report,
            None,
            self.response_domain.clone(),
            FitMethod::GeneralizedLeastSquares,
        ))
    }

    fn fit_degree_odr_diagonal(
        &self,
        degree: usize,
        _ux: &Array1<E>,
        _uy: &Array1<E>,
    ) -> Result<Fit<E>, FitError<E>> {
        Err(FitError::Unsupported {
            reason: "diagonal x/y uncertainty fitting is not implemented yet",
        })
    }

    fn fit_degree_odr_covariance(
        &self,
        degree: usize,
        _vx: &Array2<E>,
        _vy: &Array2<E>,
    ) -> Result<Fit<E>, FitError<E>> {
        Err(FitError::Unsupported {
            reason: "full x/y covariance fitting is not implemented yet",
        })
    }
}

fn y_uncertainty_to_weights<E>(uy: &Array1<E>) -> Result<Array1<E>, FitError<E>>
where
    E: Float,
{
    if uy.iter().any(|&u| !u.is_finite() || u <= E::zero()) {
        return Err(FitError::InvalidUncertainty);
    }

    Ok(uy.mapv(|u| E::one() / (u * u)))
}

#[cfg(test)]
mod fit_degree_tests {
    use super::*;
    use crate::score::ScoringStrategy;
    use crate::solve::GoodnessOfFit;

    use ndarray::{arr1, arr2};
    use poly_series::PolynomialSeries;

    const EPS: f64 = 1.0e-9;

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= EPS,
            "expected {lhs} ≈ {rhs}, difference = {}",
            (lhs - rhs).abs()
        );
    }

    fn assert_curve_matches_line(fit: &Fit<f64>) {
        for x in [0.0, 0.5, 1.0, 1.5, 2.0] {
            assert_close(fit.curve.evaluate(x), 1.0 + 2.0 * x);
        }
    }

    fn ols_problem() -> Problem<f64> {
        Problem {
            x: arr1(&[0.0, 0.5, 1.0, 1.5, 2.0]),
            y: arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]),
            uncertainty: Uncertainty::None,
            domain: 0.0..2.0,
            response_domain: 1.0..5.0,
            strategy: ScoringStrategy::ChiSquare,
            constraint: None,
            goodness_of_fit: GoodnessOfFit::Disabled,
        }
    }

    #[test]
    fn fit_degree_ols_fits_exact_linear_data() {
        let problem = ols_problem();

        let fit = problem.fit_degree_ols(1).unwrap();

        assert!(matches!(fit.method, FitMethod::OrdinaryLeastSquares));
        assert!(fit.constraint.is_none());
        assert_eq!(fit.response_domain, 1.0..5.0);
        assert_eq!(fit.curve.domain(), 0.0..2.0);
        assert_eq!(fit.free_polynomial.domain(), 0.0..2.0);

        assert_curve_matches_line(&fit);

        for residual in fit.residuals.iter() {
            assert_close(*residual, 0.0);
        }
    }

    #[test]
    fn fit_degree_ols_uses_problem_domain_not_inferred_data_domain() {
        let problem = Problem {
            x: arr1(&[0.5, 1.0, 1.5]),
            y: arr1(&[2.0, 3.0, 4.0]),
            uncertainty: Uncertainty::None,
            domain: 0.0..2.0,
            response_domain: 2.0..4.0,
            strategy: ScoringStrategy::ChiSquare,
            constraint: None,
            goodness_of_fit: GoodnessOfFit::Disabled,
        };

        let fit = problem.fit_degree_ols(1).unwrap();

        assert_eq!(fit.curve.domain(), 0.0..2.0);
        assert_curve_matches_line(&fit);
    }

    #[test]
    fn fit_degree_wls_diagonal_with_uniform_uncertainty_matches_ols() {
        let problem = Problem {
            uncertainty: Uncertainty::YDiagonal {
                uy: arr1(&[1.0, 1.0, 1.0, 1.0, 1.0]),
            },
            ..ols_problem()
        };

        let weighted = match &problem.uncertainty {
            Uncertainty::YDiagonal { uy } => problem.fit_degree_wls_diagonal(1, uy).unwrap(),
            _ => unreachable!(),
        };

        let ols = ols_problem().fit_degree_ols(1).unwrap();

        assert!(matches!(weighted.method, FitMethod::WeightedLeastSquares));

        for x in [0.0, 0.5, 1.0, 1.5, 2.0] {
            assert_close(weighted.curve.evaluate(x), ols.curve.evaluate(x));
        }
    }

    #[test]
    fn fit_degree_wls_diagonal_downweights_outlier() {
        let problem = Problem {
            x: arr1(&[0.0, 1.0, 2.0]),
            y: arr1(&[1.0, 3.0, 100.0]),
            uncertainty: Uncertainty::YDiagonal {
                uy: arr1(&[1.0, 1.0, 1.0e6]),
            },
            domain: 0.0..2.0,
            response_domain: 1.0..100.0,
            strategy: ScoringStrategy::ChiSquare,
            constraint: None,
            goodness_of_fit: GoodnessOfFit::Disabled,
        };

        let weighted = match &problem.uncertainty {
            Uncertainty::YDiagonal { uy } => problem.fit_degree_wls_diagonal(1, uy).unwrap(),
            _ => unreachable!(),
        };

        let unweighted_problem = Problem {
            uncertainty: Uncertainty::None,
            ..problem
        };
        let unweighted = unweighted_problem.fit_degree_ols(1).unwrap();

        let weighted_error_at_one = Float::abs(weighted.curve.evaluate(1.0) - 3.0);
        let unweighted_error_at_one = Float::abs(unweighted.curve.evaluate(1.0) - 3.0);

        assert!(weighted_error_at_one < unweighted_error_at_one);
        assert!(matches!(weighted.method, FitMethod::WeightedLeastSquares));
    }

    #[test]
    fn fit_degree_wls_diagonal_rejects_zero_uncertainty() {
        let problem = Problem {
            uncertainty: Uncertainty::YDiagonal {
                uy: arr1(&[1.0, 0.0, 1.0, 1.0, 1.0]),
            },
            ..ols_problem()
        };

        let err = match &problem.uncertainty {
            Uncertainty::YDiagonal { uy } => problem.fit_degree_wls_diagonal(1, uy).unwrap_err(),
            _ => unreachable!(),
        };

        assert!(matches!(err, FitError::InvalidUncertainty));
    }

    #[test]
    fn fit_degree_gls_covariance_with_diagonal_covariance_matches_wls() {
        let problem = ols_problem();

        let vy = arr2(&[
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 16.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 25.0],
        ]);

        let uy = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let gls = problem.fit_degree_gls_covariance(1, &vy).unwrap();
        let wls = problem.fit_degree_wls_diagonal(1, &uy).unwrap();

        assert!(matches!(gls.method, FitMethod::GeneralizedLeastSquares));

        for x in [0.0, 0.5, 1.0, 1.5, 2.0] {
            assert_close(gls.curve.evaluate(x), wls.curve.evaluate(x));
        }
    }

    #[test]
    fn fit_degree_gls_covariance_fits_exact_linear_data() {
        let problem = ols_problem();

        let vy = arr2(&[
            [2.0, 0.5, 0.0, 0.0, 0.0],
            [0.5, 2.0, 0.5, 0.0, 0.0],
            [0.0, 0.5, 2.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 2.0, 0.5],
            [0.0, 0.0, 0.0, 0.5, 2.0],
        ]);

        let fit = problem.fit_degree_gls_covariance(1, &vy).unwrap();

        assert!(matches!(fit.method, FitMethod::GeneralizedLeastSquares));
        assert_curve_matches_line(&fit);
    }

    #[test]
    fn fit_degree_gls_covariance_rejects_non_positive_definite_covariance() {
        let problem = ols_problem();

        let vy = arr2(&[
            [1.0, 2.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]);

        let err = problem.fit_degree_gls_covariance(1, &vy).unwrap_err();

        assert!(matches!(
            err,
            FitError::InvalidUncertainty | FitError::Linalg(_)
        ));
    }

    #[test]
    fn fit_degree_dispatch_uses_ols_for_no_uncertainty() {
        let problem = ols_problem();

        let fit = problem.fit_degree(1).unwrap();

        assert!(matches!(fit.method, FitMethod::OrdinaryLeastSquares));
        assert_curve_matches_line(&fit);
    }

    #[test]
    fn fit_degree_dispatch_uses_wls_for_y_diagonal_uncertainty() {
        let problem = Problem {
            uncertainty: Uncertainty::YDiagonal {
                uy: arr1(&[1.0, 1.0, 1.0, 1.0, 1.0]),
            },
            ..ols_problem()
        };

        let fit = problem.fit_degree(1).unwrap();

        assert!(matches!(fit.method, FitMethod::WeightedLeastSquares));
        assert_curve_matches_line(&fit);
    }

    #[test]
    fn fit_degree_dispatch_uses_gls_for_y_covariance() {
        let problem = Problem {
            uncertainty: Uncertainty::YCovariance {
                vy: arr2(&[
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ]),
            },
            ..ols_problem()
        };

        let fit = problem.fit_degree(1).unwrap();

        assert!(matches!(fit.method, FitMethod::GeneralizedLeastSquares));
        assert_curve_matches_line(&fit);
    }
}
