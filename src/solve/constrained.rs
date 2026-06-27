use crate::{
    Problem,
    fit::{Constraint, Fit, FitMethod},
    problem::Uncertainty,
};

use super::FitError;

use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, Lapack, Scalar, Solve, SolveTriangularInto, UPLO};
use num_traits::{Float, FromPrimitive};
use poly_series::{
    ChebyshevError, ChebyshevSeries, FitPolynomialSeries, PolynomialRoots, PolynomialSeries,
};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::ops::Range;

impl<E> Problem<E>
where
    E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
{
    pub(super) fn fit_degree_constrained(
        &self,
        degree: usize,
        constraint: &Constraint<E>,
    ) -> Result<Fit<E>, FitError<E>>
    where
        E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
    {
        match &self.uncertainty {
            Uncertainty::None => self.fit_degree_constrained_ols(degree, constraint),

            Uncertainty::YDiagonal { uy } => {
                self.fit_degree_constrained_wls_diagonal(degree, constraint, uy)
            }

            Uncertainty::YCovariance { vy } => {
                self.fit_degree_constrained_gls_covariance(degree, constraint, vy)
            }

            Uncertainty::XYDiagonal { ux, uy } => {
                self.fit_degree_constrained_odr_diagonal(degree, constraint, ux, uy)
            }

            Uncertainty::XYCovariance { vx, vy } => {
                self.fit_degree_constrained_odr_covariance(degree, constraint, vx, vy)
            }
        }
    }

    fn fit_degree_constrained_ols(
        &self,
        degree: usize,
        constraint: &Constraint<E>,
    ) -> Result<Fit<E>, FitError<E>>
    where
        E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
    {
        let xs = self
            .x
            .as_slice()
            .expect("validated x array should be contiguous");

        let design = constrained_design_matrix(xs, degree, &self.domain, constraint);
        let rhs = constrained_response(&self.x, &self.y, constraint);

        self.fit_from_design(
            degree,
            design,
            rhs,
            Some(constraint.clone()),
            FitMethod::OrdinaryLeastSquares,
        )
    }

    fn fit_degree_constrained_wls_diagonal(
        &self,
        degree: usize,
        constraint: &Constraint<E>,
        uy: &Array1<E>,
    ) -> Result<Fit<E>, FitError<E>>
    where
        E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
    {
        let xs = self
            .x
            .as_slice()
            .expect("validated x array should be contiguous");

        let mut design = constrained_design_matrix(xs, degree, &self.domain, constraint);
        let mut rhs = constrained_response(&self.x, &self.y, constraint);

        if uy.iter().any(|&u| !u.is_finite() || u <= E::zero()) {
            return Err(FitError::InvalidUncertainty);
        }

        for row in 0..self.x.len() {
            let scale = E::one() / uy[row];

            for col in 0..=degree {
                design[[row, col]] = design[[row, col]] * scale;
            }

            rhs[row] = rhs[row] * scale;
        }

        self.fit_from_design(
            degree,
            design,
            rhs,
            Some(constraint.clone()),
            FitMethod::WeightedLeastSquares,
        )
    }

    fn fit_degree_constrained_gls_covariance(
        &self,
        degree: usize,
        constraint: &Constraint<E>,
        vy: &Array2<E>,
    ) -> Result<Fit<E>, FitError<E>>
    where
        E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
    {
        use ndarray_linalg::{Cholesky, Diag, SolveTriangularInto, UPLO};

        let xs = self
            .x
            .as_slice()
            .expect("validated x array should be contiguous");

        let design = constrained_design_matrix(xs, degree, &self.domain, constraint);
        let rhs = constrained_response(&self.x, &self.y, constraint);

        let lower = vy.clone().cholesky(UPLO::Lower)?;

        let whitened_design = lower.solve_triangular_into(UPLO::Lower, Diag::NonUnit, design)?;

        let whitened_rhs = lower.solve_triangular_into(UPLO::Lower, Diag::NonUnit, rhs)?;

        self.fit_from_design(
            degree,
            whitened_design,
            whitened_rhs,
            Some(constraint.clone()),
            FitMethod::GeneralizedLeastSquares,
        )
    }

    fn fit_degree_constrained_odr_diagonal(
        &self,
        degree: usize,
        _constraint: &Constraint<E>,
        _ux: &Array1<E>,
        _uy: &Array1<E>,
    ) -> Result<Fit<E>, FitError<E>> {
        Err(FitError::Unsupported {
            reason: "constrained diagonal x/y uncertainty fitting is not implemented yet",
        })
    }

    fn fit_degree_constrained_odr_covariance(
        &self,
        degree: usize,
        _constraint: &Constraint<E>,
        _vx: &Array2<E>,
        _vy: &Array2<E>,
    ) -> Result<Fit<E>, FitError<E>> {
        Err(FitError::Unsupported {
            reason: "constrained full x/y covariance fitting is not implemented yet",
        })
    }

    fn fit_from_design(
        &self,
        degree: usize,
        design: Array2<E>,
        rhs: Array1<E>,
        constraint: Option<Constraint<E>>,
        method: FitMethod,
    ) -> Result<Fit<E>, FitError<E>>
    where
        E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
    {
        let xs = self
            .x
            .as_slice()
            .expect("validated x array should be contiguous");

        let ys = self
            .y
            .as_slice()
            .expect("validated y array should be contiguous");

        let report = ChebyshevSeries::fit_report_from_design_on_domain(
            xs,
            ys,
            degree,
            self.domain.clone(),
            design,
            rhs,
        )?;

        Ok(Fit::from_series_report(
            report,
            constraint,
            self.response_domain.clone(),
            method,
        ))
    }
}

fn constrained_response<E>(x: &Array1<E>, y: &Array1<E>, constraint: &Constraint<E>) -> Array1<E>
where
    E: Float + FromPrimitive,
{
    y.iter()
        .zip(x.iter())
        .map(|(&y, &x)| y - constraint.additive.evaluate(x))
        .collect()
}

fn constrained_design_matrix<E>(
    xs: &[E],
    degree: usize,
    domain: &Range<E>,
    constraint: &Constraint<E>,
) -> Array2<E>
where
    E: Float + FromPrimitive,
{
    let mut matrix = Array2::<E>::zeros((xs.len(), degree + 1));

    for (row, &x) in xs.iter().enumerate() {
        let t = poly_series::scaling::to_scaled(x, domain);
        let multiplier = constraint.multiplicative.evaluate(x);

        let mut t0 = E::one();
        matrix[[row, 0]] = multiplier * t0;

        if degree == 0 {
            continue;
        }

        let mut t1 = t;
        matrix[[row, 1]] = multiplier * t1;

        let two = E::one() + E::one();

        for col in 2..=degree {
            let next = two * t * t1 - t0;
            matrix[[row, col]] = multiplier * next;
            t0 = t1;
            t1 = next;
        }
    }

    matrix
}

#[cfg(test)]
mod constrained_fit_tests {
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

    fn base_problem(uncertainty: Uncertainty<f64>) -> Problem<f64> {
        Problem {
            x: arr1(&[-1.0, -0.5, 0.0, 0.5, 1.0]),
            y: arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]),
            uncertainty,
            domain: -1.0..1.0,
            response_domain: -2.0..2.0,
            strategy: ScoringStrategy::ChiSquare,
            constraint: None,
            goodness_of_fit: GoodnessOfFit::Disabled,
        }
    }

    fn origin_constraint() -> Constraint<f64> {
        Constraint {
            additive: ChebyshevSeries::new(vec![0.0], -1.0..1.0).unwrap(),
            multiplicative: ChebyshevSeries::new(vec![0.0, 1.0], -1.0..1.0).unwrap(),
        }
    }

    fn assert_fits_y_equals_two_x(fit: &Fit<f64>) {
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert_close(fit.curve.evaluate(x), 2.0 * x);
        }
    }

    #[test]
    fn constrained_response_subtracts_additive_component() {
        let constraint = Constraint {
            additive: ChebyshevSeries::new(vec![1.0], -1.0..1.0).unwrap(),
            multiplicative: ChebyshevSeries::new(vec![1.0], -1.0..1.0).unwrap(),
        };

        let x = arr1(&[-1.0, 0.0, 1.0]);
        let y = arr1(&[2.0, 3.0, 4.0]);

        let shifted = constrained_response(&x, &y, &constraint);

        assert_eq!(shifted, arr1(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn constrained_design_matrix_multiplies_basis_by_multiplicative_constraint() {
        let constraint = origin_constraint();

        let design = constrained_design_matrix(&[-1.0, 0.0, 1.0], 1, &(-1.0..1.0), &constraint);

        // multiplicative = t = x on canonical domain.
        //
        // columns are:
        //   t * T0 = t
        //   t * T1 = t²
        assert_eq!(design.shape(), &[3, 2]);

        assert_close(design[[0, 0]], -1.0);
        assert_close(design[[0, 1]], 1.0);

        assert_close(design[[1, 0]], 0.0);
        assert_close(design[[1, 1]], 0.0);

        assert_close(design[[2, 0]], 1.0);
        assert_close(design[[2, 1]], 1.0);
    }

    #[test]
    fn fit_degree_constrained_ols_fits_exact_origin_constrained_line() {
        let constraint = origin_constraint();
        let problem = base_problem(Uncertainty::None);

        let fit = problem.fit_degree_constrained_ols(0, &constraint).unwrap();

        assert!(matches!(fit.method, FitMethod::OrdinaryLeastSquares));
        assert!(fit.constraint.is_some());
        assert_fits_y_equals_two_x(&fit);

        // Free polynomial should be q(x) = 2, because y = x * q(x).
        assert_close(fit.free_polynomial.evaluate(-1.0), 2.0);
        assert_close(fit.free_polynomial.evaluate(0.0), 2.0);
        assert_close(fit.free_polynomial.evaluate(1.0), 2.0);
    }

    #[test]
    fn fit_degree_constrained_wls_diagonal_with_uniform_uncertainty_matches_constrained_ols() {
        let constraint = origin_constraint();
        let problem = base_problem(Uncertainty::YDiagonal {
            uy: arr1(&[1.0, 1.0, 1.0, 1.0, 1.0]),
        });

        let wls = match &problem.uncertainty {
            Uncertainty::YDiagonal { uy } => problem
                .fit_degree_constrained_wls_diagonal(0, &constraint, uy)
                .unwrap(),
            _ => unreachable!(),
        };

        let ols = base_problem(Uncertainty::None)
            .fit_degree_constrained_ols(0, &constraint)
            .unwrap();

        assert!(matches!(wls.method, FitMethod::WeightedLeastSquares));

        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert_close(wls.curve.evaluate(x), ols.curve.evaluate(x));
        }
    }

    #[test]
    fn fit_degree_constrained_wls_diagonal_downweights_outlier() {
        let constraint = origin_constraint();

        let problem = Problem {
            x: arr1(&[-1.0, 0.0, 1.0]),
            y: arr1(&[-2.0, 0.0, 100.0]),
            uncertainty: Uncertainty::YDiagonal {
                uy: arr1(&[1.0, 1.0, 1.0e6]),
            },
            domain: -1.0..1.0,
            response_domain: -2.0..100.0,
            strategy: ScoringStrategy::ChiSquare,
            constraint: None,
            goodness_of_fit: GoodnessOfFit::Disabled,
        };

        let weighted = match &problem.uncertainty {
            Uncertainty::YDiagonal { uy } => problem
                .fit_degree_constrained_wls_diagonal(0, &constraint, uy)
                .unwrap(),
            _ => unreachable!(),
        };

        let unweighted = Problem {
            uncertainty: Uncertainty::None,
            ..problem
        }
        .fit_degree_constrained_ols(0, &constraint)
        .unwrap();

        let weighted_error = (weighted.curve.evaluate(-1.0) - (-2.0)).abs();
        let unweighted_error = (unweighted.curve.evaluate(-1.0) - (-2.0)).abs();

        assert!(weighted_error < unweighted_error);
        assert!(matches!(weighted.method, FitMethod::WeightedLeastSquares));
    }

    #[test]
    fn fit_degree_constrained_wls_diagonal_rejects_zero_uncertainty() {
        let constraint = origin_constraint();
        let problem = base_problem(Uncertainty::YDiagonal {
            uy: arr1(&[1.0, 1.0, 0.0, 1.0, 1.0]),
        });

        let err = match &problem.uncertainty {
            Uncertainty::YDiagonal { uy } => problem
                .fit_degree_constrained_wls_diagonal(0, &constraint, uy)
                .unwrap_err(),
            _ => unreachable!(),
        };

        assert!(matches!(err, FitError::InvalidUncertainty));
    }

    #[test]
    fn fit_degree_constrained_gls_covariance_with_diagonal_covariance_matches_wls() {
        let constraint = origin_constraint();
        let problem = base_problem(Uncertainty::None);

        let vy = arr2(&[
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 16.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 25.0],
        ]);

        let uy = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let gls = problem
            .fit_degree_constrained_gls_covariance(0, &constraint, &vy)
            .unwrap();

        let wls = problem
            .fit_degree_constrained_wls_diagonal(0, &constraint, &uy)
            .unwrap();

        assert!(matches!(gls.method, FitMethod::GeneralizedLeastSquares));

        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert_close(gls.curve.evaluate(x), wls.curve.evaluate(x));
        }
    }

    #[test]
    fn fit_degree_constrained_gls_covariance_fits_exact_origin_constrained_line() {
        let constraint = origin_constraint();
        let problem = base_problem(Uncertainty::None);

        let vy = arr2(&[
            [2.0, 0.5, 0.0, 0.0, 0.0],
            [0.5, 2.0, 0.5, 0.0, 0.0],
            [0.0, 0.5, 2.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 2.0, 0.5],
            [0.0, 0.0, 0.0, 0.5, 2.0],
        ]);

        let fit = problem
            .fit_degree_constrained_gls_covariance(0, &constraint, &vy)
            .unwrap();

        assert!(matches!(fit.method, FitMethod::GeneralizedLeastSquares));
        assert_fits_y_equals_two_x(&fit);
    }

    #[test]
    fn fit_degree_constrained_dispatch_uses_ols_for_no_uncertainty() {
        let constraint = origin_constraint();

        let problem = Problem {
            constraint: Some(constraint),
            ..base_problem(Uncertainty::None)
        };

        let fit = problem.fit_degree(0).unwrap();

        assert!(matches!(fit.method, FitMethod::OrdinaryLeastSquares));
        assert!(fit.constraint.is_some());
        assert_fits_y_equals_two_x(&fit);
    }

    #[test]
    fn fit_degree_constrained_dispatch_uses_wls_for_y_diagonal_uncertainty() {
        let constraint = origin_constraint();

        let problem = Problem {
            constraint: Some(constraint),
            ..base_problem(Uncertainty::YDiagonal {
                uy: arr1(&[1.0, 1.0, 1.0, 1.0, 1.0]),
            })
        };

        let fit = problem.fit_degree(0).unwrap();

        assert!(matches!(fit.method, FitMethod::WeightedLeastSquares));
        assert!(fit.constraint.is_some());
        assert_fits_y_equals_two_x(&fit);
    }

    #[test]
    fn fit_degree_constrained_dispatch_uses_gls_for_y_covariance() {
        let constraint = origin_constraint();

        let problem = Problem {
            constraint: Some(constraint),
            ..base_problem(Uncertainty::YCovariance {
                vy: arr2(&[
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ]),
            })
        };

        let fit = problem.fit_degree(0).unwrap();

        assert!(matches!(fit.method, FitMethod::GeneralizedLeastSquares));
        assert!(fit.constraint.is_some());
        assert_fits_y_equals_two_x(&fit);
    }
}
