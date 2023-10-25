use super::SolverError;
use super::{Solution, SolveSystem, Uncertainty};
use crate::utils::outer_product;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis, ScalarOperand};
use ndarray_linalg::{Cholesky, Inverse, Lapack, LeastSquaresSvd, Scalar, UPLO};

pub struct WeightedLeastSquares<'a, E> {
    pub(crate) y: Array1<E>,
    pub(crate) uncertainty: Uncertainty<'a, E>,
    pub(crate) h: Array2<E>,
}

impl<'a, E: Lapack + Scalar<Real = E> + ScalarOperand> SolveSystem<E>
    for WeightedLeastSquares<'a, E>
{
    fn solve(&self) -> ::std::result::Result<Solution<E>, SolverError> {
        match self.uncertainty {
            Uncertainty::None => self.solve_unweighted(),
            Uncertainty::Diagonal(uy) => self.solve_weighted(uy),
            Uncertainty::Full(vy) => self.solve_full(vy),
        }
    }
}

impl<'a, E: Lapack + Scalar<Real = E> + ScalarOperand> WeightedLeastSquares<'a, E> {
    #[tracing::instrument(skip_all)]
    fn solve_unweighted(&self) -> ::std::result::Result<Solution<E>, SolverError> {
        let mut lhs = self.h.to_owned();
        let rhs = self.y.to_owned();
        let scaling = lhs
            .mapv(|val| val.powi(2))
            .sum_axis(Axis(0))
            .mapv(ndarray_linalg::Scalar::sqrt);

        lhs /= &scaling;

        let result = lhs.least_squares(&rhs).map_err(SolverError::LeastSquares)?;

        let coeff = (&result.solution.t() / &scaling).t().to_owned();

        let covariance = (lhs.t().dot(&lhs)).inv().map_err(SolverError::Inverse)?
            / outer_product(&scaling, &scaling).unwrap(); // This method is tested, and can
                                                          // reasonably be expected to be infallible.

        Ok(Solution {
            coeff,
            dependent_central_values: None,
            covariance,
        })
    }

    #[tracing::instrument(skip_all)]
    fn solve_weighted(
        &self,
        uy: ArrayView1<'a, E>,
    ) -> ::std::result::Result<Solution<E>, SolverError> {
        let mut lhs = self.h.to_owned();
        let uy = uy.to_owned();

        let rhs = self.y.to_owned() / uy.mapv(|x| x.powi(2));

        for (ii, uy) in uy.iter().enumerate() {
            let mut slice = lhs.slice_mut(s![ii, ..]);
            slice /= uy.powi(2);
        }

        let scaling = lhs
            .mapv(|val| val.powi(2))
            .sum_axis(Axis(0))
            .mapv(ndarray_linalg::Scalar::sqrt);

        lhs /= &scaling;

        let result = lhs.least_squares(&rhs).map_err(SolverError::LeastSquares)?;

        let coeff = (&result.solution.t() / &scaling).t().to_owned();

        let lhs = self.h.to_owned();
        let w = Array2::from_diag(&uy.to_owned().mapv(|uy| E::one() / uy.powi(2)));
        let covariance = (lhs.t().dot(&w.dot(&lhs)))
            .inv()
            .map_err(SolverError::Inverse)?;

        // let covariance = (lhs.t().dot(&lhs)).inv()? / outer_product(&scaling, &scaling)?;

        Ok(Solution {
            coeff,
            dependent_central_values: None,
            covariance,
        })
    }

    #[tracing::instrument(skip_all)]
    fn solve_full(&self, vy: ArrayView2<'a, E>) -> ::std::result::Result<Solution<E>, SolverError> {
        let _lhs = self.h.to_owned();
        let vy = vy.to_owned();

        let _lower = vy.cholesky(UPLO::Lower).map_err(SolverError::Cholesky)?;

        unimplemented!("no impl for full-rank WLS for now.");
    }
}

#[cfg(test)]
mod test {
    use ndarray::{Array1, ArrayView1, ScalarOperand};
    use ndarray_linalg::{Lapack, Scalar};
    use ndarray_rand::{
        rand::{Rng, SeedableRng},
        rand_distr::{Distribution, Normal, Standard, StandardNormal},
    };
    use num_traits::{float::FloatCore, Float};
    use rand_isaac::Isaac64Rng;
    use std::ops::Range;

    use super::WeightedLeastSquares;
    use crate::builder::ProblemBuilder;
    use crate::chebyshev::{Basis, PolynomialSeries, Series};
    use crate::solvers::{SolveSystem, Uncertainty};

    impl<E: Scalar<Real = E> + PartialOrd> Series<E> {
        pub(crate) fn from_coeff(coeff: Vec<E>, x: &[E]) -> Self {
            let domain = crate::utils::find_limits(x);
            let degree = coeff.len() - 1;
            Self {
                coeff: coeff.into(),
                domain,
                window: Range {
                    start: -E::one(),
                    end: E::one(),
                },
                basis: Basis::new(degree),
            }
        }
    }

    pub fn generate_data<E>(
        rng: &mut impl Rng,
        Range { start, end }: Range<E>,
        num_points: usize,
        degree: usize,
    ) -> (Array1<E>, Array1<E>, Series<E>)
    where
        E: Scalar<Real = E> + ScalarOperand + PartialOrd + Lapack + FloatCore,
        Standard: Distribution<E>,
    {
        let chebyshev_coeffs = (0..=degree).map(|_| rng.gen()).collect::<Vec<_>>();

        let x = (0..num_points)
            .map(|m| {
                start
                    + E::from(m).unwrap() * (end - start)
                        / (E::from(num_points).unwrap() - E::one())
            })
            .collect::<Array1<_>>();

        let series = Series::from_coeff(chebyshev_coeffs, x.as_slice().unwrap());

        let y = x.iter().map(|x| series.evaluate(*x)).collect::<Array1<E>>();

        (x, y, series)
    }

    fn wls<'a, E>(
        x: ArrayView1<'a, E>,
        y: ArrayView1<'a, E>,
        uncertainty: &Uncertainty<'a, E>,
        degree: usize,
    ) -> WeightedLeastSquares<'a, E>
    where
        E: PartialOrd + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore,
    {
        let builder = ProblemBuilder::new(x, y);
        let problem = match &uncertainty {
            Uncertainty::None => builder.build(),
            Uncertainty::Diagonal(uy) => builder.with_independent_uncertainty(*uy).build(),
            Uncertainty::Full(vy) => builder.with_independent_covariance(*vy).build(),
        };

        let h = problem.design_matrix(degree).unwrap();

        WeightedLeastSquares {
            y: y.to_owned(),
            h,
            uncertainty: uncertainty.clone(),
        }
    }

    fn generate_test_data<E>(degree: usize) -> (Vec<E>, Vec<E>)
    where
        E: Scalar<Real = E> + ScalarOperand + PartialOrd + Lapack + FloatCore,
        Standard: Distribution<E>,
    {
        let state = 42;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let number_of_data_points = rng.gen_range(50..100);

        let domain = Range {
            start: -E::one(),
            end: E::one(),
        };
        let (x, y, series) = generate_data(&mut rng, domain, number_of_data_points, degree);

        let wls: WeightedLeastSquares<'_, E> = wls(x.view(), y.view(), &Uncertainty::None, degree);

        let result = wls.solve().unwrap();

        (series.coeff(), result.coeff().to_vec())
    }

    fn generate_diagonal_test_data<E>(
        degree: usize,
        max_noise_fraction: E,
    ) -> (Vec<E>, Vec<E>, Vec<E>)
    where
        E: Scalar<Real = E> + ScalarOperand + PartialOrd + Lapack + FloatCore + Float,
        Standard: Distribution<E>,
        StandardNormal: Distribution<E>,
    {
        let state = 42;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let number_of_data_points = rng.gen_range(50..100);

        let domain = Range {
            start: -E::one(),
            end: E::one(),
        };
        let (x, y_central, series) = generate_data(&mut rng, domain, number_of_data_points, degree);

        let standard_deviation = y_central
            .iter()
            .map(|_| max_noise_fraction)
            .collect::<Array1<E>>();

        let y = y_central
            .iter()
            .zip(standard_deviation.iter())
            .map(|(y, standard_deviation)| {
                let noise_dist = Normal::new(*y, *standard_deviation).unwrap();
                noise_dist.sample(&mut rng)
            })
            .collect::<Array1<E>>();

        let wls: WeightedLeastSquares<'_, E> = wls(
            x.view(),
            y.view(),
            &Uncertainty::Diagonal(standard_deviation.view()),
            degree,
        );

        let result = wls.solve().unwrap();

        (
            series.coeff(),
            result.coeff().to_vec(),
            result.covariance.diag().to_vec(),
        )
    }

    #[test]
    fn weighted_least_squares_works_for_linear_fit_with_no_uncertainties() {
        let degree = 1;
        let (expected, calculated) = generate_test_data::<f64>(degree);
        for (exp, cal) in expected.into_iter().zip(calculated) {
            approx::assert_relative_eq!(exp, cal, max_relative = 1e-7);
        }
    }

    #[test]
    fn weighted_least_squares_works_for_degree_two_fit_with_no_uncertainties() {
        let degree = 2;
        let (expected, calculated) = generate_test_data::<f64>(degree);
        for (exp, cal) in expected.into_iter().zip(calculated) {
            approx::assert_relative_eq!(exp, cal, max_relative = 1e-7);
        }
    }

    #[test]
    fn weighted_least_squares_works_for_degree_three_fit_with_no_uncertainties() {
        let degree = 3;
        let (expected, calculated) = generate_test_data::<f64>(degree);
        for (exp, cal) in expected.into_iter().zip(calculated) {
            approx::assert_relative_eq!(exp, cal, max_relative = 1e-7);
        }
    }

    #[test]
    fn weighted_least_squares_works_for_degree_four_fit_with_no_uncertainties() {
        let degree = 4;
        let (expected, calculated) = generate_test_data::<f64>(degree);
        for (exp, cal) in expected.into_iter().zip(calculated) {
            approx::assert_relative_eq!(exp, cal, max_relative = 1e-7);
        }
    }

    #[test]
    fn weighted_least_squares_works_for_degree_five_fit_with_no_uncertainties() {
        let degree = 5;
        let (expected, calculated) = generate_test_data::<f64>(degree);
        for (exp, cal) in expected.into_iter().zip(calculated) {
            approx::assert_relative_eq!(exp, cal, max_relative = 1e-7);
        }
    }

    #[test]
    fn weighted_least_squares_works_for_degree_six_fit_with_no_uncertainties() {
        let degree = 6;
        let (expected, calculated) = generate_test_data::<f64>(degree);
        for (exp, cal) in expected.into_iter().zip(calculated) {
            approx::assert_relative_eq!(exp, cal, max_relative = 1e-7);
        }
    }

    // Test can fail, we check fits are within 3 standard deviations of the true value but although
    // this is highly likely it is not certain. If these tests fail check how close the values are
    // and adjust.
    #[test]
    fn weighted_least_squares_works_for_linear_fit_with_diagonal_uncertainties() {
        let degree = 1;
        let max_noise_sd_fractional = 0.01;
        let (expected, calculated, variance) =
            generate_diagonal_test_data::<f64>(degree, max_noise_sd_fractional);
        for ((exp, cal), var) in expected.into_iter().zip(calculated).zip(variance) {
            assert!((exp - cal).abs() < 3. * var.sqrt());
        }
    }

    #[test]
    fn weighted_least_squares_works_for_degree_two_fit_with_diagonal_uncertainties() {
        let degree = 2;
        let max_noise_sd_fractional = 0.01;
        let (expected, calculated, variance) =
            generate_diagonal_test_data::<f64>(degree, max_noise_sd_fractional);
        for ((exp, cal), var) in expected.into_iter().zip(calculated).zip(variance) {
            assert!((exp - cal).abs() < 3. * var.sqrt());
        }
    }

    #[test]
    fn weighted_least_squares_works_for_degree_three_fit_with_diagonal_uncertainties() {
        let degree = 3;
        let max_noise_sd_fractional = 0.01;
        let (expected, calculated, variance) =
            generate_diagonal_test_data::<f64>(degree, max_noise_sd_fractional);
        for ((exp, cal), var) in expected.into_iter().zip(calculated).zip(variance) {
            assert!((exp - cal).abs() < 3. * var.sqrt());
        }
    }

    #[test]
    fn weighted_least_squares_works_for_degree_four_fit_with_diagonal_uncertainties() {
        let degree = 4;
        let max_noise_sd_fractional = 0.01;
        let (expected, calculated, variance) =
            generate_diagonal_test_data::<f64>(degree, max_noise_sd_fractional);
        for ((exp, cal), var) in expected.into_iter().zip(calculated).zip(variance) {
            assert!((exp - cal).abs() < 3. * var.sqrt());
        }
    }

    #[test]
    fn weighted_least_squares_works_for_degree_five_fit_with_diagonal_uncertainties() {
        let degree = 5;
        let max_noise_sd_fractional = 0.01;
        let (expected, calculated, variance) =
            generate_diagonal_test_data::<f64>(degree, max_noise_sd_fractional);
        for ((exp, cal), var) in expected.into_iter().zip(calculated).zip(variance) {
            assert!((exp - cal).abs() < 3. * var.sqrt());
        }
    }

    #[test]
    fn weighted_least_squares_works_for_degree_six_fit_with_diagonal_uncertainties() {
        let degree = 6;
        let max_noise_sd_fractional = 0.01;
        let (expected, calculated, variance) =
            generate_diagonal_test_data::<f64>(degree, max_noise_sd_fractional);
        for ((exp, cal), var) in expected.into_iter().zip(calculated).zip(variance) {
            assert!((exp - cal).abs() < 3. * var.sqrt());
        }
    }
}
