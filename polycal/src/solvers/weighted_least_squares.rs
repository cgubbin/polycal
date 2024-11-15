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

impl<E: Lapack + Scalar<Real = E> + ScalarOperand> SolveSystem<E> for WeightedLeastSquares<'_, E> {
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
        E: Float + PartialOrd + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore,
    {
        let builder = ProblemBuilder::new(x, y).unwrap();
        let problem = match &uncertainty {
            Uncertainty::None => builder.build(),
            Uncertainty::Diagonal(uy) => builder.with_dependent_variance(*uy).unwrap().build(),
            Uncertainty::Full(vy) => builder.with_dependent_covariance(*vy).build(),
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
        E: Float + Scalar<Real = E> + ScalarOperand + PartialOrd + Lapack + FloatCore,
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

    #[test]
    fn sigmoid_fit_scaling() {
        use ndarray::arr1;
        let stimulus_values = arr1(&vec![
            0.0,
            0.03483562587663291,
            0.06967125175326581,
            0.10450687762989871,
            0.13934250350653163,
            0.17417812938316454,
            0.20901375525979743,
            0.24384938113643034,
            0.27868500701306326,
            0.31352063288969617,
            0.3483562587663291,
            0.383191884642962,
            0.41802751051959486,
            0.4528631363962278,
            0.4876987622728607,
            0.5225343881494936,
            0.5573700140261265,
            0.5922056399027594,
            0.6270412657793923,
            0.6618768916560253,
            0.6967125175326582,
            0.7315481434092911,
            0.766383769285924,
            0.8012193951625569,
            0.8360550210391897,
            0.8708906469158226,
            0.9057262727924555,
            0.9405618986690885,
            0.9753975245457214,
            1.0102331504223543,
            1.0450687762989872,
            1.0799044021756201,
            1.114740028052253,
            1.149575653928886,
            1.1844112798055189,
            1.2192469056821518,
            1.2540825315587847,
            1.2889181574354176,
            1.3237537833120505,
            1.3585894091886834,
            1.3934250350653163,
            1.4282606609419493,
            1.4630962868185822,
            1.497931912695215,
            1.532767538571848,
            1.567603164448481,
            1.6024387903251138,
            1.6372744162017467,
            1.6721100420783794,
            1.7069456679550123,
            1.7417812938316453,
            1.7766169197082782,
            1.811452545584911,
            1.846288171461544,
            1.881123797338177,
            1.9159594232148098,
            1.9507950490914427,
            1.9856306749680757,
            2.0204663008447086,
            2.0553019267213415,
            2.0901375525979744,
            2.1249731784746073,
            2.1598088043512402,
            2.194644430227873,
            2.229480056104506,
            2.264315681981139,
            2.299151307857772,
            2.333986933734405,
            2.3688225596110377,
            2.4036581854876706,
            2.4384938113643035,
            2.4733294372409365,
            2.5081650631175694,
            2.5430006889942023,
            2.577836314870835,
            2.612671940747468,
            2.647507566624101,
        ]);
        let response_values = arr1(&vec![
            0.8192248406510491,
            0.8482550112718827,
            0.8783134980136609,
            0.9094366972848017,
            0.9416622916276787,
            0.9750292950159627,
            1.0095780997364716,
            1.0453505249101418,
            1.0823898667086127,
            1.120740950324751,
            1.1604501837574333,
            1.2015656134728523,
            1.24413698200674,
            1.2882157875739544,
            1.3338553457541453,
            1.3811108533243914,
            1.4300394543121016,
            1.4807003083437922,
            1.5331546613678868,
            1.587465918832144,
            1.6436997213990072,
            1.7019240232847628,
            1.7622091733112482,
            1.824627998761584,
            1.889255892134419,
            1.95617090089407,
            2.025453820317112,
            2.0971882895390324,
            2.1714608909078885,
            2.24836125275516,
            2.327982155697452,
            2.4104196425861355,
            2.4957731322256747,
            2.5841455369849404,
            2.6756433844296947,
            2.7703769431081215,
            2.868460352625363,
            2.9700117581468466,
            3.075153449474484,
            3.1840120048437948,
            3.296718439594464,
            3.413408359871003,
            3.5342221215147798,
            3.659304994313004,
            3.7888073317750104,
            3.922884746610629,
            4.061698292090357,
            4.205414649471582,
            4.354206321680171,
            4.508251833441441,
            4.667735938059521,
            4.832849831049133,
            5.0037913708287185,
            5.180765306688915,
            5.363983514255387,
            5.553665238670095,
            5.750037345720057,
            5.953334581147634,
            6.163799838381394,
            6.381684434931474,
            6.6072483976981955,
            6.840760757447379,
            7.0824998527105,
            7.332753643372316,
            7.591820034212885,
            7.860007208675012,
            8.1376339731322,
            8.425030111935792,
            8.722536753523427,
            9.030506747873936,
            9.349305055596675,
            9.679309148945547,
            10.02090942504986,
            10.374509631655531,
            10.740527305671037,
            11.11939422481278,
            11.511556872644064,
        ]);

        let variance = ndarray::Array1::from_elem(stimulus_values.len(), 1e-2);

        let problem = ProblemBuilder::new(stimulus_values.view(), response_values.view())
            .unwrap()
            .with_dependent_variance(variance.view())
            .unwrap()
            // .with_constraint(constraint)
            .build();

        let solution = problem.solve(10).unwrap();

        dbg!(
            &solution.response(0.0.into()),
            solution.coeff(),
            solution.variance().diag()
        );

        let response_values = response_values.mapv(|each| each / 1000.0);

        let problem = ProblemBuilder::new(stimulus_values.view(), response_values.view())
            .unwrap()
            .with_dependent_variance(variance.view())
            .unwrap()
            // .with_constraint(constraint)
            .build();

        let solution = problem.solve(10).unwrap();

        dbg!(
            &solution.response(0.0.into()),
            solution.coeff(),
            solution.variance().diag()
        );
    }
}
