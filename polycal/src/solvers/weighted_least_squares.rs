use super::SolverError;
use super::{Covariance, Solution, SolveSystem};
use crate::utils::outer_product;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis, ScalarOperand};
use ndarray_linalg::{Cholesky, Inverse, Lapack, LeastSquaresSvd, Scalar, UPLO};

pub struct WeightedLeastSquares<'a, E> {
    pub(crate) y: Array1<E>,
    pub(crate) covariance: Covariance<'a, E>,
    pub(crate) h: Array2<E>,
}

impl<E: Lapack + Scalar<Real = E> + ScalarOperand> SolveSystem<E> for WeightedLeastSquares<'_, E> {
    fn solve(&self) -> ::std::result::Result<Solution<E>, SolverError> {
        match self.covariance {
            Covariance::None => self.solve_unweighted(),
            Covariance::Diagonal(uy) => self.solve_weighted(uy),
            Covariance::Matrix(vy) => self.solve_full(vy),
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
        dependent_variance: ArrayView1<'a, E>,
    ) -> ::std::result::Result<Solution<E>, SolverError> {
        let mut lhs = self.h.to_owned();
        // let vy = dependent_variance.to_owned();

        let rhs = self.y.to_owned() / dependent_variance;

        for (ii, each_variance) in dependent_variance.iter().enumerate() {
            let mut slice = lhs.slice_mut(s![ii, ..]);
            slice /= *each_variance;
        }

        let scaling = lhs
            .mapv(|val| val.powi(2))
            .sum_axis(Axis(0))
            .mapv(ndarray_linalg::Scalar::sqrt);

        lhs /= &scaling;

        let result = lhs.least_squares(&rhs).map_err(SolverError::LeastSquares)?;

        let coeff = (&result.solution.t() / &scaling).t().to_owned();

        let lhs = self.h.to_owned();
        let w = Array2::from_diag(&dependent_variance.to_owned().mapv(|each| E::one() / each));
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
    fn solve_full(
        &self,
        dependent_covariance: ArrayView2<'a, E>,
    ) -> ::std::result::Result<Solution<E>, SolverError> {
        let _lhs = self.h.to_owned();

        let _lower = dependent_covariance
            .cholesky(UPLO::Lower)
            .map_err(SolverError::Cholesky)?;

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
    use crate::solvers::{Covariance, SolveSystem};

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
        covariance: &Covariance<'a, E>,
        degree: usize,
    ) -> WeightedLeastSquares<'a, E>
    where
        E: Float + PartialOrd + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore,
    {
        let builder = ProblemBuilder::new(x, y).unwrap();
        let problem = match &covariance {
            Covariance::None => builder.build(),
            Covariance::Diagonal(uy) => builder.with_dependent_variance(*uy).unwrap().build(),
            Covariance::Matrix(vy) => builder.with_dependent_covariance(*vy).build(),
        };

        let h = problem.design_matrix(degree).unwrap();

        WeightedLeastSquares {
            y: y.to_owned(),
            h,
            covariance: covariance.clone(),
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

        let wls: WeightedLeastSquares<'_, E> = wls(x.view(), y.view(), &Covariance::None, degree);

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
            &Covariance::Diagonal(standard_deviation.view()),
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
    #[allow(clippy::too_many_lines)]
    fn sigmoid_fit_scaling() {
        use ndarray::arr1;
        let stimulus_values = arr1(&vec![
            0.0,
            0.034_835_625_876_632_91,
            0.069_671_251_753_265_81,
            0.104_506_877_629_898_71,
            0.139_342_503_506_531_63,
            0.174_178_129_383_164_54,
            0.209_013_755_259_797_43,
            0.243_849_381_136_430_34,
            0.278_685_007_013_063_26,
            0.313_520_632_889_696_17,
            0.348_356_258_766_329_1,
            0.383_191_884_642_962,
            0.418_027_510_519_594_86,
            0.452_863_136_396_227_8,
            0.487_698_762_272_860_7,
            0.522_534_388_149_493_6,
            0.557_370_014_026_126_5,
            0.592_205_639_902_759_4,
            0.627_041_265_779_392_3,
            0.661_876_891_656_025_3,
            0.696_712_517_532_658_2,
            0.731_548_143_409_291_1,
            0.766_383_769_285_924,
            0.801_219_395_162_556_9,
            0.836_055_021_039_189_7,
            0.870_890_646_915_822_6,
            0.905_726_272_792_455_5,
            0.940_561_898_669_088_5,
            0.975_397_524_545_721_4,
            1.010_233_150_422_354_3,
            1.045_068_776_298_987_2,
            1.079_904_402_175_620_1,
            1.114_740_028_052_253,
            1.149_575_653_928_886,
            1.184_411_279_805_518_9,
            1.219_246_905_682_151_8,
            1.254_082_531_558_784_7,
            1.288_918_157_435_417_6,
            1.323_753_783_312_050_5,
            1.358_589_409_188_683_4,
            1.393_425_035_065_316_3,
            1.428_260_660_941_949_3,
            1.463_096_286_818_582_2,
            1.497_931_912_695_215,
            1.532_767_538_571_848,
            1.567_603_164_448_481,
            1.602_438_790_325_113_8,
            1.637_274_416_201_746_7,
            1.672_110_042_078_379_4,
            1.706_945_667_955_012_3,
            1.741_781_293_831_645_3,
            1.776_616_919_708_278_2,
            1.811_452_545_584_911,
            1.846_288_171_461_544,
            1.881_123_797_338_177,
            1.915_959_423_214_809_8,
            1.950_795_049_091_442_7,
            1.985_630_674_968_075_7,
            2.020_466_300_844_708_6,
            2.055_301_926_721_341_5,
            2.090_137_552_597_974_4,
            2.124_973_178_474_607_3,
            2.159_808_804_351_240_2,
            2.194_644_430_227_873,
            2.229_480_056_104_506,
            2.264_315_681_981_139,
            2.299_151_307_857_772,
            2.333_986_933_734_405,
            2.368_822_559_611_037_7,
            2.403_658_185_487_670_6,
            2.438_493_811_364_303_5,
            2.473_329_437_240_936_5,
            2.508_165_063_117_569_4,
            2.543_000_688_994_202_3,
            2.577_836_314_870_835,
            2.612_671_940_747_468,
            2.647_507_566_624_101,
        ]);
        let response_values = arr1(&vec![
            0.819_224_840_651_049_1,
            0.848_255_011_271_882_7,
            0.878_313_498_013_660_9,
            0.909_436_697_284_801_7,
            0.941_662_291_627_678_7,
            0.975_029_295_015_962_7,
            1.009_578_099_736_471_6,
            1.045_350_524_910_141_8,
            1.082_389_866_708_612_7,
            1.120_740_950_324_751,
            1.160_450_183_757_433_3,
            1.201_565_613_472_852_3,
            1.244_136_982_006_74,
            1.288_215_787_573_954_4,
            1.333_855_345_754_145_3,
            1.381_110_853_324_391_4,
            1.430_039_454_312_101_6,
            1.480_700_308_343_792_2,
            1.533_154_661_367_886_8,
            1.587_465_918_832_144,
            1.643_699_721_399_007_2,
            1.701_924_023_284_762_8,
            1.762_209_173_311_248_2,
            1.824_627_998_761_584,
            1.889_255_892_134_419,
            1.956_170_900_894_07,
            2.025_453_820_317_112,
            2.097_188_289_539_032_4,
            2.171_460_890_907_888_5,
            2.248_361_252_755_16,
            2.327_982_155_697_452,
            2.410_419_642_586_135_5,
            2.495_773_132_225_674_7,
            2.584_145_536_984_940_4,
            2.675_643_384_429_694_7,
            2.770_376_943_108_121_5,
            2.868_460_352_625_363,
            2.970_011_758_146_846_6,
            3.075_153_449_474_484,
            3.184_012_004_843_794_8,
            3.296_718_439_594_464,
            3.413_408_359_871_003,
            3.534_222_121_514_779_8,
            3.659_304_994_313_004,
            3.788_807_331_775_010_4,
            3.922_884_746_610_629,
            4.061_698_292_090_357,
            4.205_414_649_471_582,
            4.354_206_321_680_171,
            4.508_251_833_441_441,
            4.667_735_938_059_521,
            4.832_849_831_049_133,
            5.003_791_370_828_718_5,
            5.180_765_306_688_915,
            5.363_983_514_255_387,
            5.553_665_238_670_095,
            5.750_037_345_720_057,
            5.953_334_581_147_634,
            6.163_799_838_381_394,
            6.381_684_434_931_474,
            6.607_248_397_698_195_5,
            6.840_760_757_447_379,
            7.082_499_852_710_5,
            7.332_753_643_372_316,
            7.591_820_034_212_885,
            7.860_007_208_675_012,
            8.137_633_973_132_2,
            8.425_030_111_935_792,
            8.722_536_753_523_427,
            9.030_506_747_873_936,
            9.349_305_055_596_675,
            9.679_309_148_945_547,
            10.020_909_425_049_86,
            10.374_509_631_655_531,
            10.740_527_305_671_037,
            11.119_394_224_812_78,
            11.511_556_872_644_064,
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
