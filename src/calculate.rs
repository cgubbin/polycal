use argmin::{
    core::{
        observers::{ObserverMode, SlogLogger},
        ArgminFloat, CostFunction, Executor, Gradient, Hessian,
    },
    solver::{linesearch::MoreThuenteLineSearch, newton::NewtonCG},
};
use ndarray::{Array1, Array2, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_rand::{rand::SeedableRng, rand_distr::{Uniform, uniform::SampleUniform, Distribution}};
use num_traits::float::FloatCore;
use rand_isaac::Isaac64Rng;
use std::ops::Range;

use crate::chebyshev::{Polynomial, PolynomialSeries, Series};
use crate::problem::Constraint;
use crate::utils::to_scaled;
use crate::Result;

pub struct Fit<E> {
    pub(crate) solution: Series<E>,
    pub(crate) covariance: Array2<E>,
    pub(crate) constraint: Option<Constraint<E>>,
}

#[derive(Copy, Clone, Debug)]
pub struct Unsure<E> {
    pub estimate: E,
    pub standard_uncertainty: E,
}

impl<E: Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd> Fit<E> {
    /// Direct evaluation y = `p_n(x`, a)
    pub fn response(&self, stimulus: Unsure<E>) -> Unsure<E> {
        let t = to_scaled(stimulus.estimate, self.domain());

        let estimate = self.evaluate_direct(t);

        // Todo: method with constraint
        let standard_uncertainty =
            self.evaluate_direct_uncertainty(t, stimulus.standard_uncertainty);

        Unsure {
            estimate,
            standard_uncertainty,
        }
    }
}

impl<E> Fit<E>
where
    E: ArgminFloat
        + Scalar<Real = E>
        + ScalarOperand
        + Lapack
        + FloatCore
        + PartialOrd
        + argmin_math::ArgminSub<E, E>
        + argmin_math::ArgminAdd<E, E>
        + argmin_math::ArgminZeroLike
        + argmin_math::ArgminConj
        + argmin_math::ArgminMul<E, E>
        + argmin_math::ArgminL2Norm<E>
        + argmin_math::ArgminDot<E, E>
        + SampleUniform,
    Uniform<E>: Distribution<E>,
{
    /// Inverse evaluation y - `p_n(x`, a) = 0
    pub fn stimulus(&self, response: Unsure<E>, guess: Option<E>) -> Result<Unsure<E>> {
        let scaled_estimate = self.evaluate_inverse(response.estimate, guess)?;

        let scaled_standard_uncertainty =
            self.evaluate_inverse_uncertainty(scaled_estimate, response.standard_uncertainty);

        // Scale back to the true data type
        let estimate = crate::utils::to_unscaled(scaled_estimate, self.domain());
        let standard_uncertainty = estimate / scaled_estimate * scaled_standard_uncertainty;
        Ok(Unsure {
            estimate,
            standard_uncertainty,
        })
    }
}

impl<E> Fit<E> {
    pub(crate) const fn domain(&self) -> &Range<E> {
        &self.solution.domain
    }

    pub(crate) const fn solution(&self) -> &Series<E> {
        &self.solution
    }

    pub(crate) const fn constraint(&self) -> Option<&Constraint<E>> {
        self.constraint.as_ref()
    }
}

impl<E: Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd> Fit<E> {
    pub(crate) fn evaluate_direct(&self, t: E) -> E {
        self.constraint().map_or_else(
            || self.solution.evaluate(t),
            |constraint| {
                self.solution.evaluate(t) * constraint.multiplicative.evaluate(t)
                    + constraint.additive.evaluate(t)
            },
        )
    }

    #[allow(clippy::suspicious_operation_groupings)]
    fn q(&self, t: E) -> E {
        let Range { start, end } = self.domain();
        let series = self.constraint.as_ref().map_or_else(
            || self.solution.clone(),
            |constraint| {
                self.solution.clone() * constraint.multiplicative.clone()
                    + constraint.additive.clone()
            },
        );
        (E::one() + E::one()) / (*end - *start) * series.derivative(1).evaluate(t)
    }

    pub(crate) fn evaluate_direct_uncertainty(&self, t: E, uncertainty_x: E) -> E {
        let g: Array1<E> = self.constraint.as_ref().map_or_else(
            || self.solution.basis.polynomials(t).into(),
            |constraint| {
                self.solution
                    .basis
                    .polynomials(t)
                    .into_iter()
                    .map(|poly| poly * constraint.multiplicative.evaluate(t))
                    .collect()
            },
        );

        (Scalar::powi(self.q(t), 2) * Scalar::powi(uncertainty_x, 2)
            + g.dot(&self.covariance.dot(&g)))
        .sqrt()
    }

    pub(crate) fn evaluate_inverse_uncertainty(&self, t: E, uncertainty_y: E) -> E {
        let g: Array1<E> = self.constraint.as_ref().map_or_else(
            || self.solution.basis.polynomials(t).into(),
            |constraint| {
                self.solution
                    .basis
                    .polynomials(t)
                    .into_iter()
                    .map(|poly| poly * constraint.multiplicative.evaluate(t))
                    .collect()
            },
        );

        E::one() / Scalar::powi(self.q(t), 2)
            * (Scalar::powi(uncertainty_y, 2) + g.dot(&self.covariance.dot(&g))).sqrt()
    }
}

struct InverseProblem<'a, E> {
    problem: &'a Series<E>,
    y0: E,
    constraint: Option<&'a Constraint<E>>,
}

impl<'a, E: ArgminFloat + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd>
    InverseProblem<'a, E>
{
    fn series(&self) -> Series<E> {
        self.constraint.map_or_else(
            || self.problem.clone(),
            |Constraint {
                 multiplicative,
                 additive,
             }| self.problem.clone() * multiplicative.clone() + additive.clone(),
        )
    }
}

impl<'a, E: ArgminFloat + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd>
    CostFunction for InverseProblem<'a, E>
{
    type Param = E;
    type Output = E;

    fn cost(
        &self,
        param: &Self::Param,
    ) -> ::std::result::Result<Self::Output, argmin::core::Error> {
        Ok(Scalar::abs(self.series().evaluate(*param) - self.y0))
    }
}

impl<'a, E: ArgminFloat + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd>
    Gradient for InverseProblem<'a, E>
{
    type Param = E;
    type Gradient = E;

    fn gradient(
        &self,
        param: &Self::Param,
    ) -> ::std::result::Result<Self::Gradient, argmin::core::Error> {
        Ok(
            self.series().derivative(1).evaluate(*param)
                * FloatCore::signum(self.series().evaluate(*param) - self.y0)
        )
    }
}

impl<'a, E: ArgminFloat + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd>
    Hessian for InverseProblem<'a, E>
{
    type Param = E;
    type Hessian = E;

    fn hessian(
        &self,
        param: &Self::Param,
    ) -> ::std::result::Result<Self::Hessian, argmin::core::Error> {
        Ok(
            self.series().derivative(2).evaluate(*param)
                * FloatCore::signum(self.series().evaluate(*param) - self.y0)
        )
    }
}

impl<E> Fit<E>
where
    E: ArgminFloat
        + Scalar<Real = E>
        + FloatCore
        + Lapack
        + ScalarOperand
        + PartialOrd
        + argmin_math::ArgminSub<E, E>
        + argmin_math::ArgminAdd<E, E>
        + argmin_math::ArgminZeroLike
        + argmin_math::ArgminConj
        + argmin_math::ArgminMul<E, E>
        + argmin_math::ArgminL2Norm<E>
        + argmin_math::ArgminDot<E, E>
        + SampleUniform,
    Uniform<E>: Distribution<E>,
{
    pub(crate) fn evaluate_inverse(&self, y0: E, initial: Option<E>) -> Result<E> {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let dist = Uniform::from(-E::one()..E::one());
        let target_domain = Range { start: - E::one(), end: E::one() };

        let mut param_in_domain = None;
        let ii = 0;

        loop {

            let init_param = if ii == 1 {
                initial.map_or_else(|| dist.sample(&mut rng), |initial| initial)
            } else {
                dist.sample(&mut rng)
            };

            let cost = InverseProblem {
                problem: self.solution(),
                y0,
                constraint: self.constraint.as_ref(),
            };

            // set up line search
            let linesearch = MoreThuenteLineSearch::new();

            // Set up solver
            let solver = NewtonCG::new(linesearch);

            // Run solver
            match Executor::new(cost, solver)
                .configure(|state| state.param(init_param).max_iters(50))
                .add_observer(SlogLogger::term(), ObserverMode::Never)
                .run() {
                Ok(res) => {
                    let mut state = res.state().clone();
                    let param = state.take_param().unwrap();

                    if target_domain.contains(&param) {
                        param_in_domain = Some(param);
                        break
                    }
                }
                Err(err) => tracing::warn!("error in minimisation {err:?}"),
            }
        }

        Ok(param_in_domain.unwrap())
    }
}

#[cfg(test)]
mod test {
    use ndarray::{Array1, ScalarOperand, Array2};
    use ndarray_linalg::{Scalar, Lapack};
    use ndarray_rand::{rand::{SeedableRng, Rng}, rand_distr::{Distribution, Standard}};
    use num_traits::float::FloatCore;
    use rand_isaac::Isaac64Rng;
    use std::ops::Range;

    use crate::chebyshev::{PolynomialSeries, Series};
    use super::{Fit, Unsure};

    pub(crate) fn generate_data<E>(rng: &mut impl Rng, Range{ start, end }: Range<E>, num_points: usize, degree: usize) -> (Array1<E>, Array1<E>, Series<E>)
    where
        E: Scalar<Real = E> + ScalarOperand + PartialOrd + Lapack + FloatCore,
        Standard: Distribution<E>,
    {
        let chebyshev_coeffs = (0..=degree).map(|_| rng.gen()).collect::<Vec<_>>();

        let x = (0..num_points).map(|m| start + E::from(m).unwrap() * (end - start) / (E::from(num_points).unwrap() - E::one())).collect::<Array1<_>>();

        let series = Series::from_coeff(chebyshev_coeffs, x.as_slice().unwrap());

        let y = x
            .iter()
            .map(|x| series.evaluate(*x))
            .collect::<Array1<E>>();

        (x, y, series)
    }

    #[test]
    fn direct_evaluation_works_for_one_degree() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let degree = 1;
        let number_of_data_points = rng.gen_range(50..100);
        let domain = Range { start: -1., end: 1. };


        let (x, y, series) = generate_data(&mut rng, domain, number_of_data_points, degree);
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: None
        };

        for (ii, x) in x.into_iter().enumerate() {
            let expected = y[ii];
            let calculated = fit.response( Unsure { estimate: x, standard_uncertainty: 0.0 });

            approx::assert_relative_eq!(expected, calculated.estimate, max_relative = 1e-7);
        }
    }

    #[test]
    fn inverse_evaluation_works_for_one_degree() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let degree = 1;
        let number_of_data_points = rng.gen_range(50..100);
        let domain = Range { start: -1., end: 1. };


        let (x, y, series) = generate_data(&mut rng, domain, number_of_data_points, degree);
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: None
        };

        for (ii, y) in y.into_iter().enumerate() {
            let expected = x[ii];
            let calculated = fit.stimulus( Unsure { estimate: y, standard_uncertainty: 0.0 }, None)
                .expect("failed to solve the minimisation problem");
            if expected == 0.0 {
                approx::assert_relative_eq!(expected, calculated.estimate, epsilon = 1e-5);
            } else {
                approx::assert_relative_eq!(expected, calculated.estimate, max_relative = 1e-5);
            }
        }
    }

    #[test]
    fn direct_evaluation_works_for_degree_five() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let degree = 5;
        let number_of_data_points = rng.gen_range(50..100);
        let domain = Range { start: -1., end: 1. };


        let (x, y, series) = generate_data(&mut rng, domain, number_of_data_points, degree);
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: None
        };

        for (ii, x) in x.into_iter().enumerate() {
            let expected = y[ii];
            let calculated = fit.response( Unsure { estimate: x, standard_uncertainty: 0.0 });

            approx::assert_relative_eq!(expected, calculated.estimate, max_relative = 1e-7);
        }
    }

    #[test]
    fn inverse_evaluation_works_for_degree_two() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let degree = 2;
        let number_of_data_points = rng.gen_range(50..100);
        let domain = Range { start: -1., end: 1. };
        let mut monotonic_series = None;
        let mut monotonic_x = None;
        let mut monotonic_y = None;

        // We need a monotonic training function
        loop {
            let (x, y, series) = generate_data(&mut rng, domain.clone(), number_of_data_points, degree);

            if series.is_monotonic()
                .expect("failure in monotonicity check")
            {
                monotonic_series = Some(series);
                monotonic_x = Some(x);
                monotonic_y = Some(y);
                break
            }
        }

        let series = monotonic_series.unwrap();
        let x = monotonic_x.unwrap();
        let y = monotonic_y.unwrap();
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: None
        };

        for (ii, y) in y.into_iter().enumerate() {
            let expected = x[ii];
            let calculated = fit.stimulus( Unsure { estimate: y, standard_uncertainty: 0.0 }, None)
                .expect("failed to solve the minimisation problem");

            if expected == 0.0 {
                approx::assert_relative_eq!(expected, calculated.estimate, epsilon = 1e-5);
            } else {
                approx::assert_relative_eq!(expected, calculated.estimate, max_relative = 1e-5);
            }
        }
    }

    #[test]
    fn inverse_evaluation_works_for_degree_five() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let degree = 5;
        let number_of_data_points = rng.gen_range(50..100);
        let domain = Range { start: -1., end: 1. };
        let mut monotonic_series = None;
        let mut monotonic_x = None;
        let mut monotonic_y = None;

        // We need a monotonic training function
        loop {
            let (x, y, series) = generate_data(&mut rng, domain.clone(), number_of_data_points, degree);

            if series.is_monotonic()
                .expect("failure in monotonicity check")
            {
                monotonic_series = Some(series);
                monotonic_x = Some(x);
                monotonic_y = Some(y);
                break
            }
        }

        let series = monotonic_series.unwrap();
        let x = monotonic_x.unwrap();
        let y = monotonic_y.unwrap();
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: None
        };

        for (ii, y) in y.into_iter().enumerate() {
            let expected = x[ii];
            let calculated = fit.stimulus( Unsure { estimate: y, standard_uncertainty: 0.0 }, None)
                .expect("failed to solve the minimisation problem");

            if expected == 0.0 {
                approx::assert_relative_eq!(expected, calculated.estimate, epsilon = 1e-5);
            } else {
                approx::assert_relative_eq!(expected, calculated.estimate, max_relative = 1e-5);
            }
        }
    }
}
