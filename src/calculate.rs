use argmin::{
    core::{
        observers::{ObserverMode, SlogLogger},
        ArgminFloat, CostFunction, Executor, Gradient, Hessian,
    },
    solver::{linesearch::MoreThuenteLineSearch, newton::NewtonCG},
};
use ndarray::{Array1, Array2, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar};
use num_traits::float::FloatCore;
use std::ops::Range;
use tracing::{event, Level};

use crate::chebyshev::{Polynomial, PolynomialSeries, Series};
use crate::error::Kind;
use crate::problem::Constraint;
use crate::utils::to_scaled;
use crate::{PolyCalError, PolyCalResult};

pub struct Fit<E> {
    /// The solution calculated using provided calibration data
    pub(crate) solution: Series<E>,
    /// Calculated covariance matrix for the fitting coefficients
    pub(crate) covariance: Array2<E>,
    /// The range of response values used in calibration
    pub(crate) response_domain: Range<E>,
    /// Constraint used in the fit procedure
    pub(crate) constraint: Option<Constraint<E>>,
}

#[derive(Copy, Clone, Debug)]
pub struct Unsure<E> {
    /// Central value, or mean, of the measurement
    pub estimate: E,
    /// Standard deviation of the measurement
    pub standard_uncertainty: E,
}

impl<E: Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd + tracing::Value>
    Fit<E>
{
    /// Direct evaluation y = `p_n(x`, a)
    ///
    /// Given a new stimulus value, estimate the response observed from the given calibration
    /// curve.
    ///
    /// # Errors
    /// If the provided stimulus lies outside the data-range used to form the calibration
    /// this method returns an error.
    #[tracing::instrument(skip(self))]
    pub fn response(&self, stimulus: Unsure<E>) -> PolyCalResult<Unsure<E>, E> {
        if !self.solution().domain().contains(&stimulus.estimate) {
            return Err(PolyCalError::OutOfRange {
                value: stimulus.estimate,
                range: self.solution().domain(),
                kind: Kind::Stimulus,
            });
        }
        let t = to_scaled(stimulus.estimate, self.domain());

        event!(Level::INFO, scaled = t, "evaluating series");
        let estimate = self.evaluate_direct(t);

        event!(Level::INFO, estimate = estimate, "evaluating uncertainty");
        let standard_uncertainty =
            self.evaluate_direct_uncertainty(t, stimulus.standard_uncertainty);

        Ok(Unsure {
            estimate,
            standard_uncertainty,
        })
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
        + tracing::Value,
{
    /// Inverse evaluation y - `p_n(x`, a) = 0
    ///
    /// Given a new response value, estimate the stimulus which led to it from the given calibration
    /// curve.
    ///
    /// # Errors
    /// If the provided response lies outside the data-range used to form the calibration
    /// this method returns an error.
    ///
    /// If there is an error in the underlying Gauss-Newton solver this method returns an error
    #[tracing::instrument(skip(self, guess))]
    pub fn stimulus(
        &self,
        response: Unsure<E>,
        guess: Option<E>,
        max_iter: Option<usize>,
    ) -> PolyCalResult<Unsure<E>, E> {
        if !self.response_domain.contains(&response.estimate) {
            return Err(PolyCalError::OutOfRange {
                value: response.estimate,
                range: self.response_domain.clone(),
                kind: Kind::Response,
            });
        }

        let scaled_estimate = self.evaluate_inverse(response.estimate, guess, max_iter)?;

        event!(Level::INFO, "evaluating uncertainty");
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
        Ok(self.series().derivative(1).evaluate(*param)
            * FloatCore::signum(self.series().evaluate(*param) - self.y0))
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
        Ok(self.series().derivative(2).evaluate(*param)
            * FloatCore::signum(self.series().evaluate(*param) - self.y0))
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
        + tracing::Value,
{
    #[tracing::instrument(skip(self, initial, max_iter))]
    pub(crate) fn evaluate_inverse(
        &self,
        y0: E,
        initial: Option<E>,
        max_iter: Option<usize>,
    ) -> ::std::result::Result<E, argmin::core::Error> {
        let target_domain = Range {
            start: -E::one(),
            end: E::one(),
        };

        // We know there will always be a root between - 1 and 1 if the stimulus value is within
        // the calibration data range. We assume this is checked by the caller, so here we can be
        // very sure the root exists.
        //
        // This does not preclude additional roots, lying outside [-1, 1] as the underlying
        // polynomial is only guaranteed to be monotonic on [-1, 1].
        //
        // If the minimisation produces a root outside [-1, 1] we search again, currently just
        // repeating indefinitely. If an initial parameter is provided this seeds the search, else
        // we start at zero.

        let mut init_param = initial.unwrap_or_else(|| E::zero());
        let mut root = FloatCore::max_value();
        let max_iter = max_iter.unwrap_or(100);

        let mut iter = 0;

        while !target_domain.contains(&root) && iter < max_iter {
            iter += 1;
            event!(
                Level::INFO,
                starting_point = init_param,
                iteration = iter,
                "beginning inverse solve"
            );

            let cost = InverseProblem {
                problem: self.solution(),
                y0,
                constraint: self.constraint.as_ref(),
            };

            // set up line search
            let linesearch = MoreThuenteLineSearch::new()
                .with_bounds(E::from(1e-8).unwrap(), E::from(1e-1).unwrap())?;

            // Set up solver
            let solver = NewtonCG::new(linesearch);

            // Run solver
            match Executor::new(cost, solver)
                .configure(|state| state.param(init_param).max_iters(50))
                .add_observer(SlogLogger::term(), ObserverMode::Never)
                .run()
            {
                Ok(res) => {
                    let mut state = res.state().clone();
                    root = state.take_param().unwrap();
                }
                Err(err) => tracing::warn!("error in minimisation {err:?}"),
            }

            if root > target_domain.end {
                init_param = (init_param + target_domain.start) / (E::one() + E::one());
            } else {
                init_param = (init_param + target_domain.end) / (E::one() + E::one());
            }
        }

        Ok(root)
    }
}

#[cfg(test)]
mod test {
    use ndarray::{Array1, Array2, ScalarOperand};
    use ndarray_linalg::{Lapack, Scalar};
    use ndarray_rand::{
        rand::{Rng, SeedableRng},
        rand_distr::{Distribution, Standard},
    };
    use num_traits::float::FloatCore;
    use rand_isaac::Isaac64Rng;
    use std::ops::Range;

    use super::{Fit, Unsure};
    use crate::chebyshev::{PolynomialSeries, Series};
    use crate::utils::find_limits;

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

    #[test]
    fn direct_evaluation_works_for_one_degree() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let degree = 1;
        let number_of_data_points = rng.gen_range(50..100);
        let domain = Range {
            start: -1.,
            end: 1.,
        };

        let (x, y, series) = generate_data(&mut rng, domain, number_of_data_points, degree);
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: None,
            response_domain: find_limits(y.as_slice().unwrap()),
        };

        for (ii, x) in x
            .into_iter()
            .enumerate()
            .skip(1)
            .take(number_of_data_points - 2)
        {
            let expected = y[ii];
            let calculated = fit
                .response(Unsure {
                    estimate: x,
                    standard_uncertainty: 0.0,
                })
                .unwrap();

            approx::assert_relative_eq!(expected, calculated.estimate, max_relative = 1e-7);
        }
    }

    #[test]
    fn inverse_evaluation_works_for_one_degree() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let degree = 1;
        let number_of_data_points = rng.gen_range(50..100);
        let domain = Range {
            start: -1.,
            end: 1.,
        };

        let (x, y, series) = generate_data(&mut rng, domain, number_of_data_points, degree);
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: None,
            response_domain: find_limits(y.as_slice().unwrap()),
        };

        for (ii, y) in y
            .into_iter()
            .enumerate()
            .skip(1)
            .take(number_of_data_points - 2)
        {
            let expected = x[ii];
            let calculated = fit
                .stimulus(
                    Unsure {
                        estimate: y,
                        standard_uncertainty: 0.0,
                    },
                    None,
                    None,
                )
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
        let domain = Range {
            start: -1.,
            end: 1.,
        };

        let (x, y, series) = generate_data(&mut rng, domain, number_of_data_points, degree);
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: None,
            response_domain: find_limits(y.as_slice().unwrap()),
        };

        for (ii, x) in x
            .into_iter()
            .enumerate()
            .skip(1)
            .take(number_of_data_points - 2)
        {
            let expected = y[ii];
            let calculated = fit
                .response(Unsure {
                    estimate: x,
                    standard_uncertainty: 0.0,
                })
                .unwrap();

            approx::assert_relative_eq!(expected, calculated.estimate, max_relative = 1e-7);
        }
    }

    #[test]
    fn inverse_evaluation_works_for_degree_two() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let degree = 2;
        let number_of_data_points = rng.gen_range(50..100);
        let domain = Range {
            start: -1.,
            end: 1.,
        };
        let mut monotonic_series = None;
        let mut monotonic_x = None;
        let mut monotonic_y = None;

        // We need a monotonic training function
        loop {
            let (x, y, series) =
                generate_data(&mut rng, domain.clone(), number_of_data_points, degree);

            if series
                .is_monotonic()
                .expect("failure in monotonicity check")
            {
                monotonic_series = Some(series);
                monotonic_x = Some(x);
                monotonic_y = Some(y);
                break;
            }
        }

        let series = monotonic_series.unwrap();
        let x = monotonic_x.unwrap();
        let y = monotonic_y.unwrap();
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: None,
            response_domain: find_limits(y.as_slice().unwrap()),
        };

        for (ii, y) in y
            .into_iter()
            .enumerate()
            .skip(1)
            .take(number_of_data_points - 2)
        {
            let expected = x[ii];
            let calculated = fit
                .stimulus(
                    Unsure {
                        estimate: y,
                        standard_uncertainty: 0.0,
                    },
                    None,
                    None,
                )
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
        let domain = Range {
            start: -1.,
            end: 1.,
        };
        let mut monotonic_series = None;
        let mut monotonic_x = None;
        let mut monotonic_y = None;

        // We need a monotonic training function
        loop {
            let (x, y, series) =
                generate_data(&mut rng, domain.clone(), number_of_data_points, degree);

            if series
                .is_monotonic()
                .expect("failure in monotonicity check")
            {
                monotonic_series = Some(series);
                monotonic_x = Some(x);
                monotonic_y = Some(y);
                break;
            }
        }

        let series = monotonic_series.unwrap();
        let x = monotonic_x.unwrap();
        let y = monotonic_y.unwrap();
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: None,
            response_domain: find_limits(y.as_slice().unwrap()),
        };

        for (ii, y) in y
            .into_iter()
            .enumerate()
            .skip(1)
            .take(number_of_data_points - 2)
        {
            let expected = x[ii];
            let calculated = fit
                .stimulus(
                    Unsure {
                        estimate: y,
                        standard_uncertainty: 0.0,
                    },
                    None,
                    None,
                )
                .expect("failed to solve the minimisation problem");

            if expected == 0.0 {
                approx::assert_relative_eq!(expected, calculated.estimate, epsilon = 1e-5);
            } else {
                approx::assert_relative_eq!(expected, calculated.estimate, max_relative = 1e-5);
            }
        }
    }
}
