//! Methods to calculate stimulus or response data from a known [`Fit`]
//!
//! Given a known [`Fit`], calculated using calibration data, we can predict new stimulus or
//! response values given the alternative.
//!
//! To predict new response values from a known stimulus we simply evaluate the underlying
//! polynomial series y = p_n(x); In the inverse case, to predict a new stimulus from a known
//! response we numerically minimise abs(y - p_n(x)) to find the root.
//!
//! Both prediction methods take an [`Unsure`] as an argument. This represents a new value with an
//! associated estimate and variance. They also return an [`Unsure`], propagating the error from
//! the input and combining it with that on the calculated fitting coefficients.

use argmin::{
    core::{
        observers::{ObserverMode, SlogLogger},
        ArgminFloat, CostFunction, Executor, Gradient, Hessian,
    },
    solver::{linesearch::MoreThuenteLineSearch, newton::NewtonCG},
};
use ndarray::{ArrayView1, Array1, Array2, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_rand::{rand_distr::{Normal, StandardNormal, Distribution}, rand::Rng};
use num_traits::float::FloatCore;
use std::ops::Range;
use tracing::{event, Level};

use crate::chebyshev::{Polynomial, PolynomialSeries, Series};
use crate::error::Kind;
use crate::problem::Constraint;
use crate::utils::to_scaled;
use crate::{PolyCalError, PolyCalResult};

#[derive(Clone, Debug)]
/// The results of a polynomial fit.
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
/// A value with associated estimate (expectation) and standard uncertainty.
pub struct Unsure<E> {
    /// Central value, or mean, of the measurement
    pub estimate: E,
    /// Standard deviation of the measurement
    pub standard_uncertainty: E,
}

impl<E> Fit<E> {
    /// Returns the range of stimulus values used in the calibration procedure.
    ///
    /// Calibrations are carried out on a finite region of parameter space. In the event a new
    /// prediction is requested using an input value outside this calibration region an error will
    /// be returned from the reconstrauction methods. Outside the calibration range the accuracy of
    /// the reconstruction is entirely uncertain.
    pub const fn stimulus_domain(&self) -> &Range<E> {
        &self.solution.domain
    }
}

impl<E: Scalar> Fit<E> {
    /// The number of coefficients in the polynomial fit.
    pub fn num_coeff(&self) -> usize {
        self.solution.coeff.len()
    }

    /// The coefficients associated with the underlying Chebyshev series
    pub fn coeff(&self) -> Vec<E> {
        self.solution.coeff()
    }

    /// The variance of the coefficients of the underlying Chebyshev series
    pub fn variance(&self) -> ArrayView1<'_, E> {
        self.covariance.diag()
    }
}

impl<E: num_traits::Float + Scalar<Real = E>> Fit<E>
where
    StandardNormal: Distribution<E>,
{
    /// Create a new [`Fit`] randomly using the known estimates and variances.
    ///
    /// Note that the [`Fit`] returned by this method should not be re-used in this function. The
    /// underlying expectations are replaced, and are no longer a good estimate of the central
    /// values of the distribution.
    ///
    /// # Panics
    /// - If distribution creation fails (unlikely)
    pub fn draw<R: Rng>(&self, rng: &mut R) -> Self {
        let coeff = self.solution.coeff();
        let var = self.covariance.diag();

        let mut fit = self.clone();

        let sampled_coeff = coeff.into_iter()
            .zip(var)
            .map(|(mean, var)| Normal::new(mean, Scalar::sqrt(*var)))
            .map(|maybe_dist| match maybe_dist {
                Ok(dist) => Ok(dist.sample(rng)),
                Err(e) => Err(e),
            })
            .collect::<Result<_,_>>()
            .unwrap();


        fit.solution.set_coeff(sampled_coeff);
        fit
    }

    /// Given a new set of coefficients, creates a new [`Fit`] with those as the central estimates.
    ///
    /// This method is helpful for callers who want to use a [`Fit`] result in a Monte Carlo
    /// method. Samples generated externally can be inserted into the [`Fit`] allowing the
    /// reconstruction methods to be utilised.
    pub fn from_coeff(&self, coeff: &[E]) -> Self {
        assert_eq!(coeff.len(), self.num_coeff());
        let mut fit = self.clone();
        fit.solution.set_coeff(coeff.to_vec());
        fit
    }

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
        let t = to_scaled(stimulus.estimate, self.stimulus_domain());

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

    /// Direct evaluation y = `p_n(x`, a)
    ///
    /// Given a new stimulus value, estimate the response observed from the given calibration
    /// curve. This method assumes the input to have no associated error, and does not calculate an
    /// associated error for the output.
    ///
    /// # Errors
    /// If the provided stimulus lies outside the data-range used to form the calibration
    /// this method returns an error.
    #[tracing::instrument(skip(self))]
    pub fn certain_response(&self, stimulus: E) -> PolyCalResult<E, E> {
        if !self.solution().domain().contains(&stimulus) {
            return Err(PolyCalError::OutOfRange {
                value: stimulus,
                range: self.solution().domain(),
                kind: Kind::Stimulus,
            });
        }
        let t = to_scaled(stimulus, self.stimulus_domain());

        let estimate = self.evaluate_direct(t);

        Ok(
            estimate,
        )
    }

    /// Direct evaluation of the derivative y' = `p_n'(x`, a)
    ///
    /// Given a new stimulus value, estimate the derivative of the response observed from the given
    /// calibration curve. This method assumes the input to have no associated error, and does not
    /// calculate an associated error for the output. This is useful for constructing Jacobians
    /// using the results of a fit.
    ///
    /// # Errors
    /// If the provided stimulus lies outside the data-range used to form the calibration
    /// this method returns an error.
    #[tracing::instrument(skip(self))]
    pub fn certain_response_derivative(&self, stimulus: E) -> PolyCalResult<E, E> {
        if !self.solution().domain().contains(&stimulus) {
            return Err(PolyCalError::OutOfRange {
                value: stimulus,
                range: self.solution().domain(),
                kind: Kind::Stimulus,
            });
        }
        let t = to_scaled(stimulus, self.stimulus_domain());

        let estimate = self.evaluate_direct_derivative(t);

        Ok(
            estimate,
        )
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
        let estimate = crate::utils::to_unscaled(scaled_estimate, self.stimulus_domain());
        let standard_uncertainty = estimate / scaled_estimate * scaled_standard_uncertainty;
        Ok(Unsure {
            estimate,
            standard_uncertainty,
        })
    }
}

impl<E> Fit<E> {
    /// Retusn the underlying solution
    pub(crate) const fn solution(&self) -> &Series<E> {
        &self.solution
    }

    /// Return the underlying constraint
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

    pub(crate) fn evaluate_direct_derivative(&self, t: E) -> E {
        self.constraint().map_or_else(
            || self.solution.derivative(1).evaluate(t),
            |constraint| {
                let poly = self.solution.clone() * constraint.multiplicative.clone() + constraint.additive.clone();
                poly.derivative(1).evaluate(t)
            },
        )
    }

    #[allow(clippy::suspicious_operation_groupings)]
    fn q(&self, t: E) -> E {
        let Range { start, end } = self.stimulus_domain();
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

struct InverseProblem<E> {
    problem: Series<E>,
    y0: E,
}

struct InverseProblemBuilder<'a, E> {
    problem: &'a Series<E>,
    y0: E,
    constraint: Option<&'a Constraint<E>>,
}

impl<'a, E> InverseProblemBuilder<'a, E>
where
    E: Scalar<Real = E>,
{
    const fn new(y0: E, problem: &'a Series<E>) -> Self {
        Self {
            y0,
            problem,
            constraint: None,
        }
    }

    const fn with_constraint(mut self, constraint: Option<&'a Constraint<E>>) -> Self {
        self.constraint = constraint;
        self
    }

    fn build(self) -> InverseProblem<E> {
        InverseProblem {
            problem: self.constraint.map_or_else(
                || self.problem.clone(),
                |Constraint {
                     multiplicative,
                     additive,
                 }| self.problem.clone() * multiplicative.clone() + additive.clone(),
                ),
            y0: self.y0
        }
    }
}


impl<E: ArgminFloat + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd>
    CostFunction for InverseProblem<E>
{
    type Param = E;
    type Output = E;

    fn cost(
        &self,
        param: &Self::Param,
    ) -> ::std::result::Result<Self::Output, argmin::core::Error> {
        Ok(Scalar::abs(self.problem.evaluate(*param) - self.y0))
    }
}

impl<E: ArgminFloat + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd>
    Gradient for InverseProblem<E>
{
    type Param = E;
    type Gradient = E;

    fn gradient(
        &self,
        param: &Self::Param,
    ) -> ::std::result::Result<Self::Gradient, argmin::core::Error> {
        Ok(self.problem.derivative(1).evaluate(*param)
            * FloatCore::signum(self.problem.evaluate(*param) - self.y0))
    }
}

impl<E: ArgminFloat + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd>
    Hessian for InverseProblem<E>
{
    type Param = E;
    type Hessian = E;

    fn hessian(
        &self,
        param: &Self::Param,
    ) -> ::std::result::Result<Self::Hessian, argmin::core::Error> {
        Ok(self.problem.derivative(2).evaluate(*param)
            * FloatCore::signum(self.problem.evaluate(*param) - self.y0))
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

            let cost = InverseProblemBuilder::new(y0, self.solution())
                .with_constraint(self.constraint.as_ref())
                .build();

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
    use crate::Constraint;

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

    pub fn generate_data_passing_through_origin<E>(
        rng: &mut impl Rng,
        Range { start, end }: Range<E>,
        num_points: usize,
        degree: usize,
    ) -> (Array1<E>, Array1<E>, Series<E>, Series<E>)
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
        let constraint = Series::from_coeff(vec![E::zero(), E::one()], x.as_slice().unwrap());

        let combined = series.clone() * constraint.clone();

        let y = x.iter().map(|x| combined.evaluate(*x)).collect::<Array1<E>>();

        (x, y, series, constraint)
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
    fn constrained_direct_evaluation_works_for_one_degree() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let degree = 1;
        let number_of_data_points = rng.gen_range(50..100);
        let domain = Range {
            start: -1.,
            end: 1.,
        };

        let (x, y, series, constraint) = generate_data_passing_through_origin(&mut rng, domain, number_of_data_points, degree);
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: Some(Constraint { multiplicative: constraint, additive: Series::from_coeff(vec![0.0], x.as_slice().unwrap()) }),
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
    fn constrained_inverse_evaluation_works_for_one_degree() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let degree = 1;
        let number_of_data_points = rng.gen_range(50..100);
        let domain = Range {
            start: -1.,
            end: 1.,
        };

        let mut monotonic_series = None;
        let mut monotonic_constraint = None;
        let mut monotonic_x = None;
        let mut monotonic_y = None;

        // We need a monotonic training function
        loop {
            let (x, y, series, constraint) = generate_data_passing_through_origin(&mut rng, domain.clone(), number_of_data_points, degree);
            let combined = series.clone() * constraint.clone();
            if combined
                .is_monotonic()
                .expect("failure in monotonicity check")
            {
                let _ = monotonic_series.insert(series);
                let _ = monotonic_constraint.insert(constraint);
                let _ = monotonic_x.insert(x);
                let _ = monotonic_y.insert(y);
                break;
            }
        }
        let series = monotonic_series.unwrap();
        let constraint = monotonic_constraint.unwrap();
        let x = monotonic_x.unwrap();
        let y = monotonic_y.unwrap();
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: Some(Constraint { multiplicative: constraint, additive: Series::from_coeff(vec![0.0], x.as_slice().unwrap()) }),
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
    fn constrained_direct_evaluation_works_for_degree_five() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let degree = 5;
        let number_of_data_points = rng.gen_range(50..100);
        let domain = Range {
            start: -1.,
            end: 1.,
        };

        let (x, y, series, constraint) = generate_data_passing_through_origin(&mut rng, domain, number_of_data_points, degree);
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: Some(Constraint { multiplicative: constraint, additive: Series::from_coeff(vec![0.0], x.as_slice().unwrap()) }),
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
                let _ = monotonic_series.insert(series);
                let _ = monotonic_x.insert(x);
                let _ = monotonic_y.insert(y);
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
    fn constrained_inverse_evaluation_works_for_degree_two() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let degree = 2;
        let number_of_data_points = rng.gen_range(50..100);
        let domain = Range {
            start: -1.,
            end: 1.,
        };

        let mut monotonic_series = None;
        let mut monotonic_constraint = None;
        let mut monotonic_x = None;
        let mut monotonic_y = None;

        // We need a monotonic training function
        loop {
            let (x, y, series, constraint) = generate_data_passing_through_origin(&mut rng, domain.clone(), number_of_data_points, degree);
            let combined = series.clone() * constraint.clone();
            if combined
                .is_monotonic()
                .expect("failure in monotonicity check")
            {
                let _ = monotonic_series.insert(series);
                let _ = monotonic_constraint.insert(constraint);
                let _ = monotonic_x.insert(x);
                let _ = monotonic_y.insert(y);
                break;
            }
        }
        let series = monotonic_series.unwrap();
        let constraint = monotonic_constraint.unwrap();
        let x = monotonic_x.unwrap();
        let y = monotonic_y.unwrap();
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: Some(Constraint { multiplicative: constraint, additive: Series::from_coeff(vec![0.0], x.as_slice().unwrap()) }),
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
                let _ = monotonic_series.insert(series);
                let _ = monotonic_x.insert(x);
                let _ = monotonic_y.insert(y);
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
    fn constrained_inverse_evaluation_works_for_degree_five() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let degree = 5;
        let number_of_data_points = rng.gen_range(50..100);
        let domain = Range {
            start: -1.,
            end: 1.,
        };

        let mut monotonic_series = None;
        let mut monotonic_constraint = None;
        let mut monotonic_x = None;
        let mut monotonic_y = None;

        // We need a monotonic training function
        loop {
            let (x, y, series, constraint) = generate_data_passing_through_origin(&mut rng, domain.clone(), number_of_data_points, degree);
            let combined = series.clone() * constraint.clone();
            if combined
                .is_monotonic()
                .expect("failure in monotonicity check")
            {
                let _ = monotonic_series.insert(series);
                let _ = monotonic_constraint.insert(constraint);
                let _ = monotonic_x.insert(x);
                let _ = monotonic_y.insert(y);
                break;
            }
        }
        let series = monotonic_series.unwrap();
        let constraint = monotonic_constraint.unwrap();
        let x = monotonic_x.unwrap();
        let y = monotonic_y.unwrap();
        let covariance = Array2::zeros((degree + 1, degree + 1));

        let fit = Fit {
            solution: series,
            covariance,
            constraint: Some(Constraint { multiplicative: constraint, additive: Series::from_coeff(vec![0.0], x.as_slice().unwrap()) }),
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
