
use argmin::{core::{ArgminFloat, CostFunction, Gradient, Hessian, Executor, observers::{SlogLogger, ObserverMode}}, solver::{linesearch::MoreThuenteLineSearch, newton::NewtonCG}};
use ndarray::{Array1, Array2, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar};
use num_traits::float::FloatCore;
use std::ops::Range;

use crate::Result;
use crate::chebyshev::{Polynomial, PolynomialSeries, Series};
use crate::problem::Constraint;
use crate::utils::to_scaled;

pub struct Fit<E> {
    pub(crate) solution: Series<E>,
    pub(crate) covariance: Array2<E>,
    pub(crate) constraint: Option<Constraint<E>>,
}


#[derive(Copy, Clone, Debug)]
pub struct Unsure<E> {
    pub(crate) estimate: E,
    pub(crate) standard_uncertainty: E,
}

impl<E: Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd> Fit<E> {
    /// Direct evaluation y = `p_n(x`, a)
    pub fn response(&self, stimulus: Unsure<E>) -> Unsure<E> {
        let t = to_scaled(stimulus.estimate, self.domain());

        let estimate = self.evaluate_direct(t);

        // Todo: method with constraint
        let standard_uncertainty = self.evaluate_direct_uncertainty(t, stimulus.standard_uncertainty);

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
        + argmin_math::ArgminDot<E, E>,
{
    /// Inverse evaluation y - `p_n(x`, a) = 0
    fn stimulus(&self, response: Unsure<E>) -> Result<Unsure<E>> {
        let estimate = self.evaluate_inverse(response.estimate)?;

        // Todo: method with constraint
        let standard_uncertainty = self.evaluate_inverse_uncertainty(
            estimate,
            response.standard_uncertainty,
        );
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
            |constraint| self.solution.clone() * constraint.multiplicative.clone() + constraint.additive.clone()
        );
        (E::one() + E::one()) / (*end - *start) * series.first_derivative().evaluate(t)
    }

    pub(crate) fn evaluate_direct_uncertainty(&self, t: E, uncertainty_x: E) -> E {
        let g: Array1<E> = self.constraint.as_ref().map_or_else(
            || self.solution.basis.polynomials(t).into(),
            |constraint| self.solution.basis.polynomials(t)
                .into_iter()
                .map(|poly| poly * constraint.multiplicative.evaluate(t))
                .collect()
        );

        (Scalar::powi(self.q(t), 2) * Scalar::powi(uncertainty_x, 2)
            + g.dot(&self.covariance.dot(&g)))
        .sqrt()
    }

    pub(crate) fn evaluate_inverse_uncertainty(&self, t: E, uncertainty_y: E) -> E {
        let g: Array1<E> = self.constraint.as_ref().map_or_else(
            || self.solution.basis.polynomials(t).into(),
            |constraint| self.solution.basis.polynomials(t)
                .into_iter()
                .map(|poly| poly * constraint.multiplicative.evaluate(t))
                .collect()
        );

        E::one() / Scalar::powi(self.q(t), 2) * (Scalar::powi(uncertainty_y, 2)
            + g.dot(&self.covariance.dot(&g)))
        .sqrt()
    }
}

struct InverseProblem<'a, E> {
    problem: &'a Series<E>,
    y0: E,
    constraint: Option<&'a Constraint<E>>,
}

impl<'a, E: ArgminFloat + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd> InverseProblem<'a, E> {
    fn series(&self) -> Series<E> {
        self.constraint.map_or_else(
            || self.problem.clone(),
            |Constraint { multiplicative, additive }| self.problem.clone() * multiplicative.clone() + additive.clone(),
        )
    }
}

impl<'a, E: ArgminFloat + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd> CostFunction for InverseProblem<'a, E> {
    type Param = E;
    type Output = E;

    fn cost(
        &self,
        param: &Self::Param,
    ) -> ::std::result::Result<Self::Output, argmin::core::Error> {
        Ok(Scalar::abs(self.series().evaluate(*param) - self.y0))
    }
}

impl<'a, E: ArgminFloat + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd> Gradient for InverseProblem<'a, E> {
    type Param = E;
    type Gradient = E;

    fn gradient(
        &self,
        param: &Self::Param,
    ) -> ::std::result::Result<Self::Gradient, argmin::core::Error> {
        Ok(self.series().first_derivative().evaluate(*param))
    }
}

impl<'a, E: ArgminFloat + Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd> Hessian for InverseProblem<'a, E> {
    type Param = E;
    type Hessian = E;

    fn hessian(
        &self,
        param: &Self::Param,
    ) -> ::std::result::Result<Self::Hessian, argmin::core::Error> {
        Ok(self.series().derivative(2).evaluate(*param))
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
        + argmin_math::ArgminDot<E, E>,
{
    pub(crate) fn evaluate_inverse(&self, y0: E) -> Result<E> {
        let cost = InverseProblem { problem: self.solution(), y0, constraint: self.constraint.as_ref() };
        let init_param = E::zero();

        // set up line search
        let linesearch = MoreThuenteLineSearch::new();

        // Set up solver
        let solver = NewtonCG::new(linesearch);

        // Run solver
        let res = Executor::new(cost, solver)
            .configure(|state| state.param(init_param).max_iters(100))
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;

        let mut state = res.state().clone();
        let param = state.take_param();

        Ok(param.unwrap())
    }
}


