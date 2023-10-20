use std::ops::{Range};

use argmin::core::ArgminFloat;
use ndarray_linalg::Scalar;

use crate::fit::ChebyshevFitResult;
use crate::Result;

#[derive(Copy, Clone, Debug)]
pub struct Unsure<E> {
    pub(crate) estimate: E,
    pub(crate) standard_uncertainty: E,
}

fn to_scaled<E: Scalar>(x: E, Range { start, end }: &Range<E>) -> E {
    (x + x - *start - *end) / (*end - *start)
}

impl<E: Scalar<Real = E>> ChebyshevFitResult<E> {
    /// Direct evaluation y = `p_n(x`, a)
    pub fn eval_from_stimulus(&self, stimulus: Unsure<E>) -> Unsure<E> {
        let t = to_scaled(stimulus.estimate, &self.solution.domain);
        dbg!(&t);
        let estimate = self.constraint.as_ref().map_or_else(
            || self.solution.eval(t),
            |constraint| self.solution.eval_with_constraint(t, &constraint.nu) + constraint.mu.eval(t)
        );
        // Todo: method with constraint
        let standard_uncertainty = self.solution.standard_uncertainty_direct(
            t,
            stimulus.standard_uncertainty,
            self.covariance.view(),
        );

        Unsure {
            estimate,
            standard_uncertainty,
        }
    }
}

impl<E> ChebyshevFitResult<E>
where
    E: ArgminFloat
        + Scalar<Real = E>
        + argmin_math::ArgminSub<E, E>
        + argmin_math::ArgminAdd<E, E>
        + argmin_math::ArgminZeroLike
        + argmin_math::ArgminConj
        + argmin_math::ArgminMul<E, E>
        + argmin_math::ArgminL2Norm<E>
        + argmin_math::ArgminDot<E, E>,
{
    /// Inverse evaluation y - `p_n(x`, a) = 0
    fn eval_from_response(&self, response: Unsure<E>) -> Result<Unsure<E>> {
        let estimate = self.constraint.as_ref().map_or_else(
            || self.solution.inverse_eval(response.estimate),
            |constraint| self.solution.inverse_eval_with_constraint(response.estimate, constraint)
        )?;
        // Todo: method with constraint
        let standard_uncertainty = self.solution.standard_uncertainty_inverse(
            estimate,
            response.standard_uncertainty,
            self.covariance.view(),
        );
        Ok(Unsure {
            estimate,
            standard_uncertainty,
        })
    }
}
