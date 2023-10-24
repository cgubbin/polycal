use argmin::core::ArgminFloat;
use ndarray::ScalarOperand;
use ndarray_linalg::{Lapack, Scalar};
use num_traits::float::FloatCore;

use crate::chebyshev::PolynomialSeries;
use crate::problem::Fit;
use crate::utils::to_scaled;
use crate::Result;

#[derive(Copy, Clone, Debug)]
pub struct Unsure<E> {
    pub(crate) estimate: E,
    pub(crate) standard_uncertainty: E,
}

impl<E: Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd> Fit<E> {
    /// Direct evaluation y = `p_n(x`, a)
    pub fn response(&self, stimulus: Unsure<E>) -> Unsure<E> {
        let t = to_scaled(stimulus.estimate, self.domain());
        let uncertainty_t = to_scaled(stimulus.standard_uncertainty, self.domain());

        let estimate = self.constraint().map_or_else(
            || self.evaluate_direct(t),
            |constraint| {
                self.evaluate_direct(t) * constraint.multiplicative.evaluate(t)
                    + constraint.additive.evaluate(t)
            },
        );
        // Todo: method with constraint
        let standard_uncertainty = self.evaluate_direct_uncertainty(t, uncertainty_t);

        Unsure {
            estimate,
            standard_uncertainty,
        }
    }
}

// impl<E> ChebyshevFitResult<E>
// where
//     E: ArgminFloat
//         + Scalar<Real = E>
//         + argmin_math::ArgminSub<E, E>
//         + argmin_math::ArgminAdd<E, E>
//         + argmin_math::ArgminZeroLike
//         + argmin_math::ArgminConj
//         + argmin_math::ArgminMul<E, E>
//         + argmin_math::ArgminL2Norm<E>
//         + argmin_math::ArgminDot<E, E>,
// {
//     /// Inverse evaluation y - `p_n(x`, a) = 0
//     fn stimulus(&self, response: Unsure<E>) -> Result<Unsure<E>> {
//         let estimate = self.constraint.as_ref().map_or_else(
//             || self.solution.inverse_eval(response.estimate),
//             |constraint| self.solution.inverse_eval_with_constraint(response.estimate, constraint)
//         )?;
//         // Todo: method with constraint
//         let standard_uncertainty = self.solution.standard_uncertainty_inverse(
//             estimate,
//             response.standard_uncertainty,
//             self.covariance.view(),
//         );
//         Ok(Unsure {
//             estimate,
//             standard_uncertainty,
//         })
//     }
// }
