use std::ops::{Range, AddAssign};

use ndarray_linalg::Scalar;

use crate::Result;
use crate::fit::ChebyshevFitResult;

pub struct Unsure<E> {
    estimate: E,
    standard_uncertainty: E,
}

fn to_scaled<E: Scalar>(x: E, Range { start, end }: &Range<E>) -> E {
    (x + x - *start - *end) / (*end - *start)
}

impl<E: Scalar<Real = E>> ChebyshevFitResult<E> {
    /// Direct evaluation y = p_n(x, a)
    fn eval_from_stimulus(&self, stimulus: Unsure<E>) -> Result<Unsure<E>> {
        let t = to_scaled(stimulus.estimate, &self.solution.domain);
        let estimate = self.solution.eval(t);
        let standard_uncertainty = self.solution.standard_uncertainty_direct(
            t,
            stimulus.standard_uncertainty,
            self.covariance.view()
        );

        Ok(Unsure { estimate, standard_uncertainty })
    }

    /// Inverse evaluation y - p_n(x, a) = 0
    fn eval_from_response(&self, response: Unsure<E>) -> Result<Unsure<E>> {
        let estimate = self.solution.inverse_eval(response.estimate);
        let standard_uncertainty = self.solution.standard_uncertainty_inverse(
            estimate,
            response.standard_uncertainty,
            self.covariance.view()
        );
        Ok(Unsure { estimate, standard_uncertainty })
    }
}
