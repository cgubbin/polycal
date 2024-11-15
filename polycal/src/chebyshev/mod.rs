mod basis;
mod builder;
mod primitive;
mod series;

pub use basis::{Basis, ConstrainedPolynomial, Polynomial};

#[allow(clippy::module_name_repetitions)]
pub use builder::ChebyshevBuilder;
use ndarray_linalg::Scalar;
use primitive::CSeries;
pub use series::Series;
use std::ops::Range;

use crate::utils::to_scaled;

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, thiserror::Error)]
pub enum ChebyshevError {
    #[error("provided data must be free of NaN, or infinities")]
    InvalidData,
    #[error("failure in eigenvalue calculation")]
    Eigenvalue(#[from] ndarray_linalg::error::LinalgError),
    #[error("shape error in forming companion matrix")]
    Shape(ndarray::ShapeError),
}

pub trait PolynomialSeries<E: PartialOrd + Scalar<Real = E>>: Clone + Sized {
    #[must_use]
    fn derivative(&self, count: usize) -> Self {
        match count {
            // zero order just returns the current Series
            0 => self.to_owned(),
            // If count exceeds the polynomial degree + 1 the series is emptied
            count if count > self.degree() => Self::null(self.domain(), self.window()),
            // Else do n differentiation ops
            count => {
                let mut current = self.to_owned();
                for _ in 0..count {
                    current = current.first_derivative();
                }
                current
            }
        }
    }
    /// Finds all roots of the polynomial
    ///
    /// # Errors
    /// - If there is an error building the companion matrix
    fn roots(&self) -> Result<Vec<E>, ChebyshevError>;
    /// Finds all roots of the polynomial which lie in `window`
    ///
    /// # Errors
    /// - If there is an error building the companion matrix
    fn roots_in_window(&self) -> Result<bool, ChebyshevError> {
        let window = self.window();
        Ok(!self.roots()?.iter().any(|root| window.contains(root)))
    }
    fn evaluate(&self, t: E) -> E;
    fn evaluate_unscaled(&self, x: E) -> E {
        let t = to_scaled(x, &self.domain());
        self.evaluate(t)
    }
    #[must_use]
    fn first_derivative(&self) -> Self;
    fn degree(&self) -> usize;
    fn number_of_coefficients(&self) -> usize {
        self.degree() + 1
    }
    fn domain(&self) -> Range<E>;
    fn window(&self) -> Range<E>;
    fn null(domain: Range<E>, window: Range<E>) -> Self;
    /// Check if the polynomial is monotonic
    ///
    /// # Errors
    /// - If there is an error calculating the roots of the polynomial, or building the companion
    ///     matrix
    fn is_monotonic(&self) -> Result<bool, ChebyshevError> {
        let derivative = self.derivative(1);
        derivative.roots_in_window()
    }
}
