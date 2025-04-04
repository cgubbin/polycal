/// This module contains the Chebyshev polynomial basis and the Chebyshev series primitives
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

/// All the necessary methods for a polynomial series
pub trait PolynomialSeries<E: PartialOrd + Scalar<Real = E>>: Clone + Sized {
    /// Calculate the `count` order derivative of the polynomial
    #[must_use]
    fn derivative(&self, count: usize) -> Self {
        match count {
            // zero order just returns the current Series
            0 => self.to_owned(),
            // If count exceeds the polynomial degree + 1 the series is emptied and the null series
            // (which is a zero polynomial) is returned
            count if count > self.degree() => Self::null(self.domain()),
            // Else do n atomic differentiation operations to find the result
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
    fn roots_in_window(&self, window: Range<E>) -> Result<bool, ChebyshevError> {
        Ok(!self.roots()?.iter().any(|root| window.contains(root)))
    }
    /// Evaluate the polynomial using an input value `t` which is scaled to the domain of the
    /// polynomial.
    ///
    /// Polynomials are scaled to the domain [-1, +1]. The true unscaled limits [a, b] are stored
    /// on creation based on the input calibration data. This function evaluates the polynomial
    /// assuming the input value t has already been rescaled from [a, b] to [-1,1]
    fn evaluate(&self, t: E) -> E;
    /// Evaluate the polynomial using the true input value `x`
    ///
    /// Polynomials are scaled to the domain [-1, +1]. The true unscaled limits [a, b] are stored
    /// on creation based on the input calibration data. This function evaluates the polynomial
    /// assuming the input value x is in [a, b], rescaling it to [-1, +1] before calling the
    /// internal evaluation method
    fn evaluate_unscaled(&self, x: E) -> E {
        let t = to_scaled(x, &self.domain());
        self.evaluate(t)
    }
    /// Take a single order derivative of the polynomial series
    #[must_use]
    fn first_derivative(&self) -> Self;
    /// Returns the degree of the polynomial
    fn degree(&self) -> usize;
    /// The number of independent coefficients of the polynomial, which is the degree of the
    /// polynomial plus one for the scalar offset
    fn number_of_coefficients(&self) -> usize {
        self.degree() + 1
    }
    /// The domain is the physical range of independent values used to create the polynomial.
    fn domain(&self) -> Range<E>;
    /// Returns the null polynomial, which is the zero-polynomial defined on the same range and
    /// window as the original
    fn null(domain: Range<E>) -> Self;
    /// Returns true if the polynomial is monotonic
    ///
    /// The monotonicity of the polynomial can be checked by analysis of the first derivative. If
    /// the first derivative has no roots in the domain of the polynomial, then the polynomial is
    /// monotonic, and if not it is.
    ///
    /// # Errors
    /// - If there is an error calculating the roots of the polynomial, or building the companion
    ///     matrix
    fn is_monotonic(&self) -> Result<bool, ChebyshevError> {
        let derivative = self.derivative(1);
        derivative.roots_in_window(self.domain())
    }
}
