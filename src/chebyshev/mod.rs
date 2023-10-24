mod basis;
mod builder;
mod primitive;
mod series;

use crate::Result;
pub use basis::{Basis, ConstrainedPolynomial, Polynomial};

#[allow(clippy::module_name_repetitions)]
pub use builder::ChebyshevBuilder;
use primitive::CSeries;
pub use series::Series;
use std::ops::Range;

pub trait PolynomialSeries<E: PartialOrd>: Clone + Sized {
    fn derivative(&self, count: usize) -> Self {
        match count {
            // zero order just returns the current Series
            0 => self.to_owned(),
            // If count exceeds the polynomial degree + 1 the series is emptied
            count if count > self.degree() + 1 => Self::null(self.domain(), self.window()),
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
    fn roots(&self) -> Result<Vec<E>>;
    fn roots_in_window(&self) -> Result<bool> {
        let window = self.window();
        Ok(self.roots()?.iter().any(|root| window.contains(root)))
    }
    fn evaluate(&self, t: E) -> E;
    fn first_derivative(&self) -> Self;
    fn degree(&self) -> usize;
    fn number_of_coefficients(&self) -> usize {
        self.degree() + 1
    }
    fn domain(&self) -> Range<E>;
    fn window(&self) -> Range<E>;
    fn null(domain: Range<E>, window: Range<E>) -> Self;
    fn is_monotonic(&self) -> Result<bool> {
        let derivative = self.first_derivative();
        derivative.roots_in_window()
    }
}
