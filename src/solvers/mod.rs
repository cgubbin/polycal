use crate::Result;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

mod total_least_squares;
mod weighted_least_squares;

pub use total_least_squares::TotalLeastSquares;
pub use weighted_least_squares::WeightedLeastSquares;

pub struct Solution<E> {
    coeff: Array1<E>,
    dependent_central_values: Option<Array1<E>>,
    covariance: Array2<E>,
}

impl<E> Solution<E> {
    pub(crate) fn coeff(&self) -> ArrayView1<'_, E> {
        self.coeff.view()
    }

    pub(crate) fn covariance(&self) -> ArrayView2<'_, E> {
        self.covariance.view()
    }
}

pub trait SolveSystem<E> {
    fn solve(&self) -> Result<Solution<E>>;
}

pub enum Uncertainty<'a, E> {
    None,
    Diagonal(ArrayView1<'a, E>),
    Full(ArrayView2<'a, E>),
}
