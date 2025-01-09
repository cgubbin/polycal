use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

mod total_least_squares;
mod weighted_least_squares;

pub use total_least_squares::TotalLeastSquares;
pub use weighted_least_squares::WeightedLeastSquares;

#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    #[error("failure in matrix inversion")]
    Inverse(ndarray_linalg::error::LinalgError),
    #[error("failure in least squares SVD")]
    LeastSquares(ndarray_linalg::error::LinalgError),
    #[error("failure in cholesky factorisation")]
    Cholesky(ndarray_linalg::error::LinalgError),
    #[error("failure in gauss-newton minimisation")]
    IterativeSolver(#[from] argmin::core::Error),
}

pub struct Solution<E> {
    pub(crate) coeff: Array1<E>,
    pub(crate) dependent_central_values: Option<Array1<E>>,
    pub(crate) covariance: Array2<E>,
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
    fn solve(&self) -> ::std::result::Result<Solution<E>, SolverError>;
}

#[derive(Clone)]
pub enum Covariance<'a, E> {
    None,
    Diagonal(ArrayView1<'a, E>),
    Matrix(ArrayView2<'a, E>),
}
