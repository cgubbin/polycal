use super::{Solution, SolveSystem, Uncertainty};
use crate::Result;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar};

pub struct TotalLeastSquares<'a, E> {
    pub(crate) y: Array1<E>,
    pub(crate) uncertainty_x: Uncertainty<'a, E>,
    pub(crate) uncertainty_y: Uncertainty<'a, E>,
    pub(crate) h: Array2<E>,
}

impl<'a, E: Lapack + Scalar<Real = E> + ScalarOperand> SolveSystem<E> for TotalLeastSquares<'a, E> {
    fn solve(&self) -> Result<Solution<E>> {
        match (&self.uncertainty_x, &self.uncertainty_y) {
            (Uncertainty::Diagonal(ux), Uncertainty::Diagonal(uy)) => self.solve_diagonal(*ux, *uy),
            (Uncertainty::Full(vx), Uncertainty::Full(vy)) => self.solve_full_rank(*vx, *vy),
            _ => unreachable!("make this inaccessible via type."),
        }
    }
}

impl<'a, E: Lapack + Scalar<Real = E> + ScalarOperand> TotalLeastSquares<'a, E> {
    fn solve_diagonal(
        &self,
        _ux: ArrayView1<'a, E>,
        _uy: ArrayView1<'a, E>,
    ) -> Result<Solution<E>> {
        unimplemented!("no diagonal TLS impl");
    }

    fn solve_full_rank(
        &self,
        _vx: ArrayView2<'a, E>,
        _vy: ArrayView2<'a, E>,
    ) -> Result<Solution<E>> {
        unimplemented!("no full TLS impl");
    }
}
