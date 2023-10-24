use super::{Solution, SolveSystem, Uncertainty};
use crate::utils::outer_product;
use crate::Result;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis, ScalarOperand};
use ndarray_linalg::{Cholesky, Inverse, Lapack, LeastSquaresSvd, Scalar, UPLO};

pub(crate) struct WeightedLeastSquares<'a, E> {
    pub(crate) y: Array1<E>,
    pub(crate) uncertainty: Uncertainty<'a, E>,
    pub(crate) h: Array2<E>,
}

impl<'a, E: Lapack + Scalar<Real = E> + ScalarOperand> SolveSystem<E>
    for WeightedLeastSquares<'a, E>
{
    fn solve(&self) -> Result<Solution<E>> {
        match self.uncertainty {
            Uncertainty::None => self.solve_unweighted(),
            Uncertainty::Diagonal(uy) => self.solve_weighted(uy),
            Uncertainty::Full(vy) => self.solve_full(vy),
        }
    }
}

impl<'a, E: Lapack + Scalar<Real = E> + ScalarOperand> WeightedLeastSquares<'a, E> {
    fn solve_unweighted(&self) -> Result<Solution<E>> {
        let mut lhs = self.h.to_owned();
        let rhs = self.y.to_owned();
        let scaling = lhs
            .mapv(|val| val.powi(2))
            .sum_axis(Axis(0))
            .mapv(ndarray_linalg::Scalar::sqrt);

        lhs /= &scaling;

        let result = lhs.least_squares(&rhs)?;

        let coeff = (&result.solution.t() / &scaling).t().to_owned();

        let covariance = (lhs.t().dot(&lhs)).inv()? / outer_product(&scaling, &scaling)?;

        Ok(Solution {
            coeff,
            dependent_central_values: None,
            covariance,
        })
    }

    fn solve_weighted(&self, uy: ArrayView1<'a, E>) -> Result<Solution<E>> {
        let mut lhs = self.h.to_owned();
        let uy = uy.to_owned();

        let rhs = self.y.to_owned() / uy.mapv(|x| x.powi(2));

        for (ii, uy) in uy.iter().enumerate() {
            let mut slice = lhs.slice_mut(s![ii, ..]);
            slice /= uy.powi(2);
        }

        let scaling = lhs
            .mapv(|val| val.powi(2))
            .sum_axis(Axis(0))
            .mapv(ndarray_linalg::Scalar::sqrt);

        lhs /= &scaling;

        let result = lhs.least_squares(&rhs)?;

        let coeff = (&result.solution.t() / &scaling).t().to_owned();

        let covariance = (lhs.t().dot(&lhs)).inv()? / outer_product(&scaling, &scaling)?;

        Ok(Solution {
            coeff,
            dependent_central_values: None,
            covariance,
        })
    }

    fn solve_full(&self, vy: ArrayView2<'a, E>) -> Result<Solution<E>> {
        let mut lhs = self.h.to_owned();
        let vy = vy.to_owned();

        let lower = vy.cholesky(UPLO::Lower)?;

        unimplemented!("no impl for full-rank WLS for now.");
    }
}
