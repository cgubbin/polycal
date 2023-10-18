mod chebyshev;
mod fit;

pub(crate) use chebyshev::ChebyshevPolynomial;

// use ndarray::{Array1, Array2, s};
// use num_traits::{Float, FromPrimitive};
// use std::ops::Range;
//
// pub struct Polynomial<E, const N: usize> {
//     /// Coefficient vector of length `N+1
//     coefficients: [E; N],
//     /// Covariance matrix
//     covariance: [[E; N]; N],
//     /// Limits of calibration data
//     limits: Range<E>,
// }
//
// pub struct PolynomialFit<E> {
//     /// Coefficient vector of length `N+1
//     coefficients: Vec<E>,
//     /// Covariance matrix
//     covariance: Vec<Vec<E>>,
//     /// Limits of calibration data
//     limits: Range<E>,
// }
//
// fn outer_product<T: Float>(a: &Array1<T>, b: &Array1<T>) -> Result<Array2<T>> {
//     let a: Array2<T> = a.clone().into_shape((a.len(), 1))?;
//     let b: Array2<T> = b.clone().into_shape((1, b.len()))?;
//
//     Ok(ndarray::linalg::kron(&a, &b))
// }
//
// impl<E: Float + FromPrimitive> PolynomialFit<E> {
//     fn n(&self) -> usize {
//         self.coefficients.len()
//     }
//
//     /// Returns true if the polynomial fit is monotonic in the fit range
//     ///
//     /// The zeros of the polynomial Q_{n-1} are the eigenvalues of the Colleague Matrix of the
//     /// polynomial. If all the eigenvalues of the colleague matrix lie outside [-1, 1] the
//     /// polynomial is monotonic in [-1, 1].
//     fn is_monotonic(&self) -> bool {
//         let colleague_matrix = self.colleague_matrix();
//
//         let eig: Array1<E> = Array1::zeros(self.n() - 1); // TODO: get the eig
//
//         eig.into_iter()
//             .all(|x| (x < - E::one()) || (x > E::one()))
//     }
//
//     /// Compute the colleague matrix for the Chebyshev polynomial.
//     fn colleague_matrix(&self) -> Array2<E> {
//         let mut matrix = Array2::zeros((self.n() - 1, self.n() - 1));
//         matrix[[0, 1]] = E::one();
//         matrix[[self.n(), self.n()-1]] = E::one() / (E::one() + E::one());
//
//         for ii in 1..self.n()-2 {
//             matrix[[ii, ii-1]] = E::one() / (E::one() + E::one());
//             matrix[[ii, ii+1]] = E::one() / (E::one() + E::one());
//         }
//
//         let mut b = Array1::zeros(self.n() + 1);
//
//         for ii in 2..(self.n()) {
//             let jj = self.n() - ii;
//             b[jj] = b[jj+2] + (E::one() + E::one()) * (E::from(ii).unwrap() + E::one()) * self.coefficients[jj + 1];
//         }
//
//         let fac = E::one() / (b[self.n() - 2] + b[self.n() - 2]);
//         let b = b.slice(s![..(self.n()-2)]).to_owned();
//
//         let mut aux = Array1::zeros(self.n()-1);
//         aux[self.n() - 2] = E::one();
//
//         let mb = outer_product(&aux, &b).unwrap();
//
//         matrix = matrix - mb.mapv(|x| fac * x);
//
//
//         matrix
//     }
// }
//
// pub struct Values<E, const M: usize> {
//     values: [E; M],
//     standard_uncertainty: Option<[E; M]>,
//     covariance: Option<[[E; M]; M]>,
// }
//
pub type Result<T> = ::std::result::Result<T, Box<dyn ::std::error::Error>>;
//
// pub enum Problem<E, const M: usize> {
//     Exact { x: [E; M], y: [E; M] },
//     ResponseUncertain { x: [E; M], y: [E; M], uy: [E; M] },
//     BothUncertain{ x: [E; M], ux: [E; M], y: [E; M], uy: [E; M] },
//     ResponseCovariance { x: [E; M], y: [E; M], vy: [[E; M]; M] },
//     BothCovariance { x: [E; M], vx: [[E; M]; M], y: [E; M], vy: [[E; M]; M] },
// }
//
// impl<E: Float + FromPrimitive, const M: usize> Problem<E, M> {
//     fn try_fit(&self, n_max: usize) -> Result<PolynomialFit<E>> {
//         let mut fits = vec![];
//         for n in 1..n_max {
//             let fit = self.fit(n)?;
//
//             // If the fit is monotonic we retain it, if not we move onto the next one
//             if fit.is_monotonic() {
//                 fits.push(fit);
//             }
//         }
//
//         todo!()
//         // let best_fit = self.find_best_fit(fits, Criteria::AICc);
//
//         // match best_fit.essess_chi_2_obs() {
//         //     Ok(fit) => Ok(fit),
//         //     Err(e) => todo!(),
//         // }
//
//     }
//
//     fn fit(&self, n: usize) -> Result<PolynomialFit<E>> {
//         match self {
//             Self::Exact { x, y } => todo!(),
//             _ => unimplemented!("not implemented..."),
//         }
//     }
// }
//
// #[cfg(test)]
// mod tests {
//     use std::ops::Range;
//
//     use crate::PolynomialFit;
//
//
//     fn monotonic_polynomials_are_monotonic() {
//         let polynomial = PolynomialFit {
//             coefficients: vec![1.0, 1.0, 1.0],
//             covariance: vec![],
//             limits: Range { start: -1.0, end: 1.0 },
//         };
//     }
// }
