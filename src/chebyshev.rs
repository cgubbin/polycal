use std::ops::{Range, AddAssign};

use ndarray::{arr1, s, Array1, Array2, ScalarOperand, ArrayView2};
use ndarray_linalg::{EigVals, Scalar, Lapack};

use crate::Result;

#[derive(Clone, Debug)]
pub(crate) struct ChebyshevPolynomial<E> {
    pub(crate) coeff: Vec<E>,
    pub(crate) domain: Range<E>,
    pub(crate) window: Range<E>,
}

impl<E> ChebyshevPolynomial<E>
{
    pub(crate) fn n(&self) -> usize {
        self.coeff.len()
    }
}

impl<E: Clone> ChebyshevPolynomial<E>
{
    fn coeff(&self) -> Vec<E> {
        self.coeff.clone()
    }
}

impl<E: PartialOrd + Scalar<Real = E>> ChebyshevPolynomial<E>
{

    pub(crate) fn constant(n: usize) -> Self {
        Self {
            coeff: vec![E::one(); n],
            domain: Range { start: -E::one(), end: E::one() },
            window: Range { start: -E::one(), end: E::one() },
        }
    }
}

impl<E: Scalar<Real = E>> ChebyshevPolynomial<E>
{
    pub(crate) fn eval(&self, t: E) -> E {
        let (c0, c1) = match self.n() < 3 {
            true => (
                self.coeff.get(0).copied().unwrap_or(E::zero()),
                self.coeff.get(1).copied().unwrap_or(E::zero()),
            ),
            _ => {
                let t2 = t + t;
                let mut c0 = self.coeff[self.n() - 2];
                let mut c1 = self.coeff[self.n() - 1];
                for i in 3..(self.n() + 1) {
                    let tmp = c0;
                    c0 = self.coeff[self.n() - i] - c1;
                    c1 = tmp + c1 * t2;
                }
                (c0, c1)
            }
        };
        c0 + t * c1
    }

    pub(crate) fn eval_u(&self, t: E) -> E {
        let (c0, c1) = match self.n() < 3 {
            true => (
                self.coeff.get(0).copied().unwrap_or(E::zero()),
                self.coeff.get(1).copied().unwrap_or(E::zero()),
            ),
            _ => {
                let t2 = t + t;
                let mut c0 = self.coeff[self.n() - 2];
                let mut c1 = self.coeff[self.n() - 1];
                for i in 3..(self.n() + 1) {
                    let tmp = c0;
                    c0 = self.coeff[self.n() - i] - c1;
                    c1 = tmp + c1 * t2;
                }
                (c0, c1)
            }
        };
        c0 + (t + t) * c1
    }

    pub(crate) fn eval_as_vec(&self, t: E) -> Vec<E> {
        match self.n() {
            1 => vec![self.coeff[0]],
            2 => vec![self.coeff[0], self.coeff[1] * t],
            _ => {
                let mut vals = vec![self.coeff[0], self.coeff[1] * t];
                for ii in 2..self.n() {
                    vals.push((E::one() + E::one()) * t * vals[ii - 1] - vals[ii - 2]);
                }
                vals
            }
        }
    }

    /// Evaluate `q`
    ///
    /// This is the inverse of the domain, multiplied by the first derivative of the series in the
    /// scaled coordinate system.
    ///
    /// The first derivative of a Chebyshev function of the first-kind is d_t T_n(t) = n
    /// U_{n-1}(t), so we calculate a vector of Chebyshev function of the second kind (U_n), then
    /// sum the product multiplied by the coefficients of the series.
    fn evaluate_q(&self, t: E) -> E {
        let u_vec = match self.n() {
            1 => vec![E::one()],
            2 => vec![E::one(), t + t],
            _ => {
                let mut vals = vec![E::one(), t + t];
                for ii in 2..self.n() {
                    vals.push((E::one() + E::one()) * t * vals[ii - 1] - vals[ii - 2]);
                }
                vals
            }
        };

        (E::one() + E::one()) / (self.domain.end - self.domain.start)
            * u_vec.into_iter().zip(&self.coeff).enumerate()
                .fold(E::zero(), |a, (ii, (un, ai))| a + E::from(ii + 1).unwrap() * un * *ai)

    }

    pub(crate) fn standard_uncertainty_direct(&self, t0: E, ux: E, cov: ArrayView2<'_, E>) -> E {
        let q = self.evaluate_q(t0);
        let g: Array1<E> = self.eval_as_vec(t0).into();
        (q.powi(2) * ux.powi(2) + g.dot(&cov.dot(&g))).sqrt()
    }

    pub(crate) fn standard_uncertainty_inverse(&self, t0: E, uy: E, cov: ArrayView2<'_, E>) -> E {
        let q = self.evaluate_q(t0);
        let g: Array1<E> = self.eval_as_vec(t0).into();
        E::one() / q.powi(2) * (uy.powi(2) + g.dot(&cov.dot(&g)))
    }
}

impl<E> ChebyshevPolynomial<E> {
    pub(crate) fn inverse_eval(&self, y0: E) -> E {
        todo!()
    }
}

impl<E: ScalarOperand + AddAssign + Scalar<Real = E> + Lapack + PartialOrd> ChebyshevPolynomial<E>
{

    pub(crate) fn is_monotonic(&self) -> Result<bool> {
        let deriv = self.deriv(1);
        let roots = deriv.roots()?;


        Ok(roots.into_iter()
            .all(|root| (root < - E::one()) || (root > E::one())))
    }

    /// Compute the derivative of order `cnt` of the Chebyshev Polynomial
    ///
    /// ```
    fn deriv(&self, cnt: usize) -> ChebyshevPolynomial<E> {
        dbg!(&cnt);
        dbg!(&self.n());
        match cnt {
            0 => self.clone(),
            cnt if cnt >= self.n() => Self {
                coeff: vec![],
                domain: self.domain.clone(),
                window: self.window.clone(),
            },
            cnt => {
                let mut n = self.n();
                let mut c = self.coeff();
                let mut coeff = None;
                for _ in 0..cnt {
                    n -= 1;
                    let mut der = vec![E::zero(); n];
                    for jj in (2..=n).rev() {
                        der[jj - 1] = E::from(2 * jj).unwrap() * c[jj];
                        c[jj - 2] =
                            c[jj - 2] + (E::from(jj).unwrap() * c[jj]) / E::from(jj - 2).unwrap();
                    }

                    if n > 1 {
                        der[1] = E::from(4).unwrap() * c[2];
                    }
                    der[0] = c[1];
                    c = der.clone();
                    coeff = Some(der);
                }

                Self {
                    coeff: coeff.unwrap(),
                    domain: self.domain.clone(),
                    window: self.window.clone(),
                }
            }
        }
    }

    fn roots(&self) -> Result<Vec<E>> {
        match self.n() {
            n if n < 2 => Ok(vec![]),
            n if n == 2 => Ok(vec![-self.coeff[0] / self.coeff[1]]),
            _ => {
                let m = self.companion_matrix()?;
                let mut r = m.eigvals()?.into_iter()
                    .filter(|x| x.im() == E::zero())
                    .map(|x| x.re())
                    .collect::<Vec<_>>();

                r.sort_by(|a, b| a.partial_cmp(b).unwrap());

                Ok(r)
            }
        }
    }

    fn companion_matrix(&self) -> Result<Array2<E>> {
        if self.n() < 2 {
            return Err("series must have maximum degree of at least 1.".into());
        }
        if self.n() == 2 {
            return Ok(Array2::from_diag_elem(1, self.coeff[0] / self.coeff[1]));
        }

        let n = self.n() - 1;
        let mut mat = Array1::zeros(n * n);

        let c = arr1(&self.coeff());

        let mut scl = vec![E::one()];
        scl.extend(std::iter::repeat((E::one()/(E::one() + E::one())).sqrt()).take(n-1));
        let scl = arr1(&scl);

        let mut top = mat.slice_mut(s![1..;n+1]);
        top +=  E::one() / (E::one() + E::one());
        top[0] = (E::one() / (E::one() + E::one())).sqrt();


        let mut bot = mat.slice_mut(s![n..;n+1]);
        bot +=  E::one() / (E::one() + E::one());
        bot[0] = (E::one() / (E::one() + E::one())).sqrt();

        let mut mat = mat.into_shape((n, n))?;

        let curr_rcol = mat.slice(s![.., n-1]).to_owned();
        mat.slice_mut(s![.., n-1])
             .assign(&(curr_rcol - c.slice(s![..n]).mapv(|v| v / (E::one() + E::one()) / c[n]) * scl.mapv(|v| v / scl[n - 1])));

        Ok(mat)
    }
    // fn generate(&self, t: E) -> Self {
    //     let mut terms = [E::zero(); N];
    //     terms[0] = E::one();
    //     terms[1] = t;
    //
    //     for ii in 2..N {
    //         terms[ii] = (E::one() + E::one()) * t * terms[ii - 1] - terms[ii-2];
    //     }
    //     Self { terms }
    // }
}

#[cfg(test)]
mod test {
    use super::ChebyshevPolynomial;
    use std::ops::Range;

    #[test]
    fn chebyshev_of_order_zero_is_evaluated_correctly() {
        let poly = ChebyshevPolynomial {
            coeff: vec![],
            domain: Range {
                start: -1.,
                end: 1.,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };
        let val = poly.eval(1.0);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn chebyshev_of_order_one_is_evaluated_correctly() {
        let poly = ChebyshevPolynomial {
            coeff: vec![1.0],
            domain: Range {
                start: -1.,
                end: 1.,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };
        let val = poly.eval(1.0);
        assert_eq!(val, 1.0);
    }

    #[test]
    fn chebyshev_of_order_three_is_evaluated_correctly() {
        let poly = ChebyshevPolynomial {
            coeff: vec![1.0, 2.0],
            domain: Range {
                start: -1.,
                end: 1.,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };
        let val = poly.eval(3.0);
        assert_eq!(val, 7.0);
    }




    #[test]
    fn first_chebyshev_derivative_is_correct() {
        let poly = ChebyshevPolynomial {
            coeff: vec![1., 2., 3., 4.],
            domain: Range {
                start: -1.,
                end: 1.,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };
        let deriv = poly.deriv(1);
        assert_eq!(deriv.coeff(), vec![14., 12., 24.]);
    }

    #[test]
    fn third_chebyshev_derivative_is_correct() {
        let poly = ChebyshevPolynomial {
            coeff: vec![1., 2., 3., 4.],
            domain: Range {
                start: -1.,
                end: 1.,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };
        let deriv = poly.deriv(3);
        assert_eq!(deriv.coeff(), vec![96.]);
    }

    #[test]
    fn chebyshev_roots_are_correct_for_order_2() {
        let poly = ChebyshevPolynomial {
            coeff: vec![1., 2.],
            domain: Range {
                start: -1.,
                end: 1.,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };
        let roots = poly.roots().unwrap();


        assert_eq!(1, roots.len());
        approx::assert_relative_eq!(-0.5, roots[0]);
    }

    #[test]
    fn chebyshev_roots_are_correct_for_order_3() {
        let poly = ChebyshevPolynomial {
            coeff: vec![1., 2., 3.],
            domain: Range {
                start: -1.,
                end: 1.,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };
        let roots = poly.roots().unwrap();


        approx::assert_relative_eq!((-1. - 13f64.sqrt()) / 6.0, roots[0]);
        approx::assert_relative_eq!((-1. + 13f64.sqrt()) / 6.0, roots[1]);
    }

    #[test]
    fn chebyshev_roots_are_correct_for_order_5() {
        let poly = ChebyshevPolynomial {
            coeff: vec![1., 2., 3., 4., 5.],
            domain: Range {
                start: -1.,
                end: 1.,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };
        let roots = poly.roots().unwrap();


        assert_eq!(4, roots.len());
        approx::assert_relative_eq!(-0.93158818, roots[0], max_relative=1e-5);
        approx::assert_relative_eq!(-0.5, roots[1]);
        approx::assert_relative_eq!(0.19171356, roots[2], max_relative=1e-5);
        approx::assert_relative_eq!(0.83987462, roots[3], max_relative=1e-5);
    }

    #[test]
    fn chebyshev_roots_are_correct_for_order_4() {
        let poly = ChebyshevPolynomial {
            coeff: vec![-1., 1., -1., 1.],
            domain: Range {
                start: -1.,
                end: 1.,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };
        let roots = poly.roots().unwrap();


        assert_eq!(3, roots.len());
        approx::assert_relative_eq!(-0.5, roots[0], max_relative=1e-5);
        approx::assert_relative_eq!(0.0, roots[1], max_relative=1e-5);
        approx::assert_relative_eq!(1.0, roots[2], max_relative=1e-5);
    }
}

