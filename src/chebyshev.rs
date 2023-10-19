use std::ops::{Range, AddAssign};

use argmin::{core::{Gradient, Hessian, CostFunction, ArgminFloat, Executor, observers::{SlogLogger, ObserverMode}}, solver::{linesearch::MoreThuenteLineSearch, newton::NewtonCG}};
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
    /// Evaluate the sum of the series using Clenshaw recursion
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

    /// Returns the vector of underlying Polynomials up to the max-order of `self`.
    pub(crate) fn underlying_polys(&self, t: E) -> Vec<E> {
        match self.n() {
            1 => vec![E::one()],
            2 => vec![E::one(), t],
            _ => {
                let mut vals = vec![E::one(),  t];
                for ii in 2..self.n() {
                    vals.push((E::one() + E::one()) * t * vals[ii - 1] - vals[ii - 2]);
                }
                vals
            }
        }
    }

    /// Compute the derivative of order `cnt` of the Chebyshev Polynomial
    ///
    /// ```
    fn deriv(&self, cnt: usize) -> ChebyshevPolynomial<E> {
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

    fn evaluate_q(&self, t: E) -> E {
        (E::one() + E::one()) / (self.domain.end - self.domain.start) * self.deriv(1).eval(t)

    }

    pub(crate) fn standard_uncertainty_direct(&self, t0: E, ux: E, cov: ArrayView2<'_, E>) -> E {
        let q = self.evaluate_q(t0);
        let g: Array1<E> = self.underlying_polys(t0).into();
        (q.powi(2) * ux.powi(2) + g.dot(&cov.dot(&g))).sqrt()
    }

    pub(crate) fn standard_uncertainty_inverse(&self, t0: E, uy: E, cov: ArrayView2<'_, E>) -> E {
        let q = self.evaluate_q(t0);
        let g: Array1<E> = self.underlying_polys(t0).into();
        E::one() / q.powi(2) * (uy.powi(2) + g.dot(&cov.dot(&g)))
    }
}

struct InverseProblem<'a, E> {
    problem: &'a ChebyshevPolynomial<E>,
    y0: E,
}

impl<'a, E: ArgminFloat + Scalar<Real = E>> CostFunction for InverseProblem<'a, E> {
    type Param = E;
    type Output = E;

    fn cost(&self, param: &Self::Param) -> ::std::result::Result<Self::Output, argmin::core::Error> {
        Ok(Scalar::abs(self.problem.eval(*param) - self.y0))
    }
}

impl<'a, E: ArgminFloat + Scalar<Real = E>> Gradient for InverseProblem<'a, E> {
    type Param = E;
    type Gradient = E;

    fn gradient(&self, param: &Self::Param) -> ::std::result::Result<Self::Gradient, argmin::core::Error> {
        Ok(self.problem.deriv(1).eval(*param))
    }
}

impl<'a, E: ArgminFloat + Scalar<Real = E>> Hessian for InverseProblem<'a, E> {
    type Param = E;
    type Hessian = E;

    fn hessian(&self, param: &Self::Param) -> ::std::result::Result<Self::Hessian, argmin::core::Error> {
        Ok(self.problem.deriv(2).eval(*param))
    }
}


impl<E> ChebyshevPolynomial<E>
where
    E: ArgminFloat + Scalar<Real = E> + argmin_math::ArgminSub<E, E> + argmin_math::ArgminAdd<E, E> + argmin_math::ArgminZeroLike + argmin_math::ArgminConj + argmin_math::ArgminMul<E, E> + argmin_math::ArgminL2Norm<E> + argmin_math::ArgminDot<E, E>,
{
    pub(crate) fn inverse_eval(&self, y0: E) -> Result<E> {
        let cost = InverseProblem { problem: &self, y0 };
        let init_param = E::zero();

        // set up line search
        let linesearch = MoreThuenteLineSearch::new();

        // Set up solver
        let solver = NewtonCG::new(linesearch);

        // Run solver
        let res = Executor::new(cost, solver)
            .configure(|state| state.param(init_param).max_iters(100))
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;

        let mut state = res.state().clone();
        let param = state.take_param();

        Ok(param.unwrap())
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
}

#[cfg(test)]
mod test {
    use ndarray_rand::rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;
    use super::ChebyshevPolynomial;
    use std::ops::Range;

    #[test]
    fn chebyshev_of_order_zero_is_evaluated_correctly() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let domain_max: f64 = rng.gen::<f64>().abs();
        let poly = ChebyshevPolynomial {
            coeff: vec![],
            domain: Range {
                start: -domain_max,
                end: domain_max,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };
        let val = poly.eval(rng.gen());
        assert_eq!(val, 0.0);
    }

    #[test]
    fn chebyshev_of_order_one_is_evaluated_correctly() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let domain_max: f64 = rng.gen::<f64>().abs();
        let coeff = rng.gen();
        let poly = ChebyshevPolynomial {
            coeff: vec![coeff],
            domain: Range {
                start: -domain_max,
                end: domain_max,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };
        let val = poly.eval(rng.gen());
        assert_eq!(val, coeff);
    }

    #[test]
    fn chebyshev_of_order_two_is_evaluated_correctly() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let domain_max: f64 = rng.gen::<f64>().abs();
        let order = 2;
        let coeff: Vec<f64> = (0..order).map(|_| rng.gen()).collect();

        let poly = ChebyshevPolynomial {
            coeff: coeff.clone(),
            domain: Range {
                start: -domain_max,
                end: domain_max,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };
        let t0 = rng.gen();
        let actual = poly.eval(t0);

        let expected = coeff[0] + t0 * coeff[1];
        assert_eq!(expected, actual);
    }


    #[test]
    fn chebyshev_of_order_three_is_evaluated_correctly() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let domain_max: f64 = rng.gen::<f64>().abs();
        let order = 3;
        let coeff: Vec<f64> = (0..order).map(|_| rng.gen()).collect();

        let poly = ChebyshevPolynomial {
            coeff: coeff.clone(),
            domain: Range {
                start: -domain_max,
                end: domain_max,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };
        let t0 = rng.gen();
        let actual = poly.eval(t0);

        let expected = coeff[0] + t0 * coeff[1] + coeff[2] * (2. * t0.powi(2) - 1.);
        assert_eq!(expected, actual);
    }

    #[test]
    fn vec_evaluation_sum_matches_scalar_eval() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let domain_max: f64 = rng.gen::<f64>().abs();
        let order = rng.gen::<i8>() as usize;
        let coeff: Vec<f64> = (0..order).map(|_| rng.gen()).collect();

        let poly = ChebyshevPolynomial {
            coeff: coeff.clone(),
            domain: Range {
                start: -domain_max,
                end: domain_max,
            },
            window: Range {
                start: -1.,
                end: 1.,
            },
        };

        let t0 = rng.gen();
        let scalar = poly.eval(t0);
        let vector = poly.underlying_polys(t0).into_iter()
            .zip(coeff)
            .fold(0., |a, (t, c)| a + t * c);

        approx::assert_relative_eq!(scalar, vector, max_relative=1e-10);
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

