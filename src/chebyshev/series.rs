use super::{Basis, CSeries, PolynomialSeries};
use crate::Result;
use ndarray::{arr1, s, Array1, Array2, ScalarOperand};
use ndarray_linalg::{eig::EigVals, Lapack, Scalar};
use num_traits::float::FloatCore;
use std::ops::Range;

#[derive(Clone, Debug)]
pub struct Series<E> {
    pub(crate) coeff: CSeries<E>,
    pub(crate) domain: Range<E>,
    pub(crate) window: Range<E>,
    pub(crate) basis: Basis,
}

impl<E: Clone> Series<E> {
    pub(crate) fn coeff(&self) -> Vec<E> {
        self.coeff.inner().to_vec()
    }
}

impl<E: Scalar<Real = E>> std::ops::Mul for Series<E> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let coeff = self.coeff * rhs.coeff;
        let degree = coeff.len() - 1;
        Self {
            coeff,
            basis: Basis::new(degree),
            domain: self.domain,
            window: self.window,
        }
    }
}

impl<E: Scalar<Real = E>> std::ops::Add for Series<E> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let coeff = self.coeff + rhs.coeff;
        let degree = coeff.len() - 1;
        Self {
            coeff,
            basis: Basis::new(degree),
            domain: self.domain,
            window: self.window,
        }
    }
}

impl<E: Scalar<Real = E>> std::ops::Sub for Series<E> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let coeff = self.coeff - rhs.coeff;
        let degree = coeff.len() - 1;
        Self {
            coeff,
            basis: Basis::new(degree),
            domain: self.domain,
            window: self.window,
        }
    }
}

impl<E: Scalar<Real = E> + ScalarOperand + Lapack + FloatCore + PartialOrd> PolynomialSeries<E>
    for Series<E>
{
    fn degree(&self) -> usize {
        self.basis.degree()
    }

    fn domain(&self) -> Range<E> {
        self.domain.clone()
    }

    fn window(&self) -> Range<E> {
        self.window.clone()
    }

    fn null(domain: Range<E>, window: Range<E>) -> Self {
        Self {
            coeff: CSeries::from(vec![]),
            basis: Basis::new(0),
            domain,
            window,
        }
    }

    fn evaluate(&self, t: E) -> E {
        let mut coeffs = self.coeff.iter();
        println!("degree {}", self.degree());
        let (c0, c1) = if self.degree() < 2 {
            (
                coeffs.next().copied().unwrap_or_else(E::zero),
                coeffs.next().copied().unwrap_or_else(E::zero),
            )
        } else {
            let t2 = t + t;
            let mut coeffs = coeffs.rev();
            let mut c1 = coeffs.next().copied().unwrap();
            let mut c0 = coeffs.next().copied().unwrap();
            for cnext in coeffs {
                let tmp = c0;
                c0 = *cnext - c1;
                c1 = tmp + c1 * t2;
            }
            (c0, c1)
        };
        c0 + t * c1
    }

    fn first_derivative(&self) -> Self {
        let mut coeff_of_derivative = vec![E::zero(); self.degree()];
        let mut coeff = self.coeff.inner().to_vec();

        for jj in (2..=self.degree()).rev() {
            coeff_of_derivative[jj - 1] = E::from(2 * jj).unwrap() * coeff[jj];
            coeff[jj - 2] =
                coeff[jj - 2] + (E::from(jj).unwrap() * coeff[jj]) / E::from(jj - 2).unwrap();
        }

        if self.degree() > 1 {
            coeff_of_derivative[1] = E::from(4).unwrap() * coeff[2];
        }
        coeff_of_derivative[0] = coeff[1];

        Self {
            coeff: coeff_of_derivative.into(),
            basis: Basis::new(self.degree() - 1),
            domain: self.domain(),
            window: self.window(),
        }
    }

    fn roots(&self) -> Result<Vec<E>> {
        match self.degree() {
            0 => Ok(vec![]),
            1 => {
                let mut coeffs = self.coeff.iter();
                Ok(vec![
                    -coeffs.next().copied().unwrap() / coeffs.next().copied().unwrap(),
                ])
            }
            _ => self.real_eigenvalues_of_companion_matrix(),
        }
    }
}

impl<E> Series<E>
where
    E: Scalar<Real = E> + ScalarOperand + Lapack + FloatCore,
{
    fn real_eigenvalues_of_companion_matrix(&self) -> Result<Vec<E>> {
        let mut eigenvalues = self
            .companion_matrix()?
            .eigvals()?
            .into_iter()
            .filter(|z| z.im() == E::zero())
            .map(|z| z.re())
            .filter(|x| x.is_finite())
            .collect::<Vec<_>>();

        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(eigenvalues)
    }

    fn companion_matrix(&self) -> Result<Array2<E>> {
        if self.degree() == 0 {
            return Err("series must have degree of at least 1.".into());
        } else if self.degree() == 1 {
            let mut coeffs = self.coeff.iter();
            return Ok(Array2::from_diag_elem(
                1,
                coeffs.next().copied().unwrap() / coeffs.next().copied().unwrap(),
            ));
        }

        let mut companion_matrix = Array1::zeros(self.degree() * self.degree());
        let c = arr1(&self.coeff());

        let mut scl = vec![E::one()];
        scl.extend(
            std::iter::repeat((E::one() / (E::one() + E::one())).sqrt()).take(self.degree() - 1),
        );
        let scl = arr1(&scl);

        let mut top = companion_matrix.slice_mut(s![1..;self.degree()+1]);
        top += E::one() / (E::one() + E::one());
        top[0] = (E::one() / (E::one() + E::one())).sqrt();

        let mut bottom = companion_matrix.slice_mut(s![self.degree()..;self.degree()+1]);
        bottom += E::one() / (E::one() + E::one());
        bottom[0] = (E::one() / (E::one() + E::one())).sqrt();

        let mut companion_matrix = companion_matrix.into_shape((self.degree(), self.degree()))?;

        let curr_rcol = companion_matrix.slice(s![.., self.degree() - 1]).to_owned();
        companion_matrix
            .slice_mut(s![.., self.degree() - 1])
            .assign(
                &(curr_rcol
                    - c.slice(s![..self.degree()])
                        .mapv(|v| v / (E::one() + E::one()) / c[self.degree()])
                        * scl.mapv(|v| v / scl[self.degree() - 1])),
            );

        Ok(companion_matrix)
    }
}

#[cfg(test)]
mod test {
    use super::super::{ChebyshevBuilder, PolynomialSeries};
    use super::Series;
    use ndarray_rand::rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;
    use std::ops::Range;

    #[test]
    fn chebyshev_series_multiply_correctly() {
        let polynomial_a = ChebyshevBuilder::new(3)
            .with_coefficients(vec![1., 2., 3.])
            .on_domain(Range {
                start: -1.,
                end: 1.,
            })
            .build();
        let polynomial_b = ChebyshevBuilder::new(3)
            .with_coefficients(vec![3., 2., 1.])
            .on_domain(Range {
                start: -1.,
                end: 1.,
            })
            .build();

        let expected = [6.5, 12., 12., 4., 1.5];

        let result = polynomial_a * polynomial_b;
        for (exp, res) in expected.iter().zip(result.coeff.iter()) {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

    #[test]
    fn null_chebyshev_series_is_evaluated_correctly() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let domain_max: f64 = rng.gen::<f64>().abs();
        let series = Series::null(
            Range {
                start: -domain_max,
                end: domain_max,
            },
            Range {
                start: -1.,
                end: 1.,
            },
        );
        let val = series.evaluate(rng.gen());
        approx::assert_relative_eq!(val, 0.0);
    }

    #[test]
    fn degree_zero_chebyshev_series_is_evaluated_correctly() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let end: f64 = rng.gen::<f64>().abs();
        let start = -end;
        let degree = 0;
        let coeff = rng.gen();
        let series = ChebyshevBuilder::new(degree)
            .with_coefficients(vec![coeff])
            .on_domain(Range { start, end })
            .build();

        let val = series.evaluate(rng.gen());
        approx::assert_relative_eq!(val, coeff);
    }

    #[test]
    fn degree_one_chebyshev_series_is_evaluated_correctly() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let end: f64 = rng.gen::<f64>().abs();
        let start = -end;
        let degree = 1;
        let coeff: Vec<f64> = (0..=degree).map(|_| rng.gen()).collect();

        let series = ChebyshevBuilder::new(degree)
            .with_coefficients(coeff.clone())
            .on_domain(Range { start, end })
            .build();

        let t0 = rng.gen();
        let actual = series.evaluate(t0);

        let expected = t0.mul_add(coeff[1], coeff[0]);
        approx::assert_relative_eq!(expected, actual);
    }

    #[test]
    fn degree_two_chebyshev_series_is_evaluated_correctly() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let end: f64 = rng.gen::<f64>().abs();
        let start = -end;
        let degree = 2;
        let coeff: Vec<f64> = (0..=degree).map(|_| rng.gen()).collect();

        let series = ChebyshevBuilder::new(degree)
            .with_coefficients(coeff.clone())
            .on_domain(Range { start, end })
            .build();

        let t0 = rng.gen();
        let actual = series.evaluate(t0);

        let expected = coeff[2].mul_add(
            2.0f64.mul_add(t0.powi(2), -1.),
            t0.mul_add(coeff[1], coeff[0]),
        );
        approx::assert_relative_eq!(expected, actual);
    }

    #[test]
    fn degree_three_chebyshev_series_is_evaluated_correctly() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let end: f64 = rng.gen::<f64>().abs();
        let start = -end;
        let degree = 3;
        let coeff: Vec<f64> = (0..=degree).map(|_| rng.gen()).collect();

        let series = ChebyshevBuilder::new(degree)
            .with_coefficients(coeff.clone())
            .on_domain(Range { start, end })
            .build();

        let t0 = rng.gen();
        let actual = series.evaluate(t0);

        let expected = coeff[3].mul_add(
            4.0f64.mul_add(t0.powi(3), -3. * t0),
            coeff[2].mul_add(
                2.0f64.mul_add(t0.powi(2), -1.),
                t0.mul_add(coeff[1], coeff[0]),
            ),
        );
        approx::assert_relative_eq!(expected, actual);
    }

    #[test]
    fn degree_four_chebyshev_series_is_evaluated_correctly() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let end: f64 = rng.gen::<f64>().abs();
        let start = -end;
        let degree = 4;
        let coeff: Vec<f64> = (0..=degree).map(|_| rng.gen()).collect();

        let series = ChebyshevBuilder::new(degree)
            .with_coefficients(coeff.clone())
            .on_domain(Range { start, end })
            .build();

        let t0 = rng.gen();
        let actual = series.evaluate(t0);

        let expected = coeff[4].mul_add(
            8.0f64.mul_add(t0.powi(4), -8. * t0.powi(2)) + 1.,
            coeff[3].mul_add(
                4.0f64.mul_add(t0.powi(3), -3. * t0),
                coeff[2].mul_add(
                    2.0f64.mul_add(t0.powi(2), -1.),
                    t0.mul_add(coeff[1], coeff[0]),
                ),
            ),
        );
        approx::assert_relative_eq!(expected, actual);
    }

    #[test]
    fn first_order_chebyshev_derivative_is_correct() {
        let series = ChebyshevBuilder::new(3)
            .with_coefficients(vec![1., 2., 3., 4.])
            .on_domain(Range {
                start: -1.,
                end: 1.,
            })
            .build();

        let result = series.first_derivative().coeff();

        let expected = [14., 12., 24.];

        for (exp, res) in expected.iter().zip(result) {
            approx::assert_relative_eq!(*exp, res);
        }
    }
}
