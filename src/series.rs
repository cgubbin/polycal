use ndarray::{s, Array1, ArrayView1, Axis};
use ndarray_linalg::Scalar;

use crate::ChebyshevPolynomial;

#[derive(Clone, Debug)]
struct CSeries<E>(Array1<E>);
#[derive(Clone, Debug)]
struct ZSeries<E>(Array1<E>);

impl<E: Scalar<Real = E>> From<CSeries<E>> for ZSeries<E> {
    fn from(value: CSeries<E>) -> Self {
        let n = value.0.len();
        let mut z = Array1::zeros(2 * n - 1);
        z.slice_mut(s![n - 1..])
            .assign(&value.0.mapv(|c| c / (E::one() + E::one())));
        Self(z.clone() + z.slice(s![..;-1]))
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Mode {
    Full,
    Same,
}

fn convolve<E: Scalar<Real = E>>(data: ArrayView1<'_, E>, window: ArrayView1<'_, E>, mode: Mode) -> Array1<E> {
    let mut window = if mode == Mode::Full {
        let mut padded_window = Array1::zeros(data.len() / 2 + data.len() / 2 + window.len());
        padded_window.slice_mut(s![data.len()/2..data.len()/2 + window.len()])
            .assign(&window);
        padded_window
    } else {
        window.to_owned()
    };

    let mut data = data.to_owned();

    if window.len() > data.len() {
        std::mem::swap(&mut window, &mut data);
    }

    let mut padded = Array1::zeros(window.len() / 2 + window.len() / 2 + data.len());
    padded.slice_mut(s![window.len()/2..window.len()/2 + data.len()])
        .assign(&data);

    let mut w = window.view();
    w.invert_axis(Axis(0));

    padded
        .axis_windows(Axis(0), w.len())
        .into_iter()
        .map(|x| (&x * &w).sum())
        .collect()
}

impl<E: Scalar<Real=E>> std::ops::Mul for ZSeries<E> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(convolve(self.0.view(), rhs.0.view(), Mode::Full))
    }
}

impl<E: Scalar<Real = E>> From<ZSeries<E>> for CSeries<E> {
    fn from(value: ZSeries<E>) -> Self {
        let n = (value.0.len() + 1) / 2;
        let mut c = value.0.slice(s![n-1..]).to_owned().mapv(|c| c * (E::one() + E::one()));
        c[0] /= E::one() + E::one();
        Self(c)
    }
}

impl<E: Scalar<Real = E>> CSeries<E> {
    fn trimmed(self) -> Array1<E> {
        if self.0.is_empty()
            || *self.0.last().unwrap() != E::zero() {
            return self.0
        }

        for (ii, ele) in self.0.iter().rev().enumerate() {
            if *ele != E::zero() {
                return self.0.slice(s![0..self.0.len()-1-ii]).to_owned();
            }
        }
        Array1::zeros(0)
    }
}

impl<E: Scalar<Real=E>> std::ops::Mul for ChebyshevPolynomial<E> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs_z_series = ZSeries::from(CSeries(self.coeff.into()));
        let rhs_z_series = ZSeries::from(CSeries(rhs.coeff.into()));
        let mul_z_series = lhs_z_series * rhs_z_series;
        let mul_c_series = CSeries::from(mul_z_series);
        Self {
            coeff: mul_c_series.trimmed().to_vec(),
            domain: self.domain,
            window: self.window
        }
    }
}

#[cfg(test)]
mod test {
    use ndarray::{array, Array1};
    use std::ops::Range;
    use super::{convolve, CSeries, Mode, ZSeries, ChebyshevPolynomial};

    #[test]
    fn convolve_odd_odd() {
        let data = array![1., 2., 3.];
        let window = array![0., 1., 0.5];
        let expected = array![1., 2.5, 4.];
        for (exp, res) in expected.iter().zip(&convolve(data.view(), window.view(), Mode::Same)) {

            approx::assert_relative_eq!(*exp, *res);
        }
    }

        #[test]
    fn convolve_odd_odd2() {
        let data = array![1., 2., 3., 4., 5.];
        let window = array![2., 1., 0., 1., 0.5];
        let result = convolve(data.view(), window.view(), Mode::Same);
        let expected = array![8., 12., 16.5, 9., 5.5];

        for (exp, res) in expected.iter().zip(&result) {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

    #[test]
    fn convolve_even_odd() {
        let data = array![1., 2., 3., 4.];
        let window = array![0., 1., 0.5];
        let expected = array![1., 2.5, 4., 5.5];

        for (exp, res) in expected.iter().zip(&convolve(data.view(), window.view(), Mode::Same)) {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

    #[test]
    fn convolve_even_even() {
        let data = array![1., 2., 3., 4.];
        let window = array![1., 0.5];
        let expected = array![1., 2.5, 4., 5.5];

        for (exp, res) in expected.iter().zip(&convolve(data.view(), window.view(), Mode::Same)) {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

    #[test]
    fn convolve_even_even2() {
        let data = array![1., 2., 3., 4.];
        let window = array![1., 0., 1., 0.5];
        let result = convolve(data.view(), window.view(), Mode::Same);
        let expected = array![2., 4., 6.5, 4.];

        for (exp, res) in expected.iter().zip(&result) {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

    #[test]
    fn convolve_odd_even() {
        let data = array![1., 2., 3., 4., 5.];
        let window = array![1., 0.5];
        let expected = array![1., 2.5, 4., 5.5, 7.];

        for (exp, res) in expected.iter().zip(&convolve(data.view(), window.view(), Mode::Same)) {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

    #[test]
    fn convolve_bigger_window() {
        let data = array![1., 2., 3.];
        let window = array![1., 0., 1., 0.5];
        let result = convolve(data.view(), window.view(), Mode::Same);
        let expected = array![2., 4., 2.5, 4.];

        for (exp, res) in expected.iter().zip(&result) {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

    #[test]
    fn convolve_large_window() {
        let data = array![0.5, 0., 0.5];
        let window = array![0.004580594210830624, 0.019638914128155882, 0.12406477973236824, 0.3805431212757002, 0.6263732815125124, 0.3805431212757002, 0.12406477973236824, 0.019638914128155882, 0.004580594210830624];
        let result = convolve(data.view(), window.view(), Mode::Full);
        let expected = array![0.0022903 , 0.00981946, 0.06432269, 0.20009102, 0.37521903,
       0.38054312, 0.37521903, 0.20009102, 0.06432269, 0.00981946,
       0.0022903];
        dbg!(&result);

        for (exp, res) in expected.iter().zip(&result) {
            approx::assert_relative_eq!(*exp, *res, max_relative=1e-4);
        }
    }

    #[test]
    fn z_series_multiply_correctly() {
        let series_a = ZSeries(Array1::from_vec(vec![1., 2., 3., 4., 5.]));
        let series_b = ZSeries(Array1::from_vec(vec![12., 13., 14., 15., 16.]));
        let expected = [12., 37., 76., 130., 200., 198., 178., 139., 80.];

        let result = series_a * series_b;

        for (exp, res) in expected.iter().zip(&result.0) {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

    #[test]
    fn z_series_roundtrip_works() {
        let z_series = ZSeries(Array1::from_vec(vec![1.5, 1., 1., 1., 1.5]));
        let c_series = CSeries::from(z_series.clone());
        let result = ZSeries::from(c_series);
        for (exp, res) in z_series.0.iter().zip(&result.0) {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

    #[test]
    fn c_series_roundtrip_works() {
        let c_series = CSeries(Array1::from_vec(vec![1., 2., 3.]));
        let z_series = ZSeries::from(c_series.clone());
        let result = CSeries::from(z_series);
        for (exp, res) in c_series.0.iter().zip(&result.0) {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

    #[test]
    fn chebyshev_multiply_correctly() {
        let poly_a = ChebyshevPolynomial {
            coeff: vec![1., 2., 3.],
            domain: Range { start: -1., end: 1. },
            window: Range { start: -1., end: 1. },
        };
        let poly_b = ChebyshevPolynomial {
            coeff: vec![3., 2., 1.],
            domain: Range { start: -1., end: 1. },
            window: Range { start: -1., end: 1. },
        };

        let expected = [  6.5,  12. ,  12. ,   4. ,   1.5];

        let result = poly_a * poly_b;
        for (exp, res) in expected.iter().zip(&result.coeff) {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

}
