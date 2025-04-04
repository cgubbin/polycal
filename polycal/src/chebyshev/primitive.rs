//! Underlying series for Chebyshev representations of Polynomials
//!
//! A Chebyshev series is represented as a sum of Chebyshev polynomials weighted by scalar
//! coefficients
//! `p_n(x) = \sum_n c_n T_n(x),`
//! where the `c_n` form a C-series.
//!
//! A related quantity is the z-series. Z-series are often more useful for doing algebra on pairs
//! of Chebyshev series. This module implements conversions between z-series and c-series, and
//! implements fundamental operations on z-series allowing the higher order Chebyshev polynomials
//! to also implement those operations based on the c-series representation.

use ndarray::{s, Array1, ArrayView1, Axis};
use ndarray_linalg::Scalar;

/// A c-series represents the coefficients of a Chebyshev series
///
/// Coefficients are stored in ascending order of Chebyshev polynomial degree. The first element is
/// the scalar offset
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CSeries<E>(Array1<E>);

impl<E> CSeries<E> {
    // Create a new Chebyshev series from a vector of coefficients
    pub(crate) fn new(series: impl Into<Array1<E>>) -> Self {
        Self(series.into())
    }

    // Returns the length of the underlying series
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns a borrowed slice of the underlying coefficients
    pub(crate) fn inner(&self) -> ArrayView1<'_, E> {
        self.0.view()
    }
}

impl<E: Copy> CSeries<E> {
    pub(crate) fn iter(&self) -> std::slice::Iter<E> {
        self.0.as_slice().unwrap().iter()
    }
}

impl<E> From<Vec<E>> for CSeries<E> {
    fn from(value: Vec<E>) -> Self {
        Self(value.into())
    }
}

impl<E: Scalar<Real = E>> std::ops::Mul for CSeries<E> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // These is no simple multiplication implementation for C-Series, so this function converts
        // the series to z-series, and multiplies them
        let lhs_z_series = ZSeries::from(self);
        let rhs_z_series = ZSeries::from(rhs);
        let mul_z_series = lhs_z_series * rhs_z_series;
        // Covert back to a c-series and remove zero coefficients
        Self::from(mul_z_series).trimmed()
    }
}

impl<E: Scalar<Real = E>> std::ops::Add for CSeries<E> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match self.len().checked_sub(rhs.len()) {
            // If the lengths are equal, just add the two sets of coefficients
            Some(0) => Self(self.0 + rhs.0),
            // The lhs is longer
            Some(_) => {
                // pad the rhs with zeroes so both arrays are of equal length
                let mut padded_rhs = Array1::zeros(self.len());
                padded_rhs.slice_mut(s![..rhs.len()]).assign(&rhs.0);
                // And add
                Self(self.0 + padded_rhs)
            }
            // The rhs is longer
            None => {
                // pad the lhs with zeroes so both arrays are of equal length
                let mut padded_self = Array1::zeros(rhs.len());
                padded_self.slice_mut(s![..self.len()]).assign(&self.0);
                // And add
                Self(padded_self + rhs.0)
            }
        }
    }
}

impl<E: Scalar<Real = E>> std::ops::Sub for CSeries<E> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match self.len().checked_sub(rhs.len()) {
            Some(0) => Self(self.0 - rhs.0),
            Some(_) => {
                let mut padded_rhs = Array1::zeros(self.len());
                padded_rhs.slice_mut(s![..rhs.len()]).assign(&rhs.0);
                Self(self.0 - padded_rhs)
            }
            None => {
                let mut padded_self = Array1::zeros(rhs.len());
                padded_self.slice_mut(s![..self.len()]).assign(&self.0);
                Self(padded_self - rhs.0)
            }
        }
    }
}

impl<E: Scalar<Real = E>> From<ZSeries<E>> for CSeries<E> {
    fn from(value: ZSeries<E>) -> Self {
        let n = (value.0.len() + 1) / 2;
        let mut c = value
            .0
            .slice(s![n - 1..])
            .to_owned()
            .mapv(|c| c * (E::one() + E::one()));
        c[0] /= E::one() + E::one();
        Self(c)
    }
}

impl<E: Scalar<Real = E>> CSeries<E> {
    // Trims the polynomial to remove any zero-elements
    //
    // This is used after any add, or multiplication operations to reduce the order of the series
    // to the true order.
    pub(crate) fn trimmed(self) -> Self {
        // If the series is empty, or the last element is not zero, the series cannot be trimmed
        if self.0.is_empty() || *self.0.last().unwrap() != E::zero() {
            return self;
        }

        // Else iterate backwards through the series until the first non-zero element is found and
        // truncate the series at that point
        for (ii, ele) in self.0.iter().rev().enumerate() {
            if *ele != E::zero() {
                return Self(self.0.slice(s![0..self.0.len() - 1 - ii]).to_owned());
            }
        }

        // If the series is all zeroes, return a new series of zeroes of order 1
        Self(Array1::zeros(0))
    }
}

// An alternate z-series representation of the Chebyshev series.
//
// Z-series are used for multiplication of Chebyshev c-series, as they can be multiplied by a
// simple convolution
#[derive(Clone, Debug)]
pub struct ZSeries<E>(Array1<E>);

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

fn convolve<E: Scalar<Real = E>>(
    data: ArrayView1<'_, E>,
    window: ArrayView1<'_, E>,
    mode: Mode,
) -> Array1<E> {
    let mut window = if mode == Mode::Full {
        let mut padded_window = Array1::zeros(data.len() / 2 + data.len() / 2 + window.len());
        padded_window
            .slice_mut(s![data.len() / 2..data.len() / 2 + window.len()])
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
    padded
        .slice_mut(s![window.len() / 2..window.len() / 2 + data.len()])
        .assign(&data);

    let mut w = window.view();
    w.invert_axis(Axis(0));

    padded
        .axis_windows(Axis(0), w.len())
        .into_iter()
        .map(|x| (&x * &w).sum())
        .collect()
}

impl<E: Scalar<Real = E>> std::ops::Mul for ZSeries<E> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(convolve(self.0.view(), rhs.0.view(), Mode::Full))
    }
}

#[cfg(test)]
mod test {
    use super::{convolve, CSeries, Mode, ZSeries};
    use ndarray::{array, Array1};

    use ndarray_rand::rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    #[test]
    fn convolve_odd_odd() {
        let data = array![1., 2., 3.];
        let window = array![0., 1., 0.5];
        let expected = array![1., 2.5, 4.];
        for (exp, res) in expected
            .iter()
            .zip(&convolve(data.view(), window.view(), Mode::Same))
        {
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

        for (exp, res) in expected
            .iter()
            .zip(&convolve(data.view(), window.view(), Mode::Same))
        {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

    #[test]
    fn convolve_even_even() {
        let data = array![1., 2., 3., 4.];
        let window = array![1., 0.5];
        let expected = array![1., 2.5, 4., 5.5];

        for (exp, res) in expected
            .iter()
            .zip(&convolve(data.view(), window.view(), Mode::Same))
        {
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

        for (exp, res) in expected
            .iter()
            .zip(&convolve(data.view(), window.view(), Mode::Same))
        {
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
        let window = array![
            0.004_580_594_210_830_624,
            0.019_638_914_128_155_882,
            0.124_064_779_732_368_24,
            0.380_543_121_275_700_2,
            0.626_373_281_512_512_4,
            0.380_543_121_275_700_2,
            0.124_064_779_732_368_24,
            0.019_638_914_128_155_882,
            0.004_580_594_210_830_624
        ];
        let result = convolve(data.view(), window.view(), Mode::Full);
        let expected = array![
            0.002_290_3,
            0.009_819_46,
            0.064_322_69,
            0.200_091_02,
            0.375_219_03,
            0.380_543_12,
            0.375_219_03,
            0.200_091_02,
            0.064_322_69,
            0.009_819_46,
            0.002_290_3
        ];
        dbg!(&result);

        for (exp, res) in expected.iter().zip(&result) {
            approx::assert_relative_eq!(*exp, *res, max_relative = 1e-4);
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
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let n = rng.gen_range(5..20);

        let mut z_series = (0..n).map(|_| rng.gen()).collect::<Vec<f64>>();
        let clone_z_series = z_series.clone();
        z_series.extend(clone_z_series.into_iter().rev().skip(1));

        let z_series = ZSeries(Array1::from_vec(z_series));
        let c_series = CSeries::from(z_series.clone());
        let result = ZSeries::from(c_series);
        for (exp, res) in z_series.0.iter().zip(&result.0) {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

    #[test]
    fn c_series_roundtrip_works() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let n = rng.gen_range(5..20);
        let c_series = (0..n).map(|_| rng.gen()).collect::<Vec<f64>>();

        let c_series = CSeries(Array1::from_vec(c_series));
        let z_series = ZSeries::from(c_series.clone());
        let result = CSeries::from(z_series);
        for (exp, res) in c_series.0.iter().zip(&result.0) {
            approx::assert_relative_eq!(*exp, *res);
        }
    }

    #[test]
    fn c_series_of_equal_length_sum_correctly() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let n = rng.gen_range(5..20);

        let values_a = (0..n).map(|_| rng.gen()).collect::<Vec<f64>>();
        let values_b = (0..n).map(|_| rng.gen()).collect::<Vec<f64>>();

        let series_a = CSeries::from(values_a.clone());
        let series_b = CSeries::from(values_b.clone());

        let expected = values_a.into_iter().zip(values_b).map(|(a, b)| a + b);

        let result = series_a + series_b;

        for (exp, res) in expected.zip(result.iter()) {
            approx::assert_relative_eq!(exp, *res);
        }
    }

    #[test]
    fn c_series_of_equal_length_subtract_correctly() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let n = rng.gen_range(5..20);

        let values_a = (0..n).map(|_| rng.gen()).collect::<Vec<f64>>();
        let values_b = (0..n).map(|_| rng.gen()).collect::<Vec<f64>>();

        let series_a = CSeries::from(values_a.clone());
        let series_b = CSeries::from(values_b.clone());

        let expected = values_a.into_iter().zip(values_b).map(|(a, b)| a - b);

        let result = series_a - series_b;

        for (exp, res) in expected.zip(result.iter()) {
            approx::assert_relative_eq!(exp, *res);
        }
    }

    #[test]
    fn c_series_of_different_lengths_sum_correctly_when_lhs_is_largest() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let n1 = rng.gen_range(5..20);

        let mut n2 = n1;

        while n1 == n2 {
            n2 = rng.gen_range(5..20);
        }

        let na = n1.max(n2);
        let nb = n1.min(n2);

        let values_a = (0..na).map(|_| rng.gen()).collect::<Vec<f64>>();
        let values_b = (0..nb).map(|_| rng.gen()).collect::<Vec<f64>>();

        let series_a = CSeries::from(values_a.clone());
        let series_b = CSeries::from(values_b.clone());

        let expected = values_a
            .into_iter()
            .zip(values_b.into_iter().chain(std::iter::repeat(0.)))
            .map(|(a, b)| a + b);

        let result = series_a + series_b;

        for (exp, res) in expected.zip(result.iter()) {
            approx::assert_relative_eq!(exp, *res);
        }
    }

    #[test]
    fn c_series_of_different_lengths_sum_correctly_when_rhs_is_largest() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let n1 = rng.gen_range(5..20);

        let mut n2 = n1;

        while n1 == n2 {
            n2 = rng.gen_range(5..20);
        }

        let nb = n1.max(n2);
        let na = n1.min(n2);

        let values_a = (0..na).map(|_| rng.gen()).collect::<Vec<f64>>();
        let values_b = (0..nb).map(|_| rng.gen()).collect::<Vec<f64>>();

        let series_a = CSeries::from(values_a.clone());
        let series_b = CSeries::from(values_b.clone());

        let expected = values_a
            .into_iter()
            .chain(std::iter::repeat(0.))
            .zip(values_b)
            .map(|(a, b)| a + b);

        let result = series_a + series_b;

        for (exp, res) in expected.zip(result.iter()) {
            approx::assert_relative_eq!(exp, *res);
        }
    }

    #[test]
    fn c_series_of_different_lengths_subtract_correctly_when_lhs_is_largest() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let n1 = rng.gen_range(5..20);

        let mut n2 = n1;

        while n1 == n2 {
            n2 = rng.gen_range(5..20);
        }

        let na = n1.max(n2);
        let nb = n1.min(n2);

        let values_a = (0..na).map(|_| rng.gen()).collect::<Vec<f64>>();
        let values_b = (0..nb).map(|_| rng.gen()).collect::<Vec<f64>>();

        let series_a = CSeries::from(values_a.clone());
        let series_b = CSeries::from(values_b.clone());

        let expected = values_a
            .into_iter()
            .zip(values_b.into_iter().chain(std::iter::repeat(0.)))
            .map(|(a, b)| a - b);

        let result = series_a - series_b;

        for (exp, res) in expected.zip(result.iter()) {
            approx::assert_relative_eq!(exp, *res);
        }
    }

    #[test]
    fn c_series_of_different_lengths_subtract_correctly_when_rhs_is_largest() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let n1 = rng.gen_range(5..20);

        let mut n2 = n1;

        while n1 == n2 {
            n2 = rng.gen_range(5..20);
        }

        let nb = n1.max(n2);
        let na = n1.min(n2);

        let values_a = (0..na).map(|_| rng.gen()).collect::<Vec<f64>>();
        let values_b = (0..nb).map(|_| rng.gen()).collect::<Vec<f64>>();

        let series_a = CSeries::from(values_a.clone());
        let series_b = CSeries::from(values_b.clone());

        let expected = values_a
            .into_iter()
            .chain(std::iter::repeat(0.))
            .zip(values_b)
            .map(|(a, b)| a - b);

        let result = series_a - series_b;

        for (exp, res) in expected.zip(result.iter()) {
            approx::assert_relative_eq!(exp, *res);
        }
    }
}
