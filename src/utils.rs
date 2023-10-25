use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::Scalar;
use std::ops::Range;

pub struct Rescaled<E> {
    pub(crate) t: Array1<E>,
    pub(crate) domain: Range<E>,
}

pub fn to_unscaled<E: Scalar>(x: E, Range { start, end }: &Range<E>) -> E {
    (x * (*end - *start) + *end + *start) / (E::one() + E::one())
}

pub fn to_scaled<E: Scalar>(x: E, Range { start, end }: &Range<E>) -> E {
    (x + x - *end - *start) / (*end - *start)
}

pub fn form_rescaled_variables<E: PartialOrd + Scalar>(x: ArrayView1<'_, E>) -> Rescaled<E> {
    let domain = find_limits(x.as_slice().unwrap());

    let t = x.into_iter().map(|&x| to_scaled(x, &domain)).collect();

    Rescaled { t, domain }
}

/// Returns the range spanning the minimium and maximum of the `variable`
///
/// # Panics
///
/// This function will panic if the variable contains NaN or infinities. It is the responsibility
/// of the caller to check and handle this error.
pub fn find_limits<E: Clone + PartialOrd>(variable: &[E]) -> Range<E> {
    let start = variable
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        .clone();
    let end = variable
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        .clone();

    Range { start, end }
}

pub fn outer_product<T: Scalar>(
    a: &Array1<T>,
    b: &Array1<T>,
) -> Result<Array2<T>, ndarray::ShapeError> {
    let a: Array2<T> = a.clone().into_shape((a.len(), 1))?;
    let b: Array2<T> = b.clone().into_shape((1, b.len()))?;

    Ok(ndarray::linalg::kron(&a, &b))
}

#[cfg(test)]
mod test {
    use ndarray_rand::rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;
    use std::ops::Range;

    use crate::utils::{to_scaled, to_unscaled};

    #[test]
    fn scaling_roundtrip_is_successful() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);

        let start: f64 = rng.gen();
        let end = rng.gen_range((2. * start)..(10. * start));

        let domain = Range { start, end };

        let input = rng.gen();
        let scaled = to_scaled(input, &domain);
        let output = to_unscaled(scaled, &domain);

        approx::assert_relative_eq!(input, output);

        let backward_output = to_scaled(output, &domain);
        approx::assert_relative_eq!(scaled, backward_output);
    }
}
