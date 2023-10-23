use ndarray::{Array1, ArrayView1};
use ndarray_linalg::Scalar;
use std::ops::Range;

pub(crate) struct Rescaled<E> {
    pub(crate) t: Array1<E>,
    pub(crate) domain: Range<E>,
}

pub(crate) fn form_rescaled_variables<E: PartialOrd + Scalar>(x: ArrayView1<'_, E>) -> Rescaled<E> {
    let domain = find_limits(x.as_slice().unwrap());

    let t = x
        .into_iter()
        .map(|&x| (x + x - domain.end - domain.start) / (domain.end - domain.start))
        .collect();

    Rescaled { t, domain }
}

/// Returns the range spanning the minimium and maximum of the `variable`
///
/// # Panics
///
/// This function will panic if the variable contains NaN or infinities. It is the responsibility
/// of the caller to check and handle this error.
pub(crate) fn find_limits<E: Clone + PartialOrd>(variable: &[E]) -> Range<E> {
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
