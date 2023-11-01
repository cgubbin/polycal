use ndarray_linalg::Scalar;
use num_traits::float::FloatCore;
use std::ops::Range;

use super::{Basis, ChebyshevError, Series};
use crate::utils::find_limits;

#[derive(Default)]
pub struct Set {}

#[derive(Default)]
pub struct Unset {}

#[allow(clippy::module_name_repetitions)]
pub struct ChebyshevBuilder<C, D, W> {
    degree: usize,
    coeff: C,
    domain: D,
    window: W,
}

impl ChebyshevBuilder<Unset, Unset, Unset> {
    pub const fn new(degree: usize) -> Self {
        Self {
            degree,
            coeff: Unset {},
            domain: Unset {},
            window: Unset {},
        }
    }
}

impl<D, W> ChebyshevBuilder<Unset, D, W> {
    pub fn with_coefficients<E, C: Into<Vec<E>>>(
        self,
        coefficients: C,
    ) -> ChebyshevBuilder<Vec<E>, D, W> {
        ChebyshevBuilder {
            degree: self.degree,
            coeff: coefficients.into(),
            domain: self.domain,
            window: self.window,
        }
    }
}

impl<E, W> ChebyshevBuilder<Vec<E>, Unset, W> {
    pub fn on_domain(self, domain: Range<E>) -> ChebyshevBuilder<Vec<E>, Range<E>, W> {
        ChebyshevBuilder {
            degree: self.degree,
            coeff: self.coeff,
            domain,
            window: self.window,
        }
    }
}

impl<E: FloatCore + PartialOrd + Clone, W> ChebyshevBuilder<Vec<E>, Unset, W> {
    pub fn on_domain_from(
        self,
        independent: &[E],
    ) -> Result<ChebyshevBuilder<Vec<E>, Range<E>, W>, ChebyshevError> {
        if independent.iter().any(|x| !x.is_finite()) {
            return Err(ChebyshevError::InvalidData);
        }
        let domain = find_limits(independent);
        Ok(ChebyshevBuilder {
            degree: self.degree,
            coeff: self.coeff,
            domain,
            window: self.window,
        })
    }
}

impl<E, D> ChebyshevBuilder<Vec<E>, D, Unset> {
    pub(crate) fn on_window(self, window: Range<E>) -> ChebyshevBuilder<Vec<E>, D, Range<E>> {
        ChebyshevBuilder {
            degree: self.degree,
            coeff: self.coeff,
            domain: self.domain,
            window,
        }
    }
}

impl<E: FloatCore + PartialOrd + Clone, D> ChebyshevBuilder<Vec<E>, D, Unset> {
    pub(crate) fn on_window_from(
        self,
        independent: &[E],
    ) -> Result<ChebyshevBuilder<Vec<E>, D, Range<E>>, ChebyshevError> {
        if independent.iter().any(|x| !x.is_finite()) {
            return Err(ChebyshevError::InvalidData);
        }
        let window = find_limits(independent);
        Ok(ChebyshevBuilder {
            degree: self.degree,
            coeff: self.coeff,
            domain: self.domain,
            window,
        })
    }
}

impl ChebyshevBuilder<Unset, Unset, Unset> {
    pub(crate) const fn build(self) -> Basis {
        Basis::new(self.degree)
    }
}

impl<E: Scalar<Real = E>> ChebyshevBuilder<Vec<E>, Range<E>, Unset> {
    pub fn build(self) -> Series<E> {
        Series {
            basis: Basis::new(self.degree),
            coeff: self.coeff.into(),
            domain: self.domain,
            window: Range {
                start: -E::one(),
                end: E::one(),
            },
        }
    }
}

impl<E: Scalar<Real = E>> ChebyshevBuilder<Vec<E>, Range<E>, Range<E>> {
    pub(crate) fn build(self) -> Series<E> {
        Series {
            basis: Basis::new(self.degree),
            coeff: self.coeff.into(),
            domain: self.domain,
            window: self.window,
        }
    }
}
