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
pub struct ChebyshevBuilder<C, D> {
    degree: usize,
    coeff: C,
    domain: D,
}

impl ChebyshevBuilder<Unset, Unset> {
    #[must_use]
    pub const fn new(degree: usize) -> Self {
        Self {
            degree,
            coeff: Unset {},
            domain: Unset {},
        }
    }
}

impl<D> ChebyshevBuilder<Unset, D> {
    pub fn with_coefficients<E, C: Into<Vec<E>>>(
        self,
        coefficients: C,
    ) -> ChebyshevBuilder<Vec<E>, D> {
        ChebyshevBuilder {
            degree: self.degree,
            coeff: coefficients.into(),
            domain: self.domain,
        }
    }
}

impl<E> ChebyshevBuilder<Vec<E>, Unset> {
    pub fn on_domain(self, domain: Range<E>) -> ChebyshevBuilder<Vec<E>, Range<E>> {
        ChebyshevBuilder {
            degree: self.degree,
            coeff: self.coeff,
            domain,
        }
    }
}

impl<E: FloatCore + PartialOrd + Clone> ChebyshevBuilder<Vec<E>, Unset> {
    /// Attach a domain, constructed from the polynomial's x-values
    ///
    /// # Errors
    /// - If the provided independent variables contain infinities or NaN values.
    pub fn on_domain_from(
        self,
        independent: &[E],
    ) -> Result<ChebyshevBuilder<Vec<E>, Range<E>>, ChebyshevError> {
        if independent.iter().any(|x| !x.is_finite()) {
            return Err(ChebyshevError::InvalidData);
        }
        let domain = find_limits(independent);
        Ok(ChebyshevBuilder {
            degree: self.degree,
            coeff: self.coeff,
            domain,
        })
    }
}

impl ChebyshevBuilder<Unset, Unset> {
    pub(crate) const fn build(self) -> Basis {
        Basis::new(self.degree)
    }
}

impl<E: Scalar<Real = E>> ChebyshevBuilder<Vec<E>, Range<E>> {
    pub fn build(self) -> Series<E> {
        Series::new(self.coeff, self.domain)
    }
}
