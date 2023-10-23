use super::PolynomialSeries;
use ndarray_linalg::Scalar;

#[derive(Clone, Debug)]
pub(crate) struct Basis {
    degree: usize,
}

impl Basis {
    pub(crate) const fn new(degree: usize) -> Self {
        Self { degree }
    }

    pub(crate) const fn degree(&self) -> usize {
        self.degree
    }
}

pub(crate) trait Polynomial<E: Scalar<Real = E> + PartialOrd> {
    /// Return the underlying polynomials as a Vec evaluated at `t`
    ///
    /// This assumes t is in the rescaled range [-1, 1], as this is the basis the polynomials are
    /// defined over. The resulting Vec has `degree + 1` entries, a first element of unity
    /// representing the constant offset from the zero-order term followed by one for each polynomial in the
    /// series of `degree`.
    fn polynomials(&self, t: E) -> Vec<E>;
}

pub(crate) trait ConstrainedPolynomial<E: Scalar<Real = E> + PartialOrd, S: PolynomialSeries<E>>: Polynomial<E> {
    /// Return the underlying polynomials as a Vec evaluated at `t`, in which each element is
    /// multiplied by the supplied constraint.
    ///
    /// This assumes t is in the rescaled range [-1, 1], as this is the basis the polynomials are
    /// defined over. The resulting Vec has `degree + 1` entries, a first element of unity
    /// representing the constant offset from the zero-order term followed by one for each polynomial in the
    /// series of `degree`.
    fn polynomials_with_constraint(&self, t: E, multiplicative_constraint: &S) -> Vec<E> {
        let mut polynomials_in_basis = self.polynomials(t);
        let constraint_value = multiplicative_constraint.evaluate(t);

        polynomials_in_basis
            .iter_mut()
            .for_each(|polynomial_in_basis| *polynomial_in_basis *= constraint_value);

        polynomials_in_basis
    }
}

impl<E: Scalar<Real = E> + PartialOrd> Polynomial<E> for Basis {
    fn polynomials(&self, t: E) -> Vec<E> {
        match self.degree() {
            0 => vec![E::one()],
            1 => vec![E::one(), t],
            _ => {
                let mut vals = vec![E::one(), t];
                for ii in 1..self.degree() {
                    vals.push((E::one() + E::one()) * t * vals[ii] - vals[ii - 1]);
                }
                vals
            }
        }
    }
}

impl<E: Scalar<Real = E> + PartialOrd, S: PolynomialSeries<E>> ConstrainedPolynomial<E, S> for Basis { }
