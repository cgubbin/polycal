use ndarray::{Array1, ArrayView1, ArrayView2};
use std::ops::Range;

use crate::chebyshev::Series;


pub enum ScoringStrategy {
    /// Akaike's method
    Aic,
    /// Akaike's corrected method
    Aicc,
    /// Bayesian
    Bic,
    /// Pure chi-squared residuals
    ChiSquare,
}

pub enum Covariance<'a, E> {
    None,
    Uncertainty {
        ux: Option<ArrayView1<'a, E>>,
        uy: ArrayView1<'a, E>,
    },
    Covariance {
        vx: Option<ArrayView2<'a, E>>,
        vy: ArrayView2<'a, E>,
    },
}

#[derive(Clone, Debug)]
pub struct Constraint<E> {
    pub(crate) additive: Series<E>,
    pub(crate) multiplicative: Series<E>,
}

pub struct Problem<'a, E> {
    pub(crate) t: Array1<E>,
    pub(crate) y: ArrayView1<'a, E>,
    pub(crate) uncertainties: Covariance<'a, E>,
    pub(crate) domain: Range<E>,
    pub(crate) strategy: ScoringStrategy,
    pub(crate) constraint: Option<Constraint<E>>,
}
