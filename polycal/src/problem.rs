use crate::{Constraint, ProblemBuilder, ScoringStrategy, Uncertainty};

use ndarray::{Array1, Array2};
use polynomial_series::ChebyshevSeries;
use std::ops::Range;

pub struct Problem<E> {
    /// Physical independent/reference values.
    pub(crate) x: Array1<E>,

    /// Physical dependent/response values.
    pub(crate) y: Array1<E>,

    /// Measurement uncertainty model.
    pub(crate) uncertainty: Uncertainty<E>,

    /// Calibration domain for the independent variable.
    pub(crate) domain: Range<E>,

    /// Model-selection/scoring strategy.
    pub(crate) strategy: ScoringStrategy,

    /// Optional calibration constraint.
    pub(crate) constraint: Option<Constraint<E>>,
}

impl<E> Problem<E> {
    /// Begin constructing a calibration problem.
    #[must_use]
    pub fn builder() -> ProblemBuilder<E> {
        ProblemBuilder::new()
    }
}
