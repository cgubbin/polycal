use crate::{fit::Constraint, score::ScoringStrategy};

use confi::ConfidenceLevel;
use ndarray::{Array1, Array2};
use std::ops::Range;

#[derive(Debug, Clone)]
pub struct Problem<E> {
    /// Physical independent/reference values.
    pub(crate) x: Array1<E>,

    /// Physical dependent/response values.
    pub(crate) y: Array1<E>,

    /// Measurement uncertainty model.
    pub(crate) uncertainty: Uncertainty<E>,

    /// Calibration domain for the independent variable.
    pub(crate) domain: Range<E>,

    /// Calibration domain for the dependent variable.
    pub(crate) response_domain: Range<E>,

    /// Model-selection/scoring strategy.
    pub(crate) strategy: ScoringStrategy,

    /// Optional calibration constraint.
    pub(crate) constraint: Option<Constraint<E>>,

    /// Validation policies
    pub(crate) goodness_of_fit: GoodnessOfFit<E>,
}

#[derive(Clone, Debug)]
pub enum Uncertainty<E> {
    None,
    YDiagonal { uy: Array1<E> },
    YCovariance { vy: Array2<E> },
    XYDiagonal { ux: Array1<E>, uy: Array1<E> },
    XYCovariance { vx: Array2<E>, vy: Array2<E> },
}

#[derive(Clone, Debug)]
pub enum GoodnessOfFit<E> {
    Disabled,
    ChiSquare { confidence: ConfidenceLevel<E> },
}

impl<E> Problem<E> {
    pub(crate) fn len(&self) -> usize {
        self.x.len()
    }

    pub(crate) fn degrees_of_freedom(&self, degree: usize) -> usize {
        let parameters = degree + 1;
        self.x.len().saturating_sub(parameters)
    }
}
