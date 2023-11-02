use crate::chebyshev::ChebyshevError;
use crate::solvers::SolverError;
use std::ops::Range;

#[derive(Debug)]
pub enum Kind {
    Stimulus,
    Response,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, thiserror::Error)]
pub enum PolyCalError<E> {
    /// The input value was outside of the calibration range, so a prediction cannot be reliably
    /// made.
    #[error("input {kind:?} ({value}) is out of {kind:?} calibration working range: {range:?}")]
    OutOfRange {
        value: E,
        range: Range<E>,
        kind: Kind,
    },
    /// Failure in the underlying inverse solver.
    #[error("failed to solve inverse problem")]
    InverseSolver(#[from] argmin::core::Error),
    /// Failure in the underlying least-squares routine
    #[error("failed to solve least-squares problem")]
    LeastSquares(#[from] SolverError),
    /// There was no minimum found using the requested scoring strategy.
    #[error("no minimum found: {scores:?}. perhaps try a different scoring criteria?")]
    NoMinimum { scores: Vec<E> },
    /// Input data contained invalid values, leaving the calculation unable to proceed
    #[error("provided data must be free of NaN, or infinities")]
    InvalidData,
    /// Error in low-level Chebyshev calculation
    #[error("error in chebyshev calculation")]
    Chebyshev(ChebyshevError),
}
