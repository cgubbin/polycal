use crate::chebyshev::ChebyshevError;
use crate::solvers::SolverError;
use std::ops::Range;

#[derive(Debug)]
pub enum Kind {
    Stimulus,
    Response,
}

#[derive(Debug, thiserror::Error)]
pub enum PolyCalError<E> {
    #[error("input {kind:?} ({value}) is out of {kind:?} calibration working range: {range}")]
    OutOfRange {
        value: E,
        range: Range<E>,
        kind: Kind,
    },
    #[error("failed to solve inverse problem")]
    InverseSolver(#[from] argmin::core::Error),
    #[error("failed to solve least-squares problem")]
    LeastSquares(#[from] SolverError),
    #[error("no minimum found: {scores:?}. perhaps try a different scoring criteria?")]
    NoMinimum { scores: Vec<E> },
    #[error("provided data must be free of NaN, or infinities")]
    InvalidData,
    #[error("error in chebyshev calculation")]
    Chebyshev(ChebyshevError),
}
