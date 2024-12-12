use crate::chebyshev::ChebyshevError;
use crate::solvers::SolverError;
use cert::AbsUncertainty;
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
    ///
    /// In this scenario we return the value anyway because the caller needs to decide what to do.
    /// In some scenarios using the value might be acceptable, particularly when the input value
    /// was very close to the calibration range. Alternatively if the caller is running an
    /// unconstrained optimisation the solver could leave the finite function domain. In that case,
    /// as long as the value solved for lies in the function domain at the end of the optimisation
    /// there is no harm in leaving it during the solve.
    ///
    /// Note though that guarantees around monotonicity are only made within the domain of the
    /// polynomial. It is possible a root outside the domain may be found, and it is the
    /// responsibility of the caller to check.
    #[error("input {kind:?} ({value}) is out of {kind:?} calibration working range: {range:?}")]
    OutOfRangeCertain {
        value: E,
        evaluated: E,
        range: Range<E>,
        kind: Kind,
    },
    /// The input value was outside of the calibration range, so a prediction cannot be reliably
    /// made.
    ///
    /// In this scenario we return the value anyway because the caller needs to decide what to do.
    /// In some scenarios using the value might be acceptable, particularly when the input value
    /// was very close to the calibration range. Alternatively if the caller is running an
    /// unconstrained optimisation the solver could leave the finite function domain. In that case,
    /// as long as the value solved for lies in the function domain at the end of the optimisation
    /// there is no harm in leaving it during the solve.
    ///
    /// Note though that guarantees around monotonicity are only made within the domain of the
    /// polynomial. It is possible a root outside the domain may be found, and it is the
    /// responsibility of the caller to check.
    #[error("input {kind:?} ({value}) is out of {kind:?} calibration working range: {range:?}")]
    OutOfRangeUncertain {
        value: E,
        evaluated: AbsUncertainty<E>,
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
    InvalidData(String),
    /// Error in low-level Chebyshev calculation
    #[error("error in chebyshev calculation")]
    Chebyshev(ChebyshevError),
    #[error("a successful fit was not possible for the given data")]
    FittingFailure {
        dependent: Vec<E>,
        independent: Vec<E>,
    },
}
