//! polycal is a library for fitting, and using, polynomial calibration functions.
//!
//! It's goal is to offer a simple, application agnostic, interface allowing for easy integration
//! into code bases in many fields.
//!
//! The implementation follows the ISO/TS 28038 standard where possible.
//!
//! To use the library, we first create a [`Problem`] using some known calibration data. Calibration
//! data is composed of independent, or stimulus variables. These describe the inputs to the
//! measurement model. For each stimulus there is an associated response, or dependent variable. A
//! calibration problem can be constructed from known stimulus and response data. Currently the
//! inputs must be provided as [`ndarray::ArrayView1`].
//!
//! Note that all methods with [`panic`] if the provided stimulus, response and uncertainties
//! contain unequal numbers of elements.
//!
//! ## Reconstruction
//!
//! Given a [`Fit`] we can reconstruct unknown response from known stimulus values. This uses the
//! calculated polynomial series directly.
//!
//! Alternatively we can calculate unknown stimulus values from known response values. This
//! numerically minimises the residual of the fit. An initial guess and maximum iteration count can
//! be provided.
//!

mod builder;
mod problem;

pub use builder::ProblemBuilder;
pub use problem::Problem;

use ndarray::{Array1, Array2};
use polynomial_series::ChebyshevSeries;

#[derive(Debug)]
pub enum Uncertainty<E> {
    None,
    YDiagonal { uy: Array1<E> },
    YCovariance { vy: Array2<E> },
    XYDiagonal { ux: Array1<E>, uy: Array1<E> },
    XYCovariance { vx: Array2<E>, vy: Array2<E> },
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// A constraint.
///
/// Given a constraint we use the problem y = `p_n(x`, a) * multiplicative(x) + additive(x). A
/// carefully constructed constraint can ensure the response variable and it's derivatives obeys
/// certain pre-conditions such as passing through the origin.
pub struct Constraint<E> {
    /// Additive component of the constraint
    pub(crate) additive: ChebyshevSeries<E>,
    /// Multiplicative component of the constraint
    pub(crate) multiplicative: ChebyshevSeries<E>,
}

/// Different scoring strategies for fit procedure
#[derive(Copy, Clone, Debug)]
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
