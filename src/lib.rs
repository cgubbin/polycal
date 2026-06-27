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
mod evaluation;
mod fit;
mod problem;
mod score;
mod solve;

pub use builder::ProblemBuilder;
pub use problem::Problem;
pub use score::ScoringStrategy;
