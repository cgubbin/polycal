#![allow(dead_code)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

mod builder;
mod calculate;
mod chebyshev;
mod error;
mod problem;
mod solvers;
mod utils;

pub type PolyCalResult<T, E> = ::std::result::Result<T, PolyCalError<E>>;

pub use builder::ProblemBuilder;
pub use calculate::{Fit, Unsure};
pub use error::PolyCalError;
pub use problem::{Constraint, Problem};
