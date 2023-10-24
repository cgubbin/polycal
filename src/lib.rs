#![allow(dead_code)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

mod builder;
mod calculate;
mod chebyshev;
mod problem;
mod solvers;
mod utils;

pub type Result<T> = ::std::result::Result<T, Box<dyn ::std::error::Error>>;

pub use builder::ProblemBuilder;
pub use calculate::{Fit, Unsure};
pub use problem::{Constraint, Problem};
