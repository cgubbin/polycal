#![allow(dead_code)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

mod builder;
mod chebyshev;
mod eval;
mod fit;
mod series;

pub type Result<T> = ::std::result::Result<T, Box<dyn ::std::error::Error>>;

pub(crate) use chebyshev::ChebyshevPolynomial;
