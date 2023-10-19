mod builder;
mod chebyshev;
mod eval;
mod fit;

pub type Result<T> = ::std::result::Result<T, Box<dyn ::std::error::Error>>;

pub(crate) use chebyshev::ChebyshevPolynomial;
