use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand, ShapeError};
use ndarray_linalg::{Lapack, Scalar};
use num_traits::float::FloatCore;
use std::ops::Range;
use tracing::{event, Level};

use crate::calculate::Fit;
use crate::chebyshev::{
    Basis, ChebyshevBuilder, ChebyshevError, ConstrainedPolynomial, Polynomial, PolynomialSeries,
    Series,
};
use crate::solvers::{
    SolveSystem, SolverError, TotalLeastSquares, Uncertainty, WeightedLeastSquares,
};
use crate::utils::find_limits;
use crate::PolyCalError;

/// Different scoring strategies for fit procedure
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

pub enum Covariance<'a, E> {
    None,
    Uncertainty {
        ux: Option<ArrayView1<'a, E>>,
        uy: ArrayView1<'a, E>,
    },
    Covariance {
        vx: Option<ArrayView2<'a, E>>,
        vy: ArrayView2<'a, E>,
    },
}

#[derive(Clone, Debug)]
/// A constraint.
///
/// Given a constraint we use the problem y = p_n(x, a) * multiplicative(x) + additive(x). A
/// carefully constructed constraint can ensure the response variable and it's derivatives obeys
/// certain pre-conditions such as passing through the origin.
pub struct Constraint<E> {
    /// Additive component of the constraint
    pub(crate) additive: Series<E>,
    /// Multiplicative component of the constraint
    pub(crate) multiplicative: Series<E>,
}

/// Problem abstraction
///
/// Problems are created using a [`ProblemBuilder`] which ensures the type-state of uncertainties
/// is consistent.
pub struct Problem<'a, E> {
    pub(crate) t: Array1<E>,
    pub(crate) y: ArrayView1<'a, E>,
    pub(crate) uncertainties: Covariance<'a, E>,
    pub(crate) domain: Range<E>,
    pub(crate) strategy: ScoringStrategy,
    pub(crate) constraint: Option<Constraint<E>>,
}

impl<'a, E> Problem<'a, E> {
    fn number_of_datapoints(&self) -> usize {
        self.t.len()
    }
}

impl<'a, E> Problem<'a, E>
where
    E: Scalar<Real = E> + PartialOrd + ScalarOperand + Lapack + FloatCore,
{
    /// Solves a problem using all polynomial degrees up to `n_max`.
    ///
    /// Each solution is checked for monotonicity, if found to be non-monotonic the solution is
    /// discarded as it is unsuitable for use as a calibration curve.
    ///
    /// After solution construction each is assessed according to the chosen [`ScoringStrategy`].
    /// If the vector of scoring strategies exhibits a minimum which is not at the endpoints this
    /// solution is selected and returned.
    ///
    /// In the event no minimum is found the chi-2 score for each solution is assessed, and the
    /// lowest order solution to beat a standard tolerance is returned.
    ///
    #[tracing::instrument(skip(self))]
    pub fn solve(&self, n_max: usize) -> ::std::result::Result<Fit<E>, PolyCalError<E>> {
        let fits = (1..n_max)
            .filter_map(|polynomial_degree| match self.fit(polynomial_degree) {
                Ok(fit) => match self.check_is_monotonic(fit.solution()) {
                    Ok(true) => Some(fit),
                    Ok(false) => {
                        eprintln!("found non-monotonic solution");
                        None
                    }
                    Err(err) => {
                        eprintln!("{err:?}");
                        None
                    }
                },
                Err(err) => {
                    eprintln!("{err:?}");
                    None
                }
            })
            .collect::<Vec<_>>();

        event!(
            Level::INFO,
            num_successes = fits.len(),
            num_failures = n_max - fits.len(),
            "finding best fit"
        );
        let (_best_score, best_fit) = self.find_best_fit(fits)?;

        // TODO: Chi-2 validation at nu = m - n - 1

        Ok(best_fit)
    }

    fn find_best_fit(
        &self,
        mut fits: Vec<Fit<E>>,
    ) -> ::std::result::Result<(E, Fit<E>), PolyCalError<E>> {
        let scores = fits
            .iter()
            .map(|fit| self.score(fit.solution()))
            .collect::<Vec<_>>();
        let diffs = scores
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect::<Vec<_>>();
        if diffs
            .windows(2)
            .all(|window| window[0].signum() == window[1].signum())
        {
            // No minimum
            // Try again...
            let chi_2_scores = fits
                .iter()
                .map(|fit| self.chi_2(fit.solution()))
                .collect::<Vec<_>>();
            let best_score = *chi_2_scores
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let scores = chi_2_scores
                .into_iter()
                .map(|score| score - best_score)
                .collect::<Vec<_>>();

            let index_of_lowest_order_acceptable_solution = scores
                .iter()
                .position(|&score| score < E::epsilon()).unwrap();

            let best_fit = fits.swap_remove(index_of_lowest_order_acceptable_solution);

            dbg!(&best_score);
            dbg!(&best_fit.solution);

            Ok((best_score, best_fit))

        } else {
            let best_score = *scores
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let scores = scores
                .into_iter()
                .map(|score| score - best_score)
                .collect::<Vec<_>>();

            // Can't fail as we just substracted the best score: this means one element will always be
            // zero
            let index = scores.iter().position(|&score| score == E::zero()).unwrap();

            let best_fit = fits.swap_remove(index);

            Ok((best_score, best_fit))
        }
    }

    fn score(&self, fit: &Series<E>) -> E {
        let chi_2_score = self.chi_2(fit);
        match self.strategy {
            ScoringStrategy::Aic => chi_2_score + E::from(2 * (fit.degree() + 1)).unwrap(),
            ScoringStrategy::Aicc => {
                let n = E::from(fit.degree() + 1).unwrap();
                chi_2_score
                    + (E::one() + E::one()) * n
                    + (E::one() + E::one()) * (n + E::one()) * (n + E::one() + E::one())
                        / (E::from(self.number_of_datapoints()).unwrap() - n - E::one())
            }
            ScoringStrategy::Bic => {
                chi_2_score
                    + (E::from(fit.degree() + 1).unwrap() + E::one())
                        * E::from(self.number_of_datapoints()).unwrap().ln()
            }
            ScoringStrategy::ChiSquare => chi_2_score,
        }
    }

    fn chi_2(&self, fit: &Series<E>) -> E {
        self.t.iter().zip(self.y).fold(E::zero(), |a, (t, y)| {
            a + Scalar::powi(
                *y - self.constraint.as_ref().map_or_else(
                    || fit.evaluate(*t),
                    |constraint| fit.evaluate(*t) * constraint.multiplicative.evaluate(*t),
                ),
                2,
            )
        })
    }

    #[tracing::instrument(skip(self))]
    fn fit(&self, polynomial_degree: usize) -> ::std::result::Result<Fit<E>, SolverError> {
        let design_matrix = self.design_matrix(polynomial_degree).unwrap(); // This method is fallible, but only because of a matrix-shape-conversion.
                                                                            // As the method takes a single parameter, then builds the matrix, this
                                                                            // cannot occur in practice
        let y = self.constraint.as_ref().map_or_else(
            || self.y.to_owned(),
            |constraint| self.shifted_independent_variable(constraint),
        );

        let result = match self.uncertainties {
            Covariance::None => WeightedLeastSquares {
                y,
                uncertainty: Uncertainty::None,
                h: design_matrix,
            }
            .solve(),
            Covariance::Uncertainty { ux, uy } if ux.is_none() => WeightedLeastSquares {
                y,
                uncertainty: Uncertainty::Diagonal(uy),
                h: design_matrix,
            }
            .solve(),
            Covariance::Covariance { vx, vy } if vx.is_none() => WeightedLeastSquares {
                y,
                uncertainty: Uncertainty::Full(vy),
                h: design_matrix,
            }
            .solve(),
            Covariance::Uncertainty { ux, uy } => TotalLeastSquares {
                y,
                uncertainty_x: Uncertainty::Diagonal(ux.unwrap()),
                uncertainty_y: Uncertainty::Diagonal(uy),
                h: design_matrix,
            }
            .solve(),
            Covariance::Covariance { vx, vy } => TotalLeastSquares {
                y,
                uncertainty_x: Uncertainty::Full(vx.unwrap()),
                uncertainty_y: Uncertainty::Full(vy),
                h: design_matrix,
            }
            .solve(),
        }?;

        Ok(Fit {
            solution: ChebyshevBuilder::new(polynomial_degree)
                .with_coefficients(result.coeff().to_vec())
                .on_domain(self.domain.clone())
                .build(),
            covariance: result.covariance().to_owned(),
            constraint: self.constraint.clone(),
            response_domain: find_limits(self.y.to_slice().unwrap()), // we build y in the
                                                                      // constructor, so know it is contiguous and in standard order. under these
                                                                      // circumstances the unwrap is infallible.
        })
    }

    fn shifted_independent_variable(
        &self,
        Constraint {
            additive,
            multiplicative: _,
        }: &Constraint<E>,
    ) -> Array1<E> {
        self.y
            .to_owned()
            .iter()
            .zip(self.t.iter())
            .map(|(y, t)| *y - additive.evaluate(*t))
            .collect()
    }

    pub(crate) fn design_matrix(
        &self,
        polynomial_degree: usize,
    ) -> ::std::result::Result<Array2<E>, ShapeError> {
        let basis = Basis::new(polynomial_degree);
        let rows = self
            .t
            .iter()
            .flat_map(|t| {
                self.constraint.as_ref().map_or_else(
                    || basis.polynomials(*t),
                    |constraint| basis.polynomials_with_constraint(*t, &constraint.multiplicative),
                )
            })
            .collect::<Vec<E>>();

        Array2::from_shape_vec((self.number_of_datapoints(), polynomial_degree + 1), rows)
    }

    /// Check whether a given solution is monotonic
    ///
    /// This function applies the constraint, if available, and checks if the resulting polynomial
    /// is monotonic.
    ///
    /// # Errors
    /// - If there is an error in the underlying root finding algorithm.
    pub fn check_is_monotonic(
        &self,
        solution: &Series<E>,
    ) -> ::std::result::Result<bool, ChebyshevError> {
        self.constraint.as_ref().map_or_else(
            || solution.is_monotonic(),
            |constraint| (solution.clone() * constraint.multiplicative.clone()).is_monotonic(),
        )
    }
}
