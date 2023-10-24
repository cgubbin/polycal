use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar};
use num_traits::float::FloatCore;
use std::ops::Range;

use crate::calculate::Fit;
use crate::chebyshev::{
    Basis, ChebyshevBuilder, ConstrainedPolynomial, Polynomial, PolynomialSeries, Series,
};
use crate::solvers::{SolveSystem, TotalLeastSquares, Uncertainty, WeightedLeastSquares};
use crate::Result;

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
pub struct Constraint<E> {
    pub(crate) additive: Series<E>,
    pub(crate) multiplicative: Series<E>,
}

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
    pub fn solve(&self, n_max: usize) -> Result<Fit<E>> {
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

        let (_best_score, best_fit) = self.find_best_fit(fits);

        // TODO: Chi-2 validation at nu = m - n - 1

        Ok(best_fit)
    }

    fn find_best_fit(&self, mut fits: Vec<Fit<E>>) -> (E, Fit<E>) {
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
            println!("no minimum found");
        }
        let best_score = *scores
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let scores = scores
            .into_iter()
            .map(|score| score - best_score)
            .collect::<Vec<_>>();

        let index = scores.iter().position(|&score| score == E::zero()).unwrap();

        let best_fit = fits.swap_remove(index);

        (best_score, best_fit)
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

    fn fit(&self, polynomial_degree: usize) -> Result<Fit<E>> {
        let design_matrix = self.design_matrix(polynomial_degree)?;
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

    fn design_matrix(&self, polynomial_degree: usize) -> Result<Array2<E>> {
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

        Ok(Array2::from_shape_vec(
            (self.number_of_datapoints(), polynomial_degree + 1),
            rows,
        )?)
    }

    fn check_is_monotonic(&self, solution: &Series<E>) -> Result<bool> {
        self.constraint.as_ref().map_or_else(
            || solution.is_monotonic(),
            |constraint| (solution.clone() * constraint.multiplicative.clone()).is_monotonic(),
        )
    }
}
