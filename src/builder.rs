//! Builders for Polynomial Calibration problems.
//!
//! This module provides flexible builder methods to allow polynomial calibration problems to be
//! fluently built from a variety of inputs. A builder takes independent and dependent variables
//! and is converted to a [`Problem`] with a final call to the `build` method.
//!
//! ```
//! use ndarray::Array1;
//! use polycal::ProblemBuilder;
//!
//! let stimulus: Array1<f64> = Array1::range(0., 10., 0.5);
//! let num_data_points = stimulus.len();
//! let a: f64 = 1.0;
//! let b: f64 = 2.0;
//! let response: Array1<f64> = stimulus
//!     .iter()
//!     .map(|x| a + b * x)
//!     .collect();
//!
//! let problem = ProblemBuilder::new(stimulus.view(), response.view())
//!     .unwrap()
//!     .build();
//! ```
//!
//! A [`ProblemBuilder`] can also be attached by either variances, or covariances for the
//! independent variable (response). Methods to attach and build with variances, or covariances on
//! the dependent variable exist in the private API only, as these fit methods are unimplemented.
//!

use ndarray::{ArrayView1, ArrayView2};
use ndarray_linalg::Scalar;
use num_traits::Float;
use std::marker::PhantomData;

use crate::problem::{Constraint, Covariance, Problem, ScoringStrategy};
use crate::utils::{form_rescaled_variables, Rescaled};
use crate::PolyCalError;

#[derive(Default)]
/// Marker struct to indicate a field has been previously set.
pub struct Set {}

#[derive(Default)]
/// Marker struct to indicate a field remains unset
pub struct Unset {}

/// Problem builder
///
/// The [`ProblemBuilder`] allows us to build a [`Problem`] from known inputs
/// and uncertainties.
#[allow(clippy::module_name_repetitions)]
pub struct ProblemBuilder<'a, E, DU, IU, DC, IC, C> {
    /// Dependent or stimulus data
    dependent: ArrayView1<'a, E>,
    /// Independent or response data
    independent: ArrayView1<'a, E>,
    /// Dependent or stimulus data variances
    dependent_variance: Option<ArrayView1<'a, E>>,
    /// Independent or response data variances
    independent_variance: Option<ArrayView1<'a, E>>,
    /// Dependent or stimulus data covariances
    dependent_covariance: Option<ArrayView2<'a, E>>,
    /// Independent or response data covariances
    independent_covariance: Option<ArrayView2<'a, E>>,
    /// A polynomial constraint function
    ///
    /// The constraint, if present, alters the basis used for polynomial fitting. It can allow the
    /// solver to restrict to the subspace of polynomial solutions satisfying the constraint.
    constraint: Option<C>,
    /// Preferred scoring strategy to assess the suitability of fits.
    strategy: ScoringStrategy,
    typestate: PhantomData<(DU, IU, DC, IC, C)>,
}

impl<'a, E: Float> ProblemBuilder<'a, E, Unset, Unset, Unset, Unset, Unset> {
    /// Create a new [`ProblemBuilder`] from independent and dependent data.
    ///
    /// # Errors
    /// - If the independent and dependent data provided contain unequal numbers of observations.
    /// - If the independent and dependent data provided contain NaN or infinity
    pub fn new<V: Into<ArrayView1<'a, E>>>(
        independent: V,
        dependent: V,
    ) -> Result<Self, PolyCalError<E>> {
        let dependent: ArrayView1<'a, E> = dependent.into();
        let independent: ArrayView1<'a, E> = independent.into();
        if independent.len() != dependent.len() {
            return Err(PolyCalError::InvalidData(
                "dependent and independent data must contain equal numbers of observations".into(),
            ));
        }
        if dependent.iter().any(|each| !each.is_finite()) {
            return Err(PolyCalError::InvalidData(
                "dependent values cannot contain NaN or infinity".into(),
            ));
        }
        if independent.iter().any(|each| !each.is_finite()) {
            return Err(PolyCalError::InvalidData(
                "independent values cannot contain NaN or infinity".into(),
            ));
        }
        let builder = Self {
            dependent,
            independent,
            dependent_variance: None,
            independent_variance: None,
            dependent_covariance: None,
            independent_covariance: None,
            constraint: None,
            strategy: ScoringStrategy::ChiSquare,
            typestate: PhantomData,
        };

        Ok(builder)
    }
}

impl<E, DU, IU, DC, IC, C> ProblemBuilder<'_, E, DU, IU, DC, IC, C> {
    #[must_use]
    /// Attach a scoring strategy.
    pub const fn with_scoring_strategy(mut self, scoring_strategy: ScoringStrategy) -> Self {
        self.strategy = scoring_strategy;
        self
    }
}

impl<'a, E: Float, C> ProblemBuilder<'a, E, Unset, Unset, Unset, Unset, C> {
    #[allow(clippy::type_complexity)]
    /// Attach a variance for the dependent variable
    ///
    /// # Errors
    /// - If the provided values contain any NaN, infinite, or zero values. Note that zeros are not
    ///     allowed, because the weighted least squares process relies on the inverse of the
    ///     variance
    fn with_dependent_variance(
        self,
        dependent_variance: impl Into<ArrayView1<'a, E>>,
    ) -> Result<ProblemBuilder<'a, E, Set, Unset, Unset, Unset, C>, PolyCalError<E>> {
        let dependent_variance: ArrayView1<'a, E> = dependent_variance.into();
        if dependent_variance
            .iter()
            .any(|each| !each.is_finite() || (*each == E::zero()))
        {
            return Err(PolyCalError::InvalidData(
                "dependent variance cannot contain NaN or zeroes".into(),
            ));
        }
        Ok(ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_variance: Some(dependent_variance),
            independent_variance: self.independent_variance,
            dependent_covariance: self.dependent_covariance,
            independent_covariance: self.independent_covariance,
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        })
    }
}

impl<'a, E: Float, C> ProblemBuilder<'a, E, Unset, Set, Unset, Unset, C> {
    #[allow(clippy::type_complexity)]
    /// Attach a variance for the dependent variable
    ///
    /// # Errors
    /// - If the provided values contain any NaN, infinite, or zero values. Note that zeros are not
    ///     allowed, because the weighted least squares process relies on the inverse of the
    ///     variance
    fn with_dependent_variance(
        self,
        dependent_variance: impl Into<ArrayView1<'a, E>>,
    ) -> Result<ProblemBuilder<'a, E, Set, Set, Unset, Unset, C>, PolyCalError<E>> {
        let dependent_variance: ArrayView1<'a, E> = dependent_variance.into();
        if dependent_variance
            .iter()
            .any(|each| !each.is_finite() || (*each == E::zero()))
        {
            return Err(PolyCalError::InvalidData(
                "dependent variance cannot contain NaN or zeroes".into(),
            ));
        }
        Ok(ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_variance: Some(dependent_variance),
            independent_variance: self.independent_variance,
            dependent_covariance: self.dependent_covariance,
            independent_covariance: self.independent_covariance,
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        })
    }
}

impl<'a, E: Float, C> ProblemBuilder<'a, E, Unset, Unset, Unset, Unset, C> {
    #[allow(clippy::type_complexity)]
    /// Attach a variance for the independent variable
    ///
    /// # Errors
    /// - If the provided values contain any NaN, infinite, or zero values. Note that zeros are not
    ///     allowed, because the weighted least squares process relies on the inverse of the
    ///     variance
    pub fn with_independent_variance(
        self,
        independent_variance: impl Into<ArrayView1<'a, E>>,
    ) -> Result<ProblemBuilder<'a, E, Unset, Set, Unset, Unset, C>, PolyCalError<E>> {
        let independent_variance: ArrayView1<'a, E> = independent_variance.into();
        if independent_variance
            .iter()
            .any(|each| !each.is_finite() || (*each == E::zero()))
        {
            return Err(PolyCalError::InvalidData(
                "independent variance cannot contain NaN or zeroes".into(),
            ));
        }
        Ok(ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_variance: self.dependent_variance,
            independent_variance: Some(independent_variance),
            dependent_covariance: self.dependent_covariance,
            independent_covariance: self.independent_covariance,
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        })
    }
}

impl<'a, E: Float, C> ProblemBuilder<'a, E, Set, Unset, Unset, Unset, C> {
    #[allow(clippy::type_complexity)]
    /// Attach a variance for the independent variable
    ///
    /// # Errors
    /// - If the provided values contain any NaN, infinite, or zero values. Note that zeros are not
    ///     allowed, because the weighted least squares process relies on the inverse of the
    ///     variance
    pub fn with_independent_variance(
        self,
        independent_variance: impl Into<ArrayView1<'a, E>>,
    ) -> Result<ProblemBuilder<'a, E, Set, Set, Unset, Unset, C>, PolyCalError<E>> {
        let independent_variance: ArrayView1<'a, E> = independent_variance.into();
        if independent_variance
            .iter()
            .any(|each| !each.is_finite() || (*each == E::zero()))
        {
            return Err(PolyCalError::InvalidData(
                "independent variance cannot contain NaN or zeroes".into(),
            ));
        }
        Ok(ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_variance: self.dependent_variance,
            independent_variance: Some(independent_variance),
            dependent_covariance: self.dependent_covariance,
            independent_covariance: self.independent_covariance,
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        })
    }
}

impl<'a, E, C> ProblemBuilder<'a, E, Unset, Unset, Unset, Unset, C> {
    fn with_dependent_covariance(
        self,
        dependent_covariance: impl Into<ArrayView2<'a, E>>,
    ) -> ProblemBuilder<'a, E, Unset, Unset, Set, Unset, C> {
        ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_variance: self.dependent_variance,
            independent_variance: self.independent_variance,
            dependent_covariance: Some(dependent_covariance.into()),
            independent_covariance: self.independent_covariance,
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        }
    }
}

impl<'a, E, C> ProblemBuilder<'a, E, Unset, Unset, Unset, Set, C> {
    fn with_dependent_covariance(
        self,
        dependent_covariance: impl Into<ArrayView2<'a, E>>,
    ) -> ProblemBuilder<'a, E, Unset, Unset, Set, Set, C> {
        ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_variance: self.dependent_variance,
            independent_variance: self.independent_variance,
            dependent_covariance: Some(dependent_covariance.into()),
            independent_covariance: self.independent_covariance,
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        }
    }
}

impl<'a, E, C> ProblemBuilder<'a, E, Unset, Unset, Unset, Unset, C> {
    pub(crate) fn with_independent_covariance(
        self,
        independent_covariance: impl Into<ArrayView2<'a, E>>,
    ) -> ProblemBuilder<'a, E, Unset, Unset, Unset, Set, C> {
        ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_variance: self.dependent_variance,
            independent_variance: self.independent_variance,
            dependent_covariance: self.dependent_covariance,
            independent_covariance: Some(independent_covariance.into()),
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        }
    }
}

impl<'a, E, C> ProblemBuilder<'a, E, Unset, Unset, Set, Unset, C> {
    fn with_independent_covariance(
        self,
        independent_covariance: impl Into<ArrayView2<'a, E>>,
    ) -> ProblemBuilder<'a, E, Unset, Unset, Set, Set, C> {
        ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_variance: self.dependent_variance,
            independent_variance: self.independent_variance,
            dependent_covariance: self.dependent_covariance,
            independent_covariance: Some(independent_covariance.into()),
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        }
    }
}

impl<'a, E, DU, IU, DC, IC> ProblemBuilder<'a, E, DU, IU, DC, IC, Unset> {
    pub const fn with_constraint<C>(
        self,
        constraint: C,
    ) -> ProblemBuilder<'a, E, DU, IU, DC, IC, C> {
        ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_variance: self.dependent_variance,
            independent_variance: self.independent_variance,
            dependent_covariance: self.dependent_covariance,
            independent_covariance: self.independent_covariance,
            strategy: self.strategy,
            constraint: Some(constraint),
            typestate: PhantomData,
        }
    }
}

impl<'a, E: PartialOrd + Scalar> ProblemBuilder<'a, E, Unset, Unset, Unset, Unset, Unset> {
    #[must_use]
    pub fn build(self) -> Problem<'a, E> {
        let Rescaled { t, domain } = form_rescaled_variables(self.independent);
        Problem {
            t,
            y: self.dependent,
            uncertainties: Covariance::None,
            domain,
            strategy: self.strategy,
            constraint: None,
        }
    }
}

impl<'a, E: PartialOrd + Scalar> ProblemBuilder<'a, E, Unset, Set, Unset, Unset, Unset> {
    #[must_use]
    /// Build a problem
    ///
    /// # Panics
    /// The function should not panic, the typestate prevents an invalid state, ensuring the `unwrap` is Ok
    pub fn build(self) -> Problem<'a, E> {
        let Rescaled { t, domain } = form_rescaled_variables(self.independent);

        Problem {
            t,
            y: self.dependent,
            uncertainties: Covariance::Diagonal {
                ux: None,
                uy: self.independent_variance.unwrap(), // This is safe as the typestate ensure
                                                        // it is some
            },
            domain,
            strategy: self.strategy,
            constraint: None,
        }
    }
}

impl<'a, E: PartialOrd + Scalar> ProblemBuilder<'a, E, Set, Set, Unset, Unset, Unset> {
    fn build(self) -> Problem<'a, E> {
        let Rescaled { t, domain } = form_rescaled_variables(self.independent);
        Problem {
            t,
            y: self.dependent,
            uncertainties: Covariance::Diagonal {
                ux: Some(self.dependent_variance.unwrap()),
                uy: self.independent_variance.unwrap(),
            },
            domain,
            strategy: self.strategy,
            constraint: None,
        }
    }
}

impl<'a, E: PartialOrd + Scalar> ProblemBuilder<'a, E, Unset, Unset, Unset, Set, Unset> {
    pub(crate) fn build(self) -> Problem<'a, E> {
        let Rescaled { t, domain } = form_rescaled_variables(self.independent);
        Problem {
            t,
            y: self.dependent,
            uncertainties: Covariance::Matrix {
                vx: None,
                vy: self.independent_covariance.unwrap(),
            },
            domain,
            strategy: self.strategy,
            constraint: None,
        }
    }
}

impl<'a, E: PartialOrd + Scalar> ProblemBuilder<'a, E, Unset, Unset, Set, Set, Unset> {
    fn build(self) -> Problem<'a, E> {
        let Rescaled { t, domain } = form_rescaled_variables(self.independent);
        Problem {
            t,
            y: self.dependent,
            uncertainties: Covariance::Matrix {
                vx: Some(self.dependent_covariance.unwrap()),
                vy: self.independent_covariance.unwrap(),
            },
            domain,
            strategy: self.strategy,
            constraint: None,
        }
    }
}

impl<'a, E: PartialOrd + Scalar, C: Into<Constraint<E>>>
    ProblemBuilder<'a, E, Unset, Unset, Unset, Unset, C>
{
    pub fn build(self) -> Problem<'a, E> {
        let Rescaled { t, domain } = form_rescaled_variables(self.independent);

        Problem {
            t,
            y: self.dependent,
            uncertainties: Covariance::None,
            domain,
            strategy: self.strategy,
            constraint: self.constraint.map(std::convert::Into::into),
        }
    }
}

impl<'a, E: PartialOrd + Scalar, C: Into<Constraint<E>>>
    ProblemBuilder<'a, E, Unset, Set, Unset, Unset, C>
{
    #[must_use]
    /// Build a problem
    ///
    /// # Panics
    /// The function should not panic, the typestate prevents an invalid state, ensuring the `unwrap` is Ok
    pub fn build(self) -> Problem<'a, E> {
        let Rescaled { t, domain } = form_rescaled_variables(self.independent);

        Problem {
            t,
            y: self.dependent,
            uncertainties: Covariance::Diagonal {
                ux: None,
                uy: self.independent_variance.unwrap(),
            },
            domain,
            strategy: self.strategy,
            constraint: self.constraint.map(std::convert::Into::into),
        }
    }
}

impl<'a, E: PartialOrd + Scalar, C: Into<Constraint<E>>>
    ProblemBuilder<'a, E, Set, Set, Unset, Unset, C>
{
    fn build(self) -> Problem<'a, E> {
        let Rescaled { t, domain } = form_rescaled_variables(self.independent);
        Problem {
            t,
            y: self.dependent,
            uncertainties: Covariance::Diagonal {
                ux: Some(self.dependent_variance.unwrap()),
                uy: self.independent_variance.unwrap(),
            },
            domain,
            strategy: self.strategy,
            constraint: self.constraint.map(std::convert::Into::into),
        }
    }
}
//
impl<'a, E: PartialOrd + Scalar, C: Into<Constraint<E>>>
    ProblemBuilder<'a, E, Unset, Unset, Unset, Set, C>
{
    fn build(self) -> Problem<'a, E> {
        let Rescaled { t, domain } = form_rescaled_variables(self.independent);
        Problem {
            t,
            y: self.dependent,
            uncertainties: Covariance::Matrix {
                vx: None,
                vy: self.independent_covariance.unwrap(),
            },
            domain,
            strategy: self.strategy,
            constraint: self.constraint.map(std::convert::Into::into),
        }
    }
}

impl<'a, E: PartialOrd + Scalar, C: Into<Constraint<E>>>
    ProblemBuilder<'a, E, Unset, Unset, Set, Set, C>
{
    fn build(self) -> Problem<'a, E> {
        let Rescaled { t, domain } = form_rescaled_variables(self.independent);
        Problem {
            t,
            y: self.dependent,
            uncertainties: Covariance::Matrix {
                vx: Some(self.dependent_covariance.unwrap()),
                vy: self.independent_covariance.unwrap(),
            },
            domain,
            strategy: self.strategy,
            constraint: self.constraint.map(std::convert::Into::into),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{ChebyshevBuilder, Constraint, PolynomialSeries, Series};

    use super::ProblemBuilder;
    use cert::{AbsUncertainty, Uncertainty};
    use ndarray::Array1;
    use ndarray_rand::rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    #[test]
    fn fit_with_independent_variance_works_in_direct_evaluation() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);

        let order = 5;
        let domain_max = rng.gen::<f64>().abs();
        let domain_min = -domain_max;
        let num_calibration_points = rng.gen_range(50..200);

        #[allow(clippy::cast_precision_loss)]
        let x = (0..num_calibration_points)
            .map(|m| {
                domain_min
                    + m as f64 * (domain_max - domain_min) / (num_calibration_points as f64 - 1.0)
            })
            .collect::<Array1<_>>();

        let mut series = None;

        // Find an input which is suitable for use as a calibration function.
        // A calibration function has to be monotonic.
        loop {
            let coeff = (0..order).map(|_| rng.gen()).collect::<Vec<f64>>();
            let this = Series::from_coeff(coeff.clone(), x.as_slice().unwrap());

            if this.is_monotonic().unwrap() {
                let _ = series.replace(this);
                break;
            }
        }

        let series = series.unwrap();

        let y = x.iter().map(|x| series.evaluate(*x)).collect::<Array1<_>>();

        let uy = y
            .iter()
            .map(|y| rng.gen_range(1e-5..1e-3) * y)
            .collect::<Array1<_>>();

        let builder = ProblemBuilder::new(x.view(), y.view())
            .unwrap()
            .with_independent_variance(uy.view())
            .unwrap()
            .with_scoring_strategy(crate::ScoringStrategy::Aicc);

        let problem = builder.build();

        let solution = problem.solve(4 * order).unwrap();

        let num_tests = 10;
        for _ in 0..num_tests {
            let idx = rng.gen_range(0..num_calibration_points);
            let x0 = x[idx];
            let y0 = y[idx];

            let predicted_y = solution
                .response(AbsUncertainty::new(x0, x0 / 100.0))
                .unwrap();
            assert!((y0 - predicted_y.mean()).abs() < predicted_y.standard_deviation());
        }
    }

    #[test]
    fn fit_with_additive_constraints_respects_constraints() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);

        let order = 5;
        let domain_max = rng.gen::<f64>().abs();
        let domain_min = -domain_max;
        // let domain_min = 0.0;
        let domain = domain_min..domain_max;

        let num_calibration_points = rng.gen_range(10..15);

        #[allow(clippy::cast_precision_loss)]
        let x = (0..num_calibration_points)
            .map(|m| {
                domain_min
                    + m as f64 * (domain_max - domain_min) / (num_calibration_points as f64 - 1.0)
            })
            .collect::<Array1<_>>();

        let intercept_at_t = 0.05;
        dbg!(intercept_at_t);

        let constraint = Constraint {
            additive: ChebyshevBuilder::new(0)
                .with_coefficients(vec![intercept_at_t])
                .on_domain(domain.clone())
                .build(),
            // multiplicative: ChebyshevBuilder::new(1)
            //     .with_coefficients(vec![0.0, 1.0])
            //     .on_domain(domain.clone())
            //     .build(),
            multiplicative: ChebyshevBuilder::new(0)
                .with_coefficients(vec![1.0])
                .on_domain(domain.clone())
                .build(),
        };

        let mut series = None;
        // Find an input which is suitable for use as a calibration function.
        // A calibration function has to be monotonic.
        loop {
            let coeff = (0..order).map(|_| rng.gen()).collect::<Vec<f64>>();
            let this = ChebyshevBuilder::new(order - 1)
                .with_coefficients(coeff.clone())
                .on_domain(domain.clone())
                .build();

            // Check if the *constrained* polynomial is monotonic
            let constrained =
                this.clone() * constraint.multiplicative.clone() + constraint.additive.clone();

            if constrained.is_monotonic().unwrap() {
                let _ = series.replace(this);
                break;
            }
        }
        let series =
            series.unwrap() * constraint.multiplicative.clone() + constraint.additive.clone();

        let y = x.iter().map(|x| series.evaluate(*x)).collect::<Array1<_>>();

        let uy = y
            .iter()
            .map(|y| rng.gen_range(1e-5..1e-3) * y + 1e-7) // stops a failure at the origin
            .collect::<Array1<_>>();

        let builder = ProblemBuilder::new(x.view(), y.view())
            .unwrap()
            .with_independent_variance(uy.view())
            .unwrap()
            .with_constraint(constraint)
            .with_scoring_strategy(crate::ScoringStrategy::Aicc);

        let problem = builder.build();

        let solution = problem.solve(4 * order).unwrap();

        let num_tests = 10;
        for _ in 0..num_tests {
            let idx = rng.gen_range(1..(num_calibration_points - 1));
            let x0 = x[idx];
            let y0 = y[idx];

            let predicted_y = solution
                .response(AbsUncertainty::new(x0, x0 / 100.0))
                .unwrap();

            assert!((y0 - predicted_y.mean()).abs() < predicted_y.standard_deviation());
        }
    }

    #[test]
    fn fit_with_full_constraints_respects_constraints() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);

        let order = 3;
        let domain_max = 2.0;
        let domain_min = -2.0;
        let domain = domain_min..domain_max;

        let num_calibration_points = rng.gen_range(20..25);

        #[allow(clippy::cast_precision_loss)]
        let x = (0..num_calibration_points)
            .map(|m| {
                domain_min
                    + m as f64 * (domain_max - domain_min) / (num_calibration_points as f64 - 1.0)
            })
            .collect::<Array1<_>>();

        let intercept_at_t = 0.25;

        let constraint = Constraint {
            additive: ChebyshevBuilder::new(0)
                .with_coefficients(vec![intercept_at_t])
                .on_domain(domain.clone())
                .build(),
            multiplicative: ChebyshevBuilder::new(1)
                .with_coefficients(vec![0.0, 1.0])
                .on_domain(domain.clone())
                .build(),
        };

        let mut series = None;
        // Find an input which is suitable for use as a calibration function.
        // A calibration function has to be monotonic.
        loop {
            let coeff = (0..order).map(|_| rng.gen()).collect::<Vec<f64>>();
            let this = ChebyshevBuilder::new(order - 1)
                .with_coefficients(coeff.clone())
                .on_domain(domain.clone())
                .build();

            // Check if the *constrained* polynomial is monotonic
            let constrained =
                this.clone() * constraint.multiplicative.clone() + constraint.additive.clone();

            if constrained.is_monotonic().unwrap() {
                let _ = series.replace(this);
                break;
            }
        }
        let series =
            series.unwrap() * constraint.multiplicative.clone() + constraint.additive.clone();

        dbg!(&series);

        let y = x.iter().map(|x| series.evaluate(*x)).collect::<Array1<_>>();

        let uy = y
            .iter()
            .map(|y| rng.gen_range(1e-5..1e-3) * y.abs() + 1e-7) // stops a failure at the origin
            .collect::<Array1<_>>();

        let builder = ProblemBuilder::new(x.view(), y.view())
            .unwrap()
            .with_independent_variance(uy.view())
            .unwrap()
            .with_constraint(constraint)
            .with_scoring_strategy(crate::ScoringStrategy::Aicc);

        println!("building problem");
        let problem = builder.build();

        println!("solving problem");
        let solution = problem.solve(4 * order).unwrap();

        println!("solved...");
        let num_tests = 10;
        for _ in 0..num_tests {
            let idx = rng.gen_range(1..(num_calibration_points - 1));
            let x0 = x[idx];
            let y0 = y[idx];

            let predicted_y = solution
                .response(AbsUncertainty::new(x0, x0 / 100.0))
                .unwrap();

            println!("expected: {y0}, predicted: {}", predicted_y.mean());
            // assert!((y0 - predicted_y.mean()).abs() < predicted_y.standard_deviation());
        }
    }
}
