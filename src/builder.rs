use ndarray::{ArrayView1, ArrayView2};
use ndarray_linalg::Scalar;
use std::marker::PhantomData;

// use crate::fit::{Covariance, PolyConstraint, Problem, ScoringStrategy};
//
use crate::problem::{Constraint, Covariance, Problem, ScoringStrategy};
use crate::utils::{form_rescaled_variables, Rescaled};

#[derive(Default)]
struct Set {}

#[derive(Default)]
struct Unset {}

struct ProblemBuilder<'a, E, DU, IU, DC, IC, C> {
    dependent: ArrayView1<'a, E>,
    independent: ArrayView1<'a, E>,
    dependent_uncertainty: Option<ArrayView1<'a, E>>,
    independent_uncertainty: Option<ArrayView1<'a, E>>,
    dependent_covariance: Option<ArrayView2<'a, E>>,
    independent_covariance: Option<ArrayView2<'a, E>>,
    constraint: Option<C>,
    strategy: ScoringStrategy,
    typestate: PhantomData<(DU, IU, DC, IC, C)>,
}

impl<'a, E> ProblemBuilder<'a, E, Unset, Unset, Unset, Unset, Unset> {
    fn new<V: Into<ArrayView1<'a, E>>>(independent: V, dependent: V) -> Self {
        Self {
            dependent: dependent.into(),
            independent: independent.into(),
            dependent_uncertainty: None,
            independent_uncertainty: None,
            dependent_covariance: None,
            independent_covariance: None,
            constraint: None,
            strategy: ScoringStrategy::ChiSquare,
            typestate: PhantomData,
        }
    }
}

impl<'a, E, DU, IU, DC, IC, C> ProblemBuilder<'a, E, DU, IU, DC, IC, C> {
    const fn with_scoring_strategy(mut self, scoring_strategy: ScoringStrategy) -> Self {
        self.strategy = scoring_strategy;
        self
    }
}

impl<'a, E, C> ProblemBuilder<'a, E, Unset, Unset, Unset, Unset, C> {
    fn with_dependent_uncertainty(
        self,
        dependent_uncertainty: impl Into<ArrayView1<'a, E>>,
    ) -> ProblemBuilder<'a, E, Set, Unset, Unset, Unset, C> {
        ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_uncertainty: Some(dependent_uncertainty.into()),
            independent_uncertainty: self.independent_uncertainty,
            dependent_covariance: self.dependent_covariance,
            independent_covariance: self.independent_covariance,
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        }
    }
}

impl<'a, E, C> ProblemBuilder<'a, E, Unset, Set, Unset, Unset, C> {
    fn with_dependent_uncertainty(
        self,
        dependent_uncertainty: impl Into<ArrayView1<'a, E>>,
    ) -> ProblemBuilder<'a, E, Set, Set, Unset, Unset, C> {
        ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_uncertainty: Some(dependent_uncertainty.into()),
            independent_uncertainty: self.independent_uncertainty,
            dependent_covariance: self.dependent_covariance,
            independent_covariance: self.independent_covariance,
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        }
    }
}

impl<'a, E, C> ProblemBuilder<'a, E, Unset, Unset, Unset, Unset, C> {
    fn with_independent_uncertainty(
        self,
        independent_uncertainty: impl Into<ArrayView1<'a, E>>,
    ) -> ProblemBuilder<'a, E, Unset, Set, Unset, Unset, C> {
        ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_uncertainty: self.dependent_uncertainty,
            independent_uncertainty: Some(independent_uncertainty.into()),
            dependent_covariance: self.dependent_covariance,
            independent_covariance: self.independent_covariance,
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        }
    }
}

impl<'a, E, C> ProblemBuilder<'a, E, Set, Unset, Unset, Unset, C> {
    fn with_independent_uncertainty(
        self,
        independent_uncertainty: impl Into<ArrayView1<'a, E>>,
    ) -> ProblemBuilder<'a, E, Set, Set, Unset, Unset, C> {
        ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_uncertainty: self.dependent_uncertainty,
            independent_uncertainty: Some(independent_uncertainty.into()),
            dependent_covariance: self.dependent_covariance,
            independent_covariance: self.independent_covariance,
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        }
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
            dependent_uncertainty: self.dependent_uncertainty,
            independent_uncertainty: self.independent_uncertainty,
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
            dependent_uncertainty: self.dependent_uncertainty,
            independent_uncertainty: self.independent_uncertainty,
            dependent_covariance: Some(dependent_covariance.into()),
            independent_covariance: self.independent_covariance,
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        }
    }
}

impl<'a, E, C> ProblemBuilder<'a, E, Unset, Unset, Unset, Unset, C> {
    fn with_independent_covariance(
        self,
        independent_covariance: impl Into<ArrayView2<'a, E>>,
    ) -> ProblemBuilder<'a, E, Unset, Unset, Unset, Set, C> {
        ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_uncertainty: self.dependent_uncertainty,
            independent_uncertainty: self.independent_uncertainty,
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
            dependent_uncertainty: self.dependent_uncertainty,
            independent_uncertainty: self.independent_uncertainty,
            dependent_covariance: self.dependent_covariance,
            independent_covariance: Some(independent_covariance.into()),
            strategy: self.strategy,
            constraint: self.constraint,
            typestate: PhantomData,
        }
    }
}

impl<'a, E, DU, IU, DC, IC> ProblemBuilder<'a, E, DU, IU, DC, IC, Unset> {
    const fn with_constraint<C>(self, constraint: C) -> ProblemBuilder<'a, E, DU, IU, DC, IC, C> {
        ProblemBuilder {
            dependent: self.dependent,
            independent: self.independent,
            dependent_uncertainty: self.dependent_uncertainty,
            independent_uncertainty: self.independent_uncertainty,
            dependent_covariance: self.dependent_covariance,
            independent_covariance: self.independent_covariance,
            strategy: self.strategy,
            constraint: Some(constraint),
            typestate: PhantomData,
        }
    }
}

impl<'a, E: PartialOrd + Scalar> ProblemBuilder<'a, E, Unset, Unset, Unset, Unset, Unset> {
    fn build(self) -> Problem<'a, E> {
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
    fn build(self) -> Problem<'a, E> {
        let Rescaled { t, domain } = form_rescaled_variables(self.independent);

        Problem {
            t,
            y: self.dependent,
            uncertainties: Covariance::Uncertainty {
                ux: None,
                uy: self.independent_uncertainty.unwrap(),
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
            uncertainties: Covariance::Uncertainty {
                ux: Some(self.dependent_uncertainty.unwrap()),
                uy: self.independent_uncertainty.unwrap(),
            },
            domain,
            strategy: self.strategy,
            constraint: None,
        }
    }
}

impl<'a, E: PartialOrd + Scalar> ProblemBuilder<'a, E, Unset, Unset, Unset, Set, Unset> {
    fn build(self) -> Problem<'a, E> {
        let Rescaled { t, domain } = form_rescaled_variables(self.independent);
        Problem {
            t,
            y: self.dependent,
            uncertainties: Covariance::Covariance {
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
            uncertainties: Covariance::Covariance {
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
    fn build(self) -> Problem<'a, E> {
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
    fn build(self) -> Problem<'a, E> {
        let Rescaled { t, domain } = form_rescaled_variables(self.independent);

        Problem {
            t,
            y: self.dependent,
            uncertainties: Covariance::Uncertainty {
                ux: None,
                uy: self.independent_uncertainty.unwrap(),
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
            uncertainties: Covariance::Uncertainty {
                ux: Some(self.dependent_uncertainty.unwrap()),
                uy: self.independent_uncertainty.unwrap(),
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
            uncertainties: Covariance::Covariance {
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
            uncertainties: Covariance::Covariance {
                vx: Some(self.dependent_covariance.unwrap()),
                vy: self.independent_covariance.unwrap(),
            },
            domain,
            strategy: self.strategy,
            constraint: self.constraint.map(std::convert::Into::into),
        }
    }
}
//
//
// #[cfg(test)]
// mod test {
//     use super::ProblemBuilder;
//     use ndarray::Array1;
//     use ndarray_rand::rand::{Rng, SeedableRng};
//     use rand_isaac::Isaac64Rng;
//     use std::ops::Range;
//
//     use crate::eval::Unsure;
//     use crate::fit::{PolyConstraint, ScoringStrategy};
//     use crate::ChebyshevPolynomial;
//
//     #[test]
//     fn fit_with_independent_uncertainty_works_in_direct_evaluation() {
//         let state = 40;
//         let mut rng = Isaac64Rng::seed_from_u64(state);
//
//         let order = 5;
//         let domain_max = rng.gen::<f64>().abs();
//         let domain_min = -domain_max;
//
//         let mut coeff = vec![];
//
//         // Find an input which is suitable for use as a calibration function.
//         // A calibration function has to be monotonic.
//         loop {
//             coeff = (0..order).map(|_| rng.gen()).collect::<Vec<f64>>();
//             let polynomial = ChebyshevPolynomial {
//                 coeff: coeff.clone(),
//                 domain: Range {
//                     start: domain_min,
//                     end: domain_max,
//                 },
//                 window: Range {
//                     start: -1.,
//                     end: 1.,
//                 },
//             };
//             if polynomial.is_monotonic().unwrap() {
//                 break;
//             }
//         }
//
//         let polynomial = ChebyshevPolynomial {
//             coeff: coeff.clone(),
//             domain: Range {
//                 start: domain_min,
//                 end: domain_max,
//             },
//             window: Range {
//                 start: -1.,
//                 end: 1.,
//             },
//         };
//
//         let num_calibration_points = rng.gen_range(5000..10000);
//         let x = (0..num_calibration_points)
//             .map(|ii| {
//                 domain_min
//                     + (domain_max - domain_min) * ii as f64 / (num_calibration_points - 1) as f64
//             })
//             .collect::<Array1<_>>();
//
//         let polynomial = ChebyshevPolynomial {
//             coeff: coeff.clone(),
//             domain: Range {
//                 start: domain_min,
//                 end: domain_max,
//             },
//             window: Range {
//                 start: -1.,
//                 end: 1.,
//             },
//         };
//
//         let y = x.iter().map(|x| polynomial.eval(*x)).collect::<Array1<_>>();
//
//         let uy = y
//             .iter()
//             .map(|y| rng.gen_range(1e-5..1e-3) * y)
//             .collect::<Array1<_>>();
//
//         let builder =
//             ProblemBuilder::new(x.view(), y.view()).with_independent_uncertainty(uy.view());
//
//         let problem = builder.build();
//
//         let solution = problem.solve(4 * order).unwrap();
//
//         let idx = rng.gen_range(0..num_calibration_points);
//         let x0 = x[idx];
//         let y0 = y[idx];
//
//         let predicted_y = solution
//             .eval_from_stimulus(Unsure {
//                 estimate: x0,
//                 standard_uncertainty: x0 / 100.0,
//             });
//
//         dbg!(&predicted_y);
//         dbg!((y0 - predicted_y.estimate).abs());
//         // assert!((y0 - predicted_y.estimate).abs() < predicted_y.standard_uncertainty);
//     }
//
//     #[test]
//     fn fit_with_constraints_respects_constraints() {
//         let state = 40;
//         let mut rng = Isaac64Rng::seed_from_u64(state);
//
//         let order = 5;
//         let domain_max = rng.gen::<f64>().abs();
//         let domain_min = 0.0;
//
//         let intercept_at_t = (- domain_min - domain_max) / (domain_max - domain_min);
//
//         let mut coeff = Vec::new();
//
//         let constraint = PolyConstraint {
//             nu: ChebyshevPolynomial {
//                 coeff: vec![-intercept_at_t, 1.],
//                 domain: Range {
//                     start: domain_min,
//                     end: domain_max,
//                 },
//                 window: Range {
//                     start: -1.,
//                     end: 1.,
//                 },
//             },
//             mu: ChebyshevPolynomial {
//                 coeff: vec![0.],
//                 domain: Range {
//                     start: domain_min,
//                     end: domain_max,
//                 },
//                 window: Range {
//                     start: -1.,
//                     end: 1.,
//                 },
//             },
//         };
//
//         // Find an input which is suitable for use as a calibration function.
//         // A calibration function has to be monotonic.
//         loop {
//             coeff = (0..order).map(|_| rng.gen()).collect::<Vec<f64>>();
//             let polynomial = ChebyshevPolynomial {
//                 coeff: coeff.clone(),
//                 domain: Range {
//                     start: domain_min,
//                     end: domain_max,
//                 },
//                 window: Range {
//                     start: -1.,
//                     end: 1.,
//                 },
//             };
//             if polynomial.is_monotonic_with_constraint(&constraint.nu).unwrap() {
//                 break;
//             }
//         }
//
//         let num_calibration_points = rng.gen_range(100..500);
//         let x = (0..num_calibration_points)
//             .map(|ii| {
//                 domain_min
//                     + (domain_max - domain_min) * ii as f64 / (num_calibration_points - 1) as f64
//             })
//             .collect::<Array1<_>>();
//
//         let polynomial = ChebyshevPolynomial {
//             coeff,
//             domain: Range {
//                 start: domain_min,
//                 end: domain_max,
//             },
//             window: Range {
//                 start: -1.,
//                 end: 1.,
//             },
//         };
//
//         let y = x.iter().map(|x| polynomial.eval(*x)).collect::<Array1<_>>();
//
//         let uy = y
//             .iter()
//             .map(|y| rng.gen_range(1e-5..1e-3) * y)
//             .collect::<Array1<_>>();
//
//         let builder = ProblemBuilder::new(x.view(), y.view())
//             .with_independent_uncertainty(uy.view())
//             .with_constraint(constraint)
//             .with_scoring_strategy(ScoringStrategy::ChiSquare);
//
//         let problem = builder.build();
//
//         let solution = problem.solve(5 * order).unwrap();
//
//         let idx = rng.gen_range(0..num_calibration_points);
//         let x0 = x[idx];
//         let y0 = y[idx];
//
//         let predicted_y = solution
//             .eval_from_stimulus(Unsure {
//                 estimate: x0,
//                 standard_uncertainty: x0 / 100.0,
//             });
//
//         dbg!(&predicted_y);
//         dbg!((y0 - predicted_y.estimate).abs());
//
//         let predicted_y = solution
//             .eval_from_stimulus(Unsure {
//                 estimate: 0.0,
//                 standard_uncertainty: x0 / 100.0,
//             });
//
//         dbg!(&predicted_y);
//     }
// }
