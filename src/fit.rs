use std::{marker::PhantomData, ops::{Range, MulAssign}};

use ndarray::{Array2, ScalarOperand, Array1, ArrayView1, ArrayView2, s, Axis, arr1};
use ndarray_linalg::{Scalar, Inverse, Lapack, LeastSquaresSvd};
use num_traits::Signed;

use crate::{ChebyshevPolynomial, Result};

enum Covariance<'a, E> {
    None,
    Uncertainty{ ux: Option<ArrayView1<'a, E>>, uy: ArrayView1<'a, E> },
    Covariance{ vx: Option<ArrayView2<'a, E>>, vy: ArrayView2<'a, E> },
}

enum ScoringStrategy {
    AIC,
    AICc,
    BIC,
}

struct Problem<'a, E> {
    t: Array1<E>,
    y: ArrayView1<'a, E>,
    uncertainties: Covariance<'a, E>,
    domain: Range<E>,
    strategy: ScoringStrategy,
}

impl<'a, E> Problem<'a, E>
where
    E:  Copy + std::fmt::Debug + ScalarOperand + std::ops::AddAssign + Scalar<Real = E> + Lapack + std::cmp::PartialOrd + Signed,
{
    fn new(x: &'a Array1<E>, y: &'a Array1<E>, uncertainties: Covariance<'a, E>, strategy: ScoringStrategy) -> Self {
        let x_max = x.iter().max_by(|a, b| a.partial_cmp(&b).unwrap()).unwrap().clone();
        let x_min = x.iter().min_by(|a, b| a.partial_cmp(&b).unwrap()).unwrap().clone();

        let t = x.into_iter().map(|&x| (x + x - x_min - x_max) / (x_max - x_min)).collect::<Vec<_>>();

        Self {
            t: arr1(&t),
            y: y.view(),
            uncertainties,
            domain: Range { start: x_min, end: x_max },
            strategy,
        }
    }

    fn m(&'a self) -> usize {
        self.t.len()
    }

    fn solve(&'a self, n_max: usize) -> Result<()> {
        let mut fits = vec![];
        for n in 1..n_max {
            match self.fit(n) {
                Ok(fit) => {
                    if fit.solution.is_monotonic()? {
                        fits.push(fit);
                    }
                    //fits.push(fit);
                }
                Err(err) => eprintln!("{:?}", err),
            }
        }
        // Scoring
        let scores = fits
            .iter()
            .map(|fit| self.score(&fit.solution))
            .collect::<Vec<_>>();

        dbg!(&scores);

        // Check the scores vec is not monotonous
        let diffs = scores.windows(2)
            .map(|window| window[1] - window[0])
            .collect::<Vec<_>>();
        if diffs.windows(2).all(|window| window[0].signum() == window[1].signum()) {
            panic!("no minimum found");
        }
        let best_score = scores.iter().min_by(|a, b| a.partial_cmp(&b).unwrap()).unwrap().clone();
        let scores = scores.into_iter()
            .map(|score| score - best_score)
            .collect::<Vec<_>>();


        let index = scores.iter().position(|&score| score == E::zero()).unwrap();

        let best_fit = fits.swap_remove(index);

        let nu = self.m() - best_fit.solution.n() - 1;


        println!("{:?}", best_score);
        dbg!(&best_fit.solution);
        Ok(())
        // match best_score < self.chi_2_percentile(nu) {
        //     true => Ok(()),
        //     false => {
        //         println!("{best_score:?}, {:?}", self.chi_2_percentile(nu));
        //         panic!();
        //     }
        // }
    }


    fn score(&'a self, fit: &ChebyshevPolynomial<E>) -> E {
        let chi_2_score = self.chi_2(fit);
        dbg!(&chi_2_score);
        match self.strategy {
            ScoringStrategy::AIC => chi_2_score + E::from(2 * fit.n()).unwrap(),
            ScoringStrategy::AICc => {
                let n = E::from(fit.n()).unwrap();
                chi_2_score + (E::one() + E::one()) * n
                    + (E::one() + E::one()) * (n + E::one()) * (n + E::one() + E::one())
                        / (E::from(self.m()).unwrap() - n - E::one())
            }
            ScoringStrategy::BIC => chi_2_score + (E::from(fit.n()).unwrap() + E::one()) * E::from(self.m()).unwrap().ln(),
        }
    }

    fn chi_2(&'a self, fit: &ChebyshevPolynomial<E>) -> E {
        self.t.iter()
            .zip(self.y)
            .fold(E::zero(), |a, (t, y)| a + (*y - fit.eval(*t)).powi(2))
    }

    fn fit(&'a self, n: usize) -> Result<ChebyshevFitResult<E>> {
        let design_matrix = design_matrix(self.t.view(), self.m(), n)?;
        let result = match self.uncertainties {
            Covariance::None => todo!(),
            Covariance::Uncertainty { ux, uy } if ux.is_none() => weighted_least_squares(self.y, uy, design_matrix),
            Covariance::Uncertainty { ux, uy } => todo!(),
            Covariance::Covariance { vx, vy } if vx.is_none() => todo!(),
            Covariance::Covariance { vx, vy } => todo!(),
        }?;

        Ok( ChebyshevFitResult {
            solution: ChebyshevPolynomial { coeff: result.solution.to_vec(), domain: self.domain.clone(), window: Range { start: -E::one(), end: E::one() } },
            covariance: result.covariance,
        })
    }
}

fn design_matrix<'a, E, I>(t: I, m: usize, n: usize) -> Result<Array2<E>>
where
    E: Lapack + PartialOrd + Scalar<Real = E> + ScalarOperand,
    I: IntoIterator<Item = &'a E>,
{
    let poly = ChebyshevPolynomial::constant(n);
    let rows = t
        .into_iter()
        .flat_map(|t| {
            poly.eval_as_vec(*t)
        })
        .collect::<Vec<_>>();

    Ok(Array2::from_shape_vec((m, n), rows)?)
}

fn outer_product<T: Scalar>(a: &Array1<T>, b: &Array1<T>) -> Result<Array2<T>> {
    let a: Array2<T> = a.clone().into_shape((a.len(), 1))?;
    let b: Array2<T> = b.clone().into_shape((1, b.len()))?;

    Ok(ndarray::linalg::kron(&a, &b))
}

struct ChebyshevFitResult<E> {
    solution: ChebyshevPolynomial<E>,
    covariance: Array2<E>,
}

#[derive(Debug)]
struct PolyfitResult<E> {
    solution: Array1<E>,
    covariance: Array2<E>,
}

fn weighted_least_squares<'a, E: Lapack + Scalar<Real = E> + ScalarOperand>(
    y: ArrayView1<'a, E>,
    uy: ArrayView1<'a, E>,
    h: Array2<E>,
) -> Result<PolyfitResult<E>> {
    let mut lhs = h;
    let rhs = y.to_owned() / uy.mapv(|x| x.powi(2));

    for (ii, uy) in uy.iter().enumerate() {
        let mut slice = lhs.slice_mut(s![ii, ..]);
        slice /= uy.powi(2);
    }

    let scaling = lhs
        .mapv(|val| val.powi(2))
        .sum_axis(Axis(0))
        .mapv(|val| val.sqrt());

    lhs /= &scaling;

    let result = lhs.least_squares(&rhs)?;

    let solution = (&result.solution.t() / &scaling).t().to_owned();

    let covariance = (lhs.t().dot(&lhs)).inv()? / outer_product(&scaling, &scaling)?;


    Ok( PolyfitResult { solution, covariance } )


}

#[cfg(test)]
mod test {
    use ndarray::arr1;
    use ndarray_rand::rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    use crate::{fit::{weighted_least_squares, design_matrix, Covariance, Problem}, chebyshev::ChebyshevPolynomial};

    #[test]
    fn weighted_least_squares_works_for_cubic_polynomial() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);

        let n = 3;
        let m = rng.gen_range(100..200);

        let coeffs = (0..n)
            .map(|_| rng.gen())
            .collect::<Vec<f64>>();

        let x = (0..m)
            .map(|x| x as f64 / m as f64)
            .collect::<Vec<f64>>();

        let y = x
            .iter()
            .map(|x| coeffs.iter().enumerate().fold(0., |a, (ii, b)| a + b * x.powi(ii as i32)))
            .collect::<Vec<_>>();

        let y = arr1(&y);

        let uy = y.iter().map(|y| y / 1000.0).collect::<Vec<_>>();
        let uy = arr1(&uy);

        let h = design_matrix(&x, m, n).unwrap();
        let result = weighted_least_squares(y.view(), uy.view(), h).unwrap();

        let coeff = result.solution.to_vec();
        let cheb = ChebyshevPolynomial {
            coeff,
            window: std::ops::Range { start: -1., end: 1. },
            domain: std::ops::Range { start: -1., end: 1. },
        };

        let yeval = x.iter()
            .map(|x| cheb.eval(*x))
            .collect::<Vec<_>>();

        for (y, calc) in y.into_iter().zip(yeval) {
            approx::assert_relative_eq!(y, calc, max_relative = 1e-7);
        }


    }

    #[test]
    fn fit_works_for_cubic() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);

        let m = rng.gen_range(100..200);

        let coeff = vec![0.6263732815125124, 0.7610862425514004, -0.10];//, 0.05, 0.045, 0.025];
        let coeff = vec![0.6263732815125124, 0.7610862425514004, -0.2, 0.05, 0.045, 0.025];
        let n = coeff.len();

        dbg!(&coeff);

        let cheb = ChebyshevPolynomial {
            coeff: coeff.clone(),
            window: std::ops::Range { start: -1., end: 1. },
            domain: std::ops::Range { start: -1., end: 1. },
        };

        dbg!(cheb.is_monotonic());

        let x = (0..m)
            .map(|x| x as f64 / (m -1) as f64)
            .map(|x| 2. * x - 1.)
            .collect::<Vec<f64>>();

        let y = x
            .iter()
            .map(|x| cheb.eval(*x))
            .collect::<Vec<_>>();

        let x = arr1(&x);
        let y = arr1(&y);

        let uy = y.iter().map(|y| y / 1000.0).collect::<Vec<_>>();
        let uy = arr1(&uy);

        let uncertainties = Covariance::Uncertainty { ux: None, uy: uy.view() };

        let problem = Problem::new(&x, &y, uncertainties, super::ScoringStrategy::AIC);

        let sol = problem.solve(20).unwrap();





    }
}
