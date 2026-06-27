//! Model selection criteria for calibration curve fitting.
//!
//! When fitting several candidate polynomial degrees, the calibration routine
//! must decide which model provides the best balance between accuracy and
//! complexity. `ScoringStrategy` defines the criterion used to rank candidate
//! fits.
//!
//! All strategies operate on the goodness-of-fit statistic (typically the
//! χ² statistic) and apply different penalties for model complexity.
//!
//! ## Available strategies
//!
//! - [`ScoringStrategy::ChiSquare`] minimises the χ² statistic alone. This
//!   favours the closest fit to the observations and does not penalise
//!   additional polynomial coefficients.
//!
//! - [`ScoringStrategy::Aic`] uses Akaike's Information Criterion,
//!
//!   ```text
//!   AIC = χ² + 2k
//!   ```
//!
//!   where *k* is the number of fitted coefficients.
//!
//!   AIC attempts to minimise expected information loss and is often a good
//!   general-purpose choice when selecting between several plausible models.
//!
//! - [`ScoringStrategy::Aicc`] applies the small-sample correction to AIC,
//!
//!   ```text
//!   AICc = AIC + 2k(k + 1)/(n - k - 1)
//!   ```
//!
//!   where *n* is the number of observations.
//!
//!   AICc should generally be preferred over AIC when the number of samples is
//!   not much larger than the number of fitted coefficients.
//!
//! - [`ScoringStrategy::Bic`] uses the Bayesian Information Criterion,
//!
//!   ```text
//!   BIC = χ² + k ln(n)
//!   ```
//!
//!   which applies a stronger penalty for model complexity than AIC. BIC tends
//!   to favour simpler calibration curves as the number of observations
//!   increases.
//!
//! ## Choosing a strategy
//!
//! For most calibration problems:
//!
//! - Use [`ScoringStrategy::Aicc`] for small or moderate datasets.
//! - Use [`ScoringStrategy::Aic`] when predictive performance is the primary
//!   objective and the sample size is large.
//! - Use [`ScoringStrategy::Bic`] when model simplicity is preferred.
//! - Use [`ScoringStrategy::ChiSquare`] when the polynomial degree is fixed and
//!   only goodness-of-fit is of interest.

use num_traits::{Float, FromPrimitive};

/// Criterion used to rank competing polynomial fits.
///
/// Lower scores are preferred.
#[derive(Copy, Clone, Debug)]
pub enum ScoringStrategy {
    /// Akaike Information Criterion (AIC).
    ///
    /// Balances goodness-of-fit against model complexity.
    Aic,

    /// Corrected Akaike Information Criterion (AICc).
    ///
    /// Recommended when the number of observations is not much larger than the
    /// number of fitted coefficients.
    Aicc,

    /// Bayesian Information Criterion (BIC).
    ///
    /// Applies a stronger complexity penalty than AIC, favouring simpler
    /// models.
    Bic,

    /// Pure χ² goodness-of-fit.
    ///
    /// No penalty is applied for additional coefficients.
    ChiSquare,
}

impl ScoringStrategy {
    pub(crate) fn score<E>(&self, degree: usize, num_samples: usize, chi_square: E) -> E
    where
        E: Float + FromPrimitive,
    {
        let two = E::one() + E::one();
        let k = E::from_usize(degree + 1).expect("degree + 1 should be representable");
        let num_samples = E::from_usize(num_samples).expect("num_samples should be representable");

        match self {
            ScoringStrategy::Aic => chi_square + two * k,
            ScoringStrategy::Aicc => {
                chi_square + two * k + two * k * (k + E::one()) / (num_samples - k - E::one())
            }
            ScoringStrategy::Bic => chi_square + k * num_samples.ln(),
            ScoringStrategy::ChiSquare => chi_square,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1.0e-12;

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= EPS,
            "expected {lhs} ≈ {rhs}, difference = {}",
            (lhs - rhs).abs()
        );
    }

    #[test]
    fn chi_square_score_returns_raw_chi_square() {
        let score = ScoringStrategy::ChiSquare.score(2, 10, 12.5_f64);

        assert_close(score, 12.5);
    }

    #[test]
    fn aic_score_matches_formula() {
        let degree = 2;
        let k = 3.0;
        let chi_square = 12.5;

        let score = ScoringStrategy::Aic.score(degree, 10, chi_square);

        assert_close(score, chi_square + 2.0 * k);
    }

    #[test]
    fn aicc_score_matches_formula() {
        let degree = 2;
        let k = 3.0;
        let n = 10.0;
        let chi_square = 12.5;

        let score = ScoringStrategy::Aicc.score(degree, 10, chi_square);

        let expected = chi_square + 2.0 * k + 2.0 * k * (k + 1.0) / (n - k - 1.0);

        assert_close(score, expected);
    }

    #[test]
    fn bic_score_matches_formula() {
        let degree = 2;
        let k = 3.0;
        let n = 10.0;
        let chi_square = 12.5;

        let score = ScoringStrategy::Bic.score(degree, 10, chi_square);

        assert_close(score, chi_square + k * n.ln());
    }

    #[test]
    fn higher_degree_increases_aic_penalty() {
        let low = ScoringStrategy::Aic.score(1, 20, 10.0_f64);
        let high = ScoringStrategy::Aic.score(3, 20, 10.0_f64);

        assert!(high > low);
    }

    #[test]
    fn higher_sample_count_increases_bic_penalty_for_fixed_degree() {
        let small_n = ScoringStrategy::Bic.score(2, 10, 10.0_f64);
        let large_n = ScoringStrategy::Bic.score(2, 100, 10.0_f64);

        assert!(large_n > small_n);
    }
}
