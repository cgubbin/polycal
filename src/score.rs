use num_traits::{Float, FromPrimitive};

/// Different scoring strategies for fit procedure
#[derive(Copy, Clone, Debug)]
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
