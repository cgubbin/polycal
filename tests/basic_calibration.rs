use poly_series::PolynomialRoots;
use polycal::{Problem, ScoringStrategy};

const EPS: f64 = 1.0e-6;

fn assert_close(lhs: f64, rhs: f64) {
    assert!(
        (lhs - rhs).abs() <= EPS,
        "expected {lhs} ≈ {rhs}, difference = {}",
        (lhs - rhs).abs()
    );
}

#[test]
fn solves_linear_calibration_without_uncertainty() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];

    let fit = Problem::builder()
        .with_data(x, y)
        .infer_domain()
        .score_by(ScoringStrategy::Aicc)
        .without_goodness_of_fit()
        .build()
        .unwrap()
        .solve_degree(1)
        .unwrap();

    assert_close(fit.response(2.5).unwrap(), 6.0);
    assert_close(fit.stimulus(6.0).unwrap(), 2.5);
    assert!(fit.calibration_curve().is_monotonic().unwrap());
}

#[test]
fn solves_weighted_linear_calibration() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0, 3.0, 5.0, 7.0, 20.0];
    let uy = vec![0.1, 0.1, 0.1, 0.1, 1000.0];

    let fit = Problem::builder()
        .with_data(x, y)
        .with_y_uncertainty(uy)
        .infer_domain()
        .without_goodness_of_fit()
        .build()
        .unwrap()
        .solve_degree(1)
        .unwrap();

    assert_close(fit.response(2.0).unwrap(), 5.0);
    assert_close(fit.stimulus(5.0).unwrap(), 2.0);
}

#[test]
fn solve_up_to_degree_selects_usable_candidate() {
    let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let y = vec![-4.0, -2.0, 0.0, 2.0, 4.0];

    let fit = Problem::builder()
        .with_data(x, y)
        .infer_domain()
        .score_by(ScoringStrategy::Aicc)
        .without_goodness_of_fit()
        .build()
        .unwrap()
        .solve_up_to_degree(3)
        .unwrap();

    assert_close(fit.response(1.5).unwrap(), 3.0);
    assert_close(fit.stimulus(3.0).unwrap(), 1.5);
}
