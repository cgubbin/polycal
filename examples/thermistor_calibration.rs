use polycal::{Problem, ScoringStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example: a monotonic sensor calibration.
    //
    // x is temperature in °C.
    // y is measured bridge voltage in V.
    let temperature_c = vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0];
    let voltage_v = vec![0.50, 0.82, 1.18, 1.57, 1.99, 2.44];

    let voltage_uncertainty = vec![0.01; 6];

    let fit = Problem::builder()
        .with_data(temperature_c, voltage_v)
        .with_y_uncertainty(voltage_uncertainty)
        .infer_domain()
        .score_by(ScoringStrategy::Aicc)
        .without_goodness_of_fit()
        .build()?
        .solve_up_to_degree(3)?;

    let temperature = 25.0;
    let voltage = fit.response(temperature)?;
    let temperature_standard_uncertainty = temperature / 100.0;
    let uncertain_voltage =
        fit.response_with_uncertainty(temperature, temperature_standard_uncertainty)?;

    let measured_voltage = 1.80;
    let inferred_temperature = fit.stimulus(measured_voltage)?;
    let measured_voltage_standard_uncertainty = measured_voltage / 100.0;
    let uncertain_temperature =
        fit.stimulus_estimate_first_order(measured_voltage, measured_voltage_standard_uncertainty)?;

    println!("calibration method: {:?}", fit.method());
    println!("{temperature:.1} °C -> {voltage:.4} V");
    println!("{measured_voltage:.4} V -> {inferred_temperature:.2} °C");

    println!("with uncertainty");
    println!(
        "{temperature:.1} ± {temperature_standard_uncertainty:.3} °C -> {uncertain_voltage:.4} V"
    );
    println!(
        "{measured_voltage:.1} ± {measured_voltage_standard_uncertainty:.3} V -> {uncertain_temperature:.2} °C"
    );

    Ok(())
}
