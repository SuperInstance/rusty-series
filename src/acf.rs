use crate::series::TimeSeries;

/// Autocorrelation at lag k.
pub fn autocorrelation(values: &[f64], lag: usize) -> f64 {
    let n = values.len();
    if n == 0 || lag >= n { return 0.0; }
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let var: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
    if var == 0.0 { return 0.0; }
    let cov: f64 = (0..n - lag)
        .map(|i| (values[i] - mean) * (values[i + lag] - mean))
        .sum::<f64>();
    cov / var
}

/// Full ACF up to max_lag.
pub fn acf(values: &[f64], max_lag: usize) -> Vec<f64> {
    (0..=max_lag).map(|k| autocorrelation(values, k)).collect()
}

/// Partial autocorrelation via Durbin-Levinson recursion.
pub fn pacf(values: &[f64], max_lag: usize) -> Vec<f64> {
    let n = values.len();
    if n == 0 { return vec![]; }
    let max_lag = max_lag.min(n - 1);
    let acf_vals = acf(values, max_lag);

    // Durbin-Levinson
    let mut phi: Vec<Vec<f64>> = Vec::new();
    let mut result = Vec::with_capacity(max_lag + 1);
    result.push(1.0); // lag 0

    if max_lag >= 1 {
        let phi1 = acf_vals[1];
        result.push(phi1);
        phi.push(vec![phi1]);
    }

    for k in 2..=max_lag {
        let prev = &phi[k - 2];
        let num: f64 = acf_vals[k] + prev.iter().enumerate()
            .map(|(j, &p)| p * acf_vals[k - 1 - j])
            .sum::<f64>();
        let den: f64 = 1.0 - prev.iter().enumerate()
            .map(|(j, &p)| p * acf_vals[j + 1])
            .sum::<f64>();
        let phi_k = if den.abs() < 1e-12 { 0.0 } else { num / den };
        result.push(phi_k);

        let mut new_phi = Vec::with_capacity(k);
        for j in 0..k - 1 {
            new_phi.push(prev[j] - phi_k * prev[k - 2 - j]);
        }
        new_phi.push(phi_k);
        phi.push(new_phi);
    }

    result
}

/// Ljung-Box test statistic for testing independence.
pub fn ljung_box(values: &[f64], lag: usize) -> f64 {
    let n = values.len() as f64;
    let acf_vals = acf(values, lag);
    let stat: f64 = (1..=lag)
        .map(|k| acf_vals[k].powi(2) / (n - k as f64))
        .sum::<f64>() * n * (n + 2.0);
    stat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acf_perfect_autocorr() {
        // Constant series => acf = 0 (var=0)
        let v = vec![5.0; 10];
        assert_eq!(autocorrelation(&v, 1), 0.0);
    }

    #[test]
    fn test_acf_lag0_is_one() {
        let v: Vec<f64> = (0..20).map(|i| (i as f64).sin()).collect();
        let acf0 = autocorrelation(&v, 0);
        assert!((acf0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_acf_decreasing() {
        let v: Vec<f64> = (0..50).map(|i| i as f64).collect();
        // Linear trend should have high ACF at lag 1
        let acf1 = autocorrelation(&v, 1);
        assert!(acf1 > 0.9);
    }
}
