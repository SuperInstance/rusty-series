use crate::series::TimeSeries;

/// Result of decomposing a time series into trend + seasonal + residual.
#[derive(Debug, Clone)]
pub struct Decomposition {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
    pub period: usize,
}

/// Moving average of width `w` (centred if odd, asymmetric if even).
pub fn moving_average(values: &[f64], w: usize) -> Vec<f64> {
    if w == 0 || values.is_empty() { return vec![]; }
    let n = values.len();
    let half = w / 2;
    let mut result = vec![f64::NAN; n];
    for i in half..n.saturating_sub(half) {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let count = end - start;
        let sum: f64 = values[start..end].iter().sum();
        result[i] = sum / count as f64;
    }
    result
}

/// Additive decomposition: y = trend + seasonal + residual
pub fn decompose(ts: &TimeSeries, period: usize) -> Decomposition {
    let n = ts.len();
    // 1. Trend via centered moving average of length `period`
    let trend = moving_average(&ts.values, period);

    // 2. Detrend
    let detrended: Vec<f64> = (0..n)
        .map(|i| if trend[i].is_nan() { f64::NAN } else { ts.values[i] - trend[i] })
        .collect();

    // 3. Seasonal: average detrended value per season position
    let mut seasonal_sums = vec![0.0; period];
    let mut seasonal_counts = vec![0usize; period];
    for i in 0..n {
        if !detrended[i].is_nan() {
            let s = i % period;
            seasonal_sums[s] += detrended[i];
            seasonal_counts[s] += 1;
        }
    }
    let seasonal: Vec<f64> = (0..n)
        .map(|i| {
            let s = i % period;
            if seasonal_counts[s] > 0 {
                seasonal_sums[s] / seasonal_counts[s] as f64
            } else {
                0.0
            }
        })
        .collect();

    // 4. Residual
    let residual: Vec<f64> = (0..n)
        .map(|i| {
            let t = if trend[i].is_nan() { 0.0 } else { trend[i] };
            ts.values[i] - t - seasonal[i]
        })
        .collect();

    Decomposition { trend, seasonal, residual, period }
}

/// Multiplicative decomposition: y = trend × seasonal × residual
pub fn decompose_multiplicative(ts: &TimeSeries, period: usize) -> Decomposition {
    let n = ts.len();
    let trend = moving_average(&ts.values, period);

    let detrended: Vec<f64> = (0..n)
        .map(|i| if trend[i].is_nan() || trend[i] == 0.0 { f64::NAN } else { ts.values[i] / trend[i] })
        .collect();

    let mut seasonal_sums = vec![0.0; period];
    let mut seasonal_counts = vec![0usize; period];
    for i in 0..n {
        if !detrended[i].is_nan() {
            let s = i % period;
            seasonal_sums[s] += detrended[i];
            seasonal_counts[s] += 1;
        }
    }
    let seasonal: Vec<f64> = (0..n)
        .map(|i| {
            let s = i % period;
            if seasonal_counts[s] > 0 { seasonal_sums[s] / seasonal_counts[s] as f64 } else { 1.0 }
        })
        .collect();

    let residual: Vec<f64> = (0..n)
        .map(|i| {
            let t = if trend[i].is_nan() { 1.0 } else { trend[i] };
            let s = seasonal[i];
            if s == 0.0 { 0.0 } else { ts.values[i] / (t * s) }
        })
        .collect();

    Decomposition { trend, seasonal, residual, period }
}

/// Extract trend component only using moving average.
pub fn extract_trend(values: &[f64], window: usize) -> Vec<f64> {
    moving_average(values, window)
}

/// Extract seasonality (assumes additive model).
pub fn extract_seasonal(ts: &TimeSeries, period: usize) -> Vec<f64> {
    decompose(ts, period).seasonal
}
