use crate::series::TimeSeries;

/// Exponential smoothing result.
#[derive(Debug, Clone)]
pub struct SmoothResult {
    pub fitted: Vec<f64>,
    pub level: Vec<f64>,
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub alpha: f64,
    pub beta: Option<f64>,
    pub gamma: Option<f64>,
}

/// Simple exponential smoothing (SES).
pub fn simple_exponential_smoothing(values: &[f64], alpha: f64) -> SmoothResult {
    let n = values.len();
    if n == 0 {
        return SmoothResult {
            fitted: vec![], level: vec![], trend: vec![], seasonal: vec![],
            alpha, beta: None, gamma: None,
        };
    }
    let mut level = vec![0.0; n];
    let mut fitted = vec![0.0; n];
    level[0] = values[0];
    fitted[0] = values[0];
    for i in 1..n {
        level[i] = alpha * values[i] + (1.0 - alpha) * level[i - 1];
        fitted[i] = level[i - 1]; // forecast is previous level
    }
    SmoothResult {
        fitted, level, trend: vec![], seasonal: vec![],
        alpha, beta: None, gamma: None,
    }
}

/// Double exponential smoothing (Holt's linear trend).
pub fn double_exponential_smoothing(values: &[f64], alpha: f64, beta: f64) -> SmoothResult {
    let n = values.len();
    if n == 0 {
        return SmoothResult {
            fitted: vec![], level: vec![], trend: vec![], seasonal: vec![],
            alpha, beta: Some(beta), gamma: None,
        };
    }
    let mut level = vec![0.0; n];
    let mut trend = vec![0.0; n];
    let mut fitted = vec![0.0; n];

    level[0] = values[0];
    trend[0] = if n > 1 { values[1] - values[0] } else { 0.0 };
    fitted[0] = values[0];

    for i in 1..n {
        level[i] = alpha * values[i] + (1.0 - alpha) * (level[i - 1] + trend[i - 1]);
        trend[i] = beta * (level[i] - level[i - 1]) + (1.0 - beta) * trend[i - 1];
        fitted[i] = level[i - 1] + trend[i - 1];
    }

    SmoothResult {
        fitted, level, trend, seasonal: vec![],
        alpha, beta: Some(beta), gamma: None,
    }
}

/// Holt-Winters (additive) seasonal smoothing.
pub fn holt_winters(values: &[f64], period: usize, alpha: f64, beta: f64, gamma: f64) -> SmoothResult {
    let n = values.len();
    if n < period * 2 {
        // Not enough data for seasonal decomposition, fall back to double
        return double_exponential_smoothing(values, alpha, beta);
    }

    let mut level = vec![0.0; n];
    let mut trend = vec![0.0; n];
    let mut seasonal = vec![0.0; n];
    let mut fitted = vec![0.0; n];

    // Initialize level as average of first period
    level[0] = values[..period].iter().sum::<f64>() / period as f64;
    trend[0] = 0.0;

    // Initialize seasonal factors
    let first_period_avg = level[0];
    for i in 0..period {
        seasonal[i] = values[i] - first_period_avg;
    }

    // Warm-up: fit first period
    for i in 0..period {
        fitted[i] = values[i];
    }

    for i in period..n {
        level[i] = alpha * (values[i] - seasonal[i - period])
            + (1.0 - alpha) * (level[i - 1] + trend[i - 1]);
        trend[i] = beta * (level[i] - level[i - 1]) + (1.0 - beta) * trend[i - 1];
        seasonal[i] = gamma * (values[i] - level[i]) + (1.0 - gamma) * seasonal[i - period];
        fitted[i] = level[i - 1] + trend[i - 1] + seasonal[i - period];
    }

    SmoothResult {
        fitted, level, trend, seasonal,
        alpha, beta: Some(beta), gamma: Some(gamma),
    }
}

impl SmoothResult {
    /// Forecast h steps ahead.
    pub fn forecast(&self, h: usize, period: Option<usize>) -> Vec<f64> {
        if self.level.is_empty() { return vec![]; }
        let last_level = *self.level.last().unwrap();
        let last_trend = if self.trend.is_empty() { 0.0 } else { *self.trend.last().unwrap() };

        (1..=h).map(|i| {
            let base = last_level + i as f64 * last_trend;
            if let Some(p) = period {
                if !self.seasonal.is_empty() {
                    let s_idx = self.seasonal.len().saturating_sub(p) + ((i - 1) % p);
                    if s_idx < self.seasonal.len() {
                        return base + self.seasonal[s_idx];
                    }
                }
            }
            base
        }).collect()
    }

    /// MAE against actual values.
    pub fn mae(&self, actual: &[f64]) -> f64 {
        let n = actual.len().min(self.fitted.len());
        if n == 0 { return 0.0; }
        actual[..n].iter().zip(&self.fitted[..n])
            .map(|(a, f)| (a - f).abs())
            .sum::<f64>() / n as f64
    }

    /// RMSE against actual values.
    pub fn rmse(&self, actual: &[f64]) -> f64 {
        let n = actual.len().min(self.fitted.len());
        if n == 0 { return 0.0; }
        let mse: f64 = actual[..n].iter().zip(&self.fitted[..n])
            .map(|(a, f)| (a - f).powi(2))
            .sum::<f64>() / n as f64;
        mse.sqrt()
    }
}
