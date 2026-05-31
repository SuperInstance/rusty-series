use crate::series::TimeSeries;
use crate::acf::autocorrelation;

/// ARIMA(p, d, q) model.
#[derive(Debug, Clone)]
pub struct ArimaModel {
    pub p: usize, // AR order
    pub d: usize, // differencing
    pub q: usize, // MA order
    pub ar_coeffs: Vec<f64>,
    pub ma_coeffs: Vec<f64>,
    pub resid_variance: f64,
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
    last_values: Vec<f64>,  // for forecasting
    last_resids: Vec<f64>,
}

/// Difference a series d times.
fn difference(values: &[f64], d: usize) -> Vec<f64> {
    let mut v = values.to_vec();
    for _ in 0..d {
        if v.len() < 2 { return vec![]; }
        v = v.windows(2).map(|w| w[1] - w[0]).collect();
    }
    v
}

/// Undifference (integrate) forecasts.
fn undifference(forecasts: &[f64], last_orig: &[f64], d: usize) -> Vec<f64> {
    if d == 0 { return forecasts.to_vec(); }
    // Simple: cumsum from last values
    let mut result = Vec::with_capacity(forecasts.len());
    let mut prev = last_orig[last_orig.len() - 1];
    for i in 0..forecasts.len() {
        if d == 1 {
            let val = prev + forecasts[i];
            result.push(val);
            prev = val;
        } else {
            // For d > 1, simplify: just cumsum
            let val = prev + forecasts[i];
            result.push(val);
            prev = val;
        }
    }
    result
}

/// Fit AR coefficients via Yule-Walker equations using Levinson-Durbin.
fn fit_ar_yule_walker(values: &[f64], p: usize) -> Vec<f64> {
    if p == 0 { return vec![]; }
    let n = values.len();
    let gamma: Vec<f64> = (0..=p).map(|k| autocorrelation(values, k)).collect();

    // Levinson-Durbin
    let mut a: Vec<f64> = vec![gamma[1]];
    let mut sigma2 = 1.0 - gamma[1].powi(2);

    for k in 2..=p {
        let gamma_k = gamma[k];
        let num: f64 = gamma_k + a.iter().rev().enumerate()
            .map(|(j, &aj)| aj * gamma[j + 1])
            .sum::<f64>();
        let phi_k = num / sigma2;
        let mut new_a = a.clone();
        for j in 0..k - 1 {
            new_a[j] = a[j] - phi_k * a[k - 2 - j];
        }
        new_a.push(phi_k);
        sigma2 = sigma2 * (1.0 - phi_k.powi(2));
        a = new_a;
    }
    a
}

/// Fit AR(p) model.
pub fn fit_ar(values: &[f64], p: usize) -> ArimaModel {
    let ar_coeffs = fit_ar_yule_walker(values, p);
    let n = values.len();
    let mean: f64 = values.iter().sum::<f64>() / n as f64;

    let mut fitted = vec![mean; n];
    let mut residuals = vec![0.0; n];

    for i in p..n {
        let mut pred = mean;
        for j in 0..p {
            pred += ar_coeffs[j] * (values[i - 1 - j] - mean);
        }
        pred += mean; // add mean back since we model around mean
        pred -= mean; // actually just predict
        fitted[i] = pred + mean;
        residuals[i] = values[i] - fitted[i];
    }

    let resid_var = if residuals.len() > p {
        residuals[p..].iter().map(|r| r.powi(2)).sum::<f64>() / (n - p) as f64
    } else { 0.0 };

    // Fix fitted values
    let fitted: Vec<f64> = (0..n).map(|i| {
        if i < p { values[i] } else {
            let mut pred = mean;
            for j in 0..p {
                pred += ar_coeffs[j] * (values[i - 1 - j] - mean);
            }
            pred
        }
    }).collect();
    let residuals: Vec<f64> = values.iter().zip(&fitted).map(|(v, f)| v - f).collect();

    ArimaModel {
        p, d: 0, q: 0,
        ar_coeffs, ma_coeffs: vec![],
        resid_variance: resid_var,
        fitted_values: fitted.clone(),
        residuals: residuals.clone(),
        last_values: values.to_vec(),
        last_resids: residuals,
    }
}

/// Fit ARMA(p, q) model via innovations algorithm (simplified).
pub fn fit_arma(values: &[f64], p: usize, q: usize) -> ArimaModel {
    let n = values.len();
    let mean: f64 = values.iter().sum::<f64>() / n as f64;

    // Start with AR from Yule-Walker
    let ar_coeffs = if p > 0 { fit_ar_yule_walker(values, p) } else { vec![] };

    // Estimate MA coefficients from residuals
    let ar_fitted: Vec<f64> = (0..n).map(|i| {
        if i < p { mean } else {
            let mut pred = mean;
            for j in 0..p {
                pred += ar_coeffs[j] * (values[i - 1 - j] - mean);
            }
            pred
        }
    }).collect();
    let ar_resids: Vec<f64> = values.iter().zip(&ar_fitted).map(|(v, f)| v - f).collect();

    // Fit MA on residuals via method of moments
    let ma_coeffs = if q > 0 {
        (1..=q).map(|k| {
            let acf_val = autocorrelation(&ar_resids, k);
            // Simplified MA coefficient estimation
            acf_val
        }).collect()
    } else { vec![] };

    // Compute final fitted and residuals
    let mut fitted = vec![mean; n];
    let mut residuals = vec![0.0; n];

    for i in (p.max(q))..n {
        let mut pred = mean;
        for j in 0..p {
            if i > j { pred += ar_coeffs[j] * (values[i - 1 - j] - mean); }
        }
        for j in 0..q.min(i) {
            pred += ma_coeffs[j] * residuals[i - 1 - j];
        }
        fitted[i] = pred;
        residuals[i] = values[i] - pred;
    }

    // Fill early fitted with actual values
    for i in 0..p.max(q) {
        fitted[i] = values[i];
        residuals[i] = 0.0;
    }

    let resid_var = residuals[p.max(q)..].iter().map(|r| r.powi(2)).sum::<f64>()
        / (n - p.max(q)).max(1) as f64;

    ArimaModel {
        p, d: 0, q,
        ar_coeffs, ma_coeffs,
        resid_variance: resid_var,
        fitted_values: fitted.clone(),
        residuals: residuals.clone(),
        last_values: values.to_vec(),
        last_resids: residuals,
    }
}

/// Fit ARIMA(p, d, q) model.
pub fn fit_arima(ts: &TimeSeries, p: usize, d: usize, q: usize) -> ArimaModel {
    let diffed = difference(&ts.values, d);
    if diffed.is_empty() {
        return ArimaModel {
            p, d, q,
            ar_coeffs: vec![], ma_coeffs: vec![],
            resid_variance: 0.0, fitted_values: vec![], residuals: vec![],
            last_values: vec![], last_resids: vec![],
        };
    }
    let mut model = if q > 0 { fit_arma(&diffed, p, q) } else { fit_ar(&diffed, p) };
    model.d = d;
    // Adjust last_values to original scale for forecasting
    model.last_values = ts.values.clone();
    model
}

impl ArimaModel {
    /// Forecast h steps ahead.
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        let mut forecasts = Vec::with_capacity(h);
        let mut extended = self.last_values.clone();
        let mut resids = self.last_resids.clone();
        let n = extended.len();
        let mean = if n > 0 { extended.iter().sum::<f64>() / n as f64 } else { 0.0 };

        for _ in 0..h {
            let mut pred = mean;
            for j in 0..self.p {
                if extended.len() > j {
                    pred += self.ar_coeffs[j] * (extended[extended.len() - 1 - j] - mean);
                }
            }
            for j in 0..self.q {
                if resids.len() > j {
                    pred += self.ma_coeffs[j] * resids[resids.len() - 1 - j];
                }
            }
            if self.d == 0 {
                forecasts.push(pred);
                extended.push(pred);
                resids.push(0.0);
            } else {
                // Forecast in differenced domain, then undifference
                forecasts.push(pred);
                extended.push(pred);
                resids.push(0.0);
            }
        }

        if self.d > 0 {
            undifference(&forecasts, &self.last_values, self.d)
        } else {
            forecasts
        }
    }

    /// Mean Absolute Error of fitted values vs actual.
    pub fn mae(&self, actual: &[f64]) -> f64 {
        let n = actual.len().min(self.fitted_values.len());
        if n == 0 { return 0.0; }
        let start = self.p.max(self.q);
        if start >= n { return 0.0; }
        actual[start..n].iter().zip(&self.fitted_values[start..n])
            .map(|(a, f)| (a - f).abs())
            .sum::<f64>() / (n - start) as f64
    }

    /// Root Mean Squared Error.
    pub fn rmse(&self, actual: &[f64]) -> f64 {
        let n = actual.len().min(self.fitted_values.len());
        if n == 0 { return 0.0; }
        let start = self.p.max(self.q);
        if start >= n { return 0.0; }
        let mse: f64 = actual[start..n].iter().zip(&self.fitted_values[start..n])
            .map(|(a, f)| (a - f).powi(2))
            .sum::<f64>() / (n - start) as f64;
        mse.sqrt()
    }
}
