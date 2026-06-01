# lau-time-series

> Time series analysis in Rust — decomposition, ARIMA forecasting, exponential smoothing, change point detection, anomaly detection, spectral analysis, and telemetry pipelines.

## What This Does

`lau-time-series` is a multi-module time series library covering the full analysis pipeline: from raw data ingestion through decomposition, forecasting, anomaly detection, and spectral analysis. It also includes a dedicated **telemetry module** for monitoring agent/system metrics like response times, error rates, and CPU usage.

The library is designed for the PLATO ecosystem but is general-purpose — any `Vec<f64>` of observations is fair game.

---

## Key Idea

A single, cohesive toolkit that takes you from raw time series data to actionable insights:

1. **TimeSeries** — the core data structure with statistics, slicing, differencing, and resampling.
2. **Decomposition** — separate trend, seasonal, and residual components (additive & multiplicative).
3. **ACF/PACF** — autocorrelation and partial autocorrelation for understanding serial dependence.
4. **ARIMA** — autoregressive and ARIMA model fitting with forecasting.
5. **Smoothing** — single, double, and Holt-Winters exponential smoothing.
6. **Change Point Detection** — CUSUM, PELT, and binary segmentation algorithms.
7. **Anomaly Detection** — Z-score, IQR, and isolation forest methods.
8. **Spectral Analysis** — FFT-based frequency detection and power spectral density.
9. **Telemetry** — domain-specific pipeline for system metrics with degradation detection.

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-time-series = "0.1"
```

Requires **Rust 2021 edition** or later. Dependencies: `serde`, `nalgebra`, `rand`.

---

## Quick Start

```rust
use lau_time_series::*;
use lau_time_series::decomposition::*;
use lau_time_series::smoothing::*;
use lau_time_series::arima::*;
use lau_time_series::spectral::*;

// 1. Create a time series
let ts = TimeSeries::from_values("sensor-A", vec![
    10.0, 12.0, 14.0, 11.0, 13.0, 15.0, 12.0, 14.0, 16.0, 13.0,
    15.0, 17.0, 14.0, 16.0, 18.0, 15.0, 17.0, 19.0, 16.0, 18.0,
]);

// 2. Statistics
println!("Mean: {}, Std: {}", ts.mean(), ts.std_dev());

// 3. Decompose (additive, period=4)
let decomp = decompose(&ts, 4);
println!("Trend: {:?}", &decomp.trend[..5]);
println!("Seasonal: {:?}", &decomp.seasonal[..4]);

// 4. Forecast with Holt-Winters
let hw = holt_winters(&ts.values, 4, 0.3, 0.1, 0.3);
let forecast = hw.forecast(4, Some(4));
println!("Next 4 values: {:?}", forecast);

// 5. Find dominant frequency
let freq = dominant_frequency(&ts.values);
println!("Dominant frequency: {}", freq);
```

---

## API Reference

### `series` — Core Time Series

```rust
pub struct TimeSeries {
    pub timestamps: Vec<f64>,
    pub values: Vec<f64>,
    pub name: String,
}
```

| Method | Description |
|--------|-------------|
| `new(name)` | Empty series. |
| `from_vec(name, timestamps, values)` | From explicit timestamps. |
| `from_values(name, values)` | Equally-spaced starting at t=0. |
| `push(t, v)` | Append an observation. |
| `len()`, `is_empty()` | Basic queries. |
| `mean()`, `variance()`, `std_dev()` | Sample statistics (Bessel-corrected variance). |
| `min()`, `max()`, `median()` | Order statistics. |
| `slice(range)` | Sub-series by index range. |
| `diff()` | First differences: y[t] - y[t-1]. |
| `cumsum()` | Cumulative sum. |
| `resample_mean(step)` | Bin-average resampling. |

Implements `Index<usize>` for direct value access: `ts[0]`.

### `decomposition` — Trend + Seasonal + Residual

```rust
pub struct Decomposition {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
    pub period: usize,
}
```

| Function | Description |
|----------|-------------|
| `moving_average(values, width)` | Centered moving average (NaN at edges). |
| `decompose(ts, period)` | Additive: y = trend + seasonal + residual. |
| `decompose_multiplicative(ts, period)` | Multiplicative: y = trend × seasonal × residual. |
| `extract_trend(values, width)` | Moving average trend extraction. |
| `extract_seasonal(ts, period)` | Seasonal component only. |

### `acf` — Autocorrelation

| Function | Description |
|----------|-------------|
| `acf(values, max_lag)` | Autocorrelation function. Result[0] = 1.0. |
| `pacf(values, max_lag)` | Partial autocorrelation via Yule-Walker. |
| `ljung_box(values, lag)` | Ljung-Box Q statistic for testing independence. |

### `arima` — Forecasting Models

```rust
// AR model
let model = fit_ar(&values, p);
model.forecast(horizon);
model.mae(&values);
model.rmse(&values);

// ARMA model
let model = fit_arma(&values, p, q);
model.forecast(horizon);

// ARIMA model (integrated)
let model = fit_arima(&ts, p, d, q);
model.forecast(horizon);
```

| Function | Description |
|----------|-------------|
| `fit_ar(values, p)` | Fit AR(p) via Yule-Walker equations. |
| `fit_arma(values, p, q)` | Fit ARMA(p,q) with innovations algorithm. |
| `fit_arima(ts, p, d, q)` | Fit ARIMA(p,d,q): difference d times, then fit ARMA. |
| `ARModel::forecast(h)` | Predict h steps ahead. |
| `ARModel::mae(values)` | Mean absolute error on training data. |
| `ARModel::rmse(values)` | Root mean squared error on training data. |

### `smoothing` — Exponential Smoothing

| Function | Description |
|----------|-------------|
| `simple_exponential_smoothing(values, alpha)` | SES (level only). |
| `double_exponential_smoothing(values, alpha, beta)` | Holt's method (level + trend). |
| `holt_winters(values, period, alpha, beta, gamma)` | Holt-Winters (level + trend + seasonal). |

All return a result type with:
- `fitted: Vec<f64>` — In-sample fitted values.
- `seasonal: Vec<f64>` — Seasonal component (Holt-Winters only).
- `forecast(h, period)` — h-step ahead forecast.
- `mae(values)`, `rmse(values)` — Error metrics.

### `changepoint` — Change Point Detection

```rust
pub struct ChangePoint {
    pub index: usize,
    pub statistic: f64,
    pub confidence: f64,
}
```

| Function | Description |
|----------|-------------|
| `cusum_detect(values, threshold)` | CUSUM cumulative sum test. |
| `pelt_detect(values, penalty)` | Pruned Exact Linear Time (PELT) method. |
| `binary_segmentation(values, threshold)` | Binary segmentation (top-down). |

### `anomaly` — Anomaly Detection

```rust
pub struct Anomaly {
    pub index: usize,
    pub value: f64,
    pub score: f64,
    pub method: String,
}
```

| Function | Description |
|----------|-------------|
| `zscore_detect(values, threshold)` | Z-score based detection. |
| `iqr_detect(values, k)` | Interquartile range method. |
| `isolation_forest(values, n_trees, sample_size)` | Isolation forest (random partitioning). |

### `spectral` — Frequency Analysis

| Function | Description |
|----------|-------------|
| `fft_magnitude(values)` | FFT magnitude spectrum (zero-padded to next power of 2). |
| `dominant_frequency(values)` | Frequency with highest power. |
| `top_frequencies(values, k)` | Top k frequencies by power. |
| `band_power(values, lo_freq, hi_freq)` | Total power in a frequency band. |

### `telemetry` — System Metrics Pipeline

```rust
pub enum MetricKind {
    ResponseTime, ErrorRate, CpuUsage, MemoryUsage,
    DiskIo, NetworkThroughput, Custom(String),
}

pub struct TelemetryEvent {
    pub timestamp: f64,
    pub metric: MetricKind,
    pub value: f64,
    pub labels: Vec<String>,
}

pub enum TrendDirection { Up, Down, Flat }
```

| Function | Description |
|----------|-------------|
| `events_to_series(events, metric)` | Filter events by metric and convert to TimeSeries. |
| `analyze_telemetry(events, metric) → TelemetryReport` | Full analysis: count, mean, std, trend, anomalies, changepoints. |
| `detect_degradation(ts, window) → Vec<Degradation>` | Detect performance degradation in a metric series. |

---

## How It Works

### Full Analysis Pipeline

```
Raw Data → TimeSeries
    │
    ├── Decompose (trend + seasonal + residual)
    ├── ACF/PACF → identify AR/MA orders
    ├── Fit ARIMA → forecast
    ├── Holt-Winters → seasonal forecast
    ├── Change Point Detection → structural breaks
    ├── Anomaly Detection → outliers
    └── Spectral Analysis → periodicities
```

### Decomposition (Additive)

1. Compute centered moving average of width = period → **trend**.
2. Subtract trend from data → **detrended**.
3. Average detrended values at each season position → **seasonal**.
4. Subtract trend + seasonal from original → **residual**.

### ARIMA Fitting

1. Difference the series `d` times to achieve stationarity.
2. Fit ARMA(p, q) on the differenced series using Yule-Walker (AR part) and innovations algorithm (MA part).
3. Forecast by inverting the differencing.

### Holt-Winters

Three smoothing equations updated recursively:
- **Level:** `l_t = α(y_t - s_{t-p}) + (1-α)(l_{t-1} + b_{t-1})`
- **Trend:** `b_t = β(l_t - l_{t-1}) + (1-β)b_{t-1}`
- **Seasonal:** `s_t = γ(y_t - l_t) + (1-γ)s_{t-p}`

### Change Point Detection

- **CUSUM:** Accumulate deviations from the mean. Exceeds threshold → change point.
- **PELT:** Optimal partitioning minimizing a cost function with a per-change penalty.
- **Binary Segmentation:** Recursively split the series at the most significant change point.

### Anomaly Detection

- **Z-score:** Flag values more than `threshold` standard deviations from the mean.
- **IQR:** Flag values below Q1 - k×IQR or above Q3 + k×IQR.
- **Isolation Forest:** Randomly partition the feature space. Anomalies are isolated in fewer splits.

---

## The Math

**Autocorrelation (ACF):**

$$\rho(k) = \frac{\sum_{t=1}^{N-k}(x_t - \bar{x})(x_{t+k} - \bar{x})}{\sum_{t=1}^{N}(x_t - \bar{x})^2}$$

**Ljung-Box Q Statistic:**

$$Q = N(N+2)\sum_{k=1}^{h}\frac{\hat{\rho}_k^2}{N-k}$$

Under H₀ (independence), Q ~ χ²(h).

**Simple Exponential Smoothing:**

$$\hat{y}_{t+1} = \alpha y_t + (1-\alpha)\hat{y}_t$$

**Holt-Winters (Additive) Forecast:**

$$\hat{y}_{t+h} = l_t + h b_t + s_{t+h-p}$$

**CUSUM Statistic:**

$$S_t = \max(0, S_{t-1} + (x_t - \mu_0 - k))$$

Where `k` is the allowance/slack parameter and μ₀ is the in-control mean.

**Z-score Anomaly:**

$$z_i = \frac{|x_i - \mu|}{\sigma}$$

Flagged if z_i > threshold.

---

## Tests

56 integration tests covering:

- **TimeSeries basics:** creation, push, mean, variance, std_dev, min/max, median, slice, diff, cumsum, resample
- **Decomposition:** moving average, additive decomposition, multiplicative decomposition, trend/seasonal extraction
- **ACF/PACF:** lag-0 = 1.0, periodic signal detection, PACF basic properties, Ljung-Box
- **ARIMA:** AR(1) coefficient recovery, AR forecast, ARMA fit, ARIMA(1,1,1) fit and forecast, MAE/RMSE
- **Exponential Smoothing:** SES basic/forecast, double exponential, Holt-Winters fit/forecast, MAE/RMSE
- **Change Point Detection:** CUSUM step detection, CUSUM no-change, PELT single/multiple changes, binary segmentation
- **Anomaly Detection:** Z-score detection, IQR detection, isolation forest
- **Spectral Analysis:** FFT magnitude, dominant frequency, top frequencies, band power
- **Telemetry:** events to series, analysis with trend, degradation detection, empty input
- **Full Pipeline:** end-to-end decompose → ACF → Holt-Winters forecast → spectral → anomaly detection

Run with:
```bash
cargo test
```

---

## License

MIT
