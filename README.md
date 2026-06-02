# rusty-series

> The most complete time-series library in Rust.

Decomposition, ARIMA forecasting, exponential smoothing, change point detection, anomaly detection, spectral analysis, and telemetry pipelines — all in one crate.

## Why This Exists

Python has `statsmodels`. R has `forecast`. Rust had… fragments. `rusty-series` is a single, cohesive toolkit that takes you from raw data to actionable insights without leaving Rust.

No external C dependencies. No Python FFI. Pure Rust.

## What's Inside

| Module | What It Does |
|--------|-------------|
| **TimeSeries** | Core data structure — stats, slicing, differencing, resampling |
| **Decomposition** | Additive & multiplicative: trend + seasonal + residual |
| **ACF/PACF** | Autocorrelation, partial autocorrelation, Ljung-Box test |
| **ARIMA** | AR(p), ARMA(p,q), ARIMA(p,d,q) fitting & forecasting |
| **Smoothing** | Single, double, and Holt-Winters exponential smoothing |
| **Change Point** | CUSUM, PELT, and binary segmentation |
| **Anomaly** | Z-score, IQR, and isolation forest detection |
| **Spectral** | FFT magnitude, dominant frequency, band power |
| **Telemetry** | System metrics pipeline with degradation detection |

## Quick Start

```toml
[dependencies]
rusty-series = "0.1"
```

```rust
use rusty_series::*;
use rusty_series::decomposition::*;
use rusty_series::smoothing::*;
use rusty_series::arima::*;
use rusty_series::spectral::*;

fn main() {
    // Create a time series
    let ts = TimeSeries::from_values("sensor-A", vec![
        10.0, 12.0, 14.0, 11.0, 13.0, 15.0, 12.0, 14.0, 16.0, 13.0,
        15.0, 17.0, 14.0, 16.0, 18.0, 15.0, 17.0, 19.0, 16.0, 18.0,
    ]);

    println!("Mean: {:.2}, Std: {:.2}", ts.mean(), ts.std_dev());

    // Decompose into trend + seasonal + residual
    let decomp = decompose(&ts, 4);
    println!("Trend:     {:?}", &decomp.trend[..5]);
    println!("Seasonal:  {:?}", &decomp.seasonal[..4]);

    // Forecast 4 steps ahead with Holt-Winters
    let hw = holt_winters(&ts.values, 4, 0.3, 0.1, 0.3);
    let forecast = hw.forecast(4, Some(4));
    println!("Forecast:  {:?}", forecast);

    // Find dominant frequency
    let freq = dominant_frequency(&ts.values);
    println!("Dominant freq: {:.4}", freq);
}
```

## Compared to Python's statsmodels

| Feature | `statsmodels` (Python) | `rusty-series` |
|---------|----------------------|-----------------|
| Decomposition (additive/multiplicative) | ✅ | ✅ |
| ARIMA | ✅ | ✅ |
| Exponential smoothing | ✅ | ✅ |
| Change point detection | ❌ (separate pkg) | ✅ |
| Anomaly detection | ❌ (separate pkg) | ✅ |
| Spectral analysis | ✅ (scipy) | ✅ |
| Zero runtime dependencies | ❌ | ✅ |
| Compile-time type safety | ❌ | ✅ |
| Speed | Python | Native Rust |

## API Tour

### Decomposition

```rust
let decomp = decompose(&ts, 4);                    // Additive
let decomp = decompose_multiplicative(&ts, 4);     // Multiplicative
```

### ARIMA Forecasting

```rust
let model = fit_ar(&values, 2);                    // AR(2)
let model = fit_arma(&values, 2, 1);               // ARMA(2,1)
let model = fit_arima(&ts, 1, 1, 1);               // ARIMA(1,1,1)
let forecast = model.forecast(10);                  // 10 steps ahead
println!("MAE: {:.4}", model.mae(&values));
```

### Anomaly Detection

```rust
let outliers = zscore_detect(&values, 2.5);         // Z-score
let outliers = iqr_detect(&values, 1.5);            // IQR method
let outliers = isolation_forest(&values, 100, 64);  // Isolation forest
```

### Spectral Analysis

```rust
let spectrum = fft_magnitude(&values);
let freq = dominant_frequency(&values);
let top = top_frequencies(&values, 3);
let power = band_power(&values, 0.1, 0.5);
```

### Change Point Detection

```rust
let changes = cusum_detect(&values, 5.0);
let changes = pelt_detect(&values, 10.0);
let changes = binary_segmentation(&values, 5.0);
```

## Requirements

- Rust 2021 edition or later
- Dependencies: `serde`, `nalgebra`, `rand`

## Running Tests

```bash
cargo test
```

56 integration tests covering every module plus an end-to-end pipeline test.

## License

MIT OR Apache-2.0
