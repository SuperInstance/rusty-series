#[cfg(test)]
mod tests {
    use rusty_series::*;
    use rusty_series::decomposition::*;
    use rusty_series::acf::*;
    use rusty_series::arima::*;
    use rusty_series::smoothing::*;
    use rusty_series::changepoint::*;
    use rusty_series::anomaly::*;
    use rusty_series::spectral::*;
    use rusty_series::telemetry::*;

    // ─── TimeSeries Basics ───

    #[test]
    fn test_ts_creation() {
        let ts = TimeSeries::from_values("test", vec![1.0, 2.0, 3.0]);
        assert_eq!(ts.len(), 3);
        assert!(!ts.is_empty());
        assert_eq!(ts.name, "test");
    }

    #[test]
    fn test_ts_push() {
        let mut ts = TimeSeries::new("test");
        ts.push(0.0, 10.0);
        ts.push(1.0, 20.0);
        assert_eq!(ts.len(), 2);
        assert_eq!(ts[0], 10.0);
    }

    #[test]
    fn test_ts_mean() {
        let ts = TimeSeries::from_values("test", vec![2.0, 4.0, 6.0]);
        assert!((ts.mean() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ts_variance() {
        let ts = TimeSeries::from_values("test", vec![2.0, 4.0, 6.0]);
        // sample variance: ((2-4)^2 + (4-4)^2 + (6-4)^2) / 2 = 4
        assert!((ts.variance() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ts_std_dev() {
        let ts = TimeSeries::from_values("test", vec![2.0, 4.0, 6.0]);
        assert!((ts.std_dev() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ts_min_max() {
        let ts = TimeSeries::from_values("test", vec![3.0, 1.0, 4.0, 1.5, 9.0]);
        assert!((ts.min() - 1.0).abs() < 1e-10);
        assert!((ts.max() - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_ts_median_odd() {
        let ts = TimeSeries::from_values("test", vec![3.0, 1.0, 2.0]);
        assert!((ts.median() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ts_median_even() {
        let ts = TimeSeries::from_values("test", vec![3.0, 1.0, 4.0, 2.0]);
        assert!((ts.median() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_ts_slice() {
        let ts = TimeSeries::from_values("test", vec![10.0, 20.0, 30.0, 40.0]);
        let slice = ts.slice(1..3);
        assert_eq!(slice.len(), 2);
        assert!((slice.values[0] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_ts_diff() {
        let ts = TimeSeries::from_values("test", vec![1.0, 3.0, 6.0, 10.0]);
        let d = ts.diff();
        assert_eq!(d.len(), 3);
        assert!((d.values[0] - 2.0).abs() < 1e-10);
        assert!((d.values[1] - 3.0).abs() < 1e-10);
        assert!((d.values[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ts_cumsum() {
        let ts = TimeSeries::from_values("test", vec![1.0, 2.0, 3.0]);
        let cs = ts.cumsum();
        assert!((cs.values[0] - 1.0).abs() < 1e-10);
        assert!((cs.values[1] - 3.0).abs() < 1e-10);
        assert!((cs.values[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_ts_resample() {
        let ts = TimeSeries::from_vec("test", vec![0.0, 1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0, 40.0]);
        let resampled = ts.resample_mean(2.0);
        assert!(resampled.len() >= 1);
    }

    // ─── Decomposition ───

    #[test]
    fn test_moving_average() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = moving_average(&v, 3);
        assert!(!ma[1].is_nan());
        assert!((ma[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_decompose_additive() {
        // y = trend + seasonal + noise, period=4
        let n = 40;
        let values: Vec<f64> = (0..n).map(|i| {
            let trend = i as f64 * 0.5;
            let seasonal = if i % 4 < 2 { 2.0 } else { -2.0 };
            trend + seasonal
        }).collect();
        let ts = TimeSeries::from_values("test", values);
        let decomp = decompose(&ts, 4);
        assert_eq!(decomp.seasonal.len(), n);
        assert_eq!(decomp.residual.len(), n);
        // Residuals should be small relative to signal
        let resid_mean: f64 = decomp.residual.iter().sum::<f64>() / n as f64;
        assert!(resid_mean.abs() < 2.0);
    }

    #[test]
    fn test_decompose_multiplicative() {
        let n = 48;
        let values: Vec<f64> = (0..n).map(|i| {
            let trend = 10.0 + i as f64 * 0.2;
            let seasonal = if i % 12 < 6 { 1.2 } else { 0.8 };
            trend * seasonal
        }).collect();
        let ts = TimeSeries::from_values("test", values);
        let decomp = decompose_multiplicative(&ts, 12);
        assert_eq!(decomp.seasonal.len(), n);
    }

    #[test]
    fn test_extract_trend() {
        let v: Vec<f64> = (0..20).map(|i| i as f64 * 2.0 + (i as f64).sin()).collect();
        let trend = extract_trend(&v, 5);
        assert_eq!(trend.len(), 20);
        // Trend should be close to the linear part
        assert!(trend[10].is_finite());
    }

    #[test]
    fn test_extract_seasonal() {
        let values: Vec<f64> = (0..24).map(|i| {
            (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin()
        }).collect();
        let ts = TimeSeries::from_values("test", values);
        let seasonal = extract_seasonal(&ts, 12);
        assert_eq!(seasonal.len(), 24);
    }

    // ─── ACF / PACF ───

    #[test]
    fn test_acf_lag0() {
        let v: Vec<f64> = (0..30).map(|i| (i as f64).sin()).collect();
        let result = acf(&v, 5);
        assert!((result[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_acf_periodic() {
        // Sine wave should have negative ACF at half-period
        let v: Vec<f64> = (0..100).map(|i| (i as f64 * 2.0 * std::f64::consts::PI / 20.0).sin()).collect();
        let acf_vals = acf(&v, 30);
        // At lag ~10 (half period), ACF should be negative
        assert!(acf_vals[10] < 0.0);
        // At lag ~20 (full period), ACF should be positive
        assert!(acf_vals[20] > 0.0);
    }

    #[test]
    fn test_pacf_basic() {
        let v: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let pacf_vals = pacf(&v, 5);
        assert_eq!(pacf_vals.len(), 6); // 0..=5
        assert!((pacf_vals[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ljung_box() {
        let v: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
        let lb = ljung_box(&v, 10);
        // Structured data should have high Ljung-Box stat
        assert!(lb > 0.0);
    }

    // ─── ARIMA ───

    #[test]
    fn test_ar_fit() {
        // AR(1) with coeff ~0.8: x[t] = 0.8 * x[t-1] + small noise
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let mut v = vec![1.0];
        for i in 1..500 {
            let prev = v[i - 1];
            let noise: f64 = rng.gen::<f64>() * 0.1 - 0.05;
            v.push(0.8 * prev + noise);
        }
        let model = fit_ar(&v, 1);
        assert_eq!(model.p, 1);
        assert_eq!(model.ar_coeffs.len(), 1);
        // Coefficient should be reasonably close to 0.8
        assert!((model.ar_coeffs[0] - 0.8).abs() < 0.35);
    }

    #[test]
    fn test_ar_forecast() {
        let v: Vec<f64> = (0..50).map(|i| 0.5 * i as f64).collect();
        let model = fit_ar(&v, 2);
        let fc = model.forecast(5);
        assert_eq!(fc.len(), 5);
    }

    #[test]
    fn test_arma_fit() {
        let v: Vec<f64> = (0..60).map(|i| (i as f64 * 0.1).sin() * 10.0).collect();
        let model = fit_arma(&v, 1, 1);
        assert_eq!(model.p, 1);
        assert_eq!(model.q, 1);
        assert!(!model.fitted_values.is_empty());
    }

    #[test]
    fn test_arima_fit() {
        let ts = TimeSeries::from_values("test",
            (0..80).map(|i| i as f64 + (i as f64 * 0.1).sin() * 3.0).collect()
        );
        let model = fit_arima(&ts, 1, 1, 1);
        assert_eq!(model.d, 1);
        let fc = model.forecast(5);
        assert_eq!(fc.len(), 5);
        // Forecasts should be reasonable (near continuation of trend)
        assert!(fc[0] > 50.0);
    }

    #[test]
    fn test_arima_mae_rmse() {
        let v: Vec<f64> = (0..50).map(|i| i as f64 * 0.5).collect();
        let model = fit_ar(&v, 2);
        let mae = model.mae(&v);
        let rmse = model.rmse(&v);
        assert!(mae >= 0.0);
        assert!(rmse >= 0.0);
    }

    // ─── Exponential Smoothing ───

    #[test]
    fn test_ses_basic() {
        let v = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let result = simple_exponential_smoothing(&v, 0.5);
        assert_eq!(result.fitted.len(), 5);
        assert!((result.fitted[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_ses_forecast() {
        let v = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let result = simple_exponential_smoothing(&v, 0.3);
        let fc = result.forecast(3, None);
        assert_eq!(fc.len(), 3);
        // Should be close to the last level
        assert!(fc[0] > 12.0);
    }

    #[test]
    fn test_double_exponential() {
        let v: Vec<f64> = (0..20).map(|i| i as f64 * 2.0).collect();
        let result = double_exponential_smoothing(&v, 0.5, 0.5);
        assert_eq!(result.fitted.len(), 20);
        let fc = result.forecast(3, None);
        // Should continue upward trend
        assert!(fc[2] > fc[0]);
    }

    #[test]
    fn test_holt_winters() {
        // Seasonal data: period 4, trend up
        let n = 48;
        let v: Vec<f64> = (0..n).map(|i| {
            let trend = i as f64 * 0.3;
            let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 4.0).sin() * 3.0;
            trend + seasonal
        }).collect();
        let result = holt_winters(&v, 4, 0.3, 0.1, 0.3);
        assert_eq!(result.fitted.len(), n);
        assert_eq!(result.seasonal.len(), n);
    }

    #[test]
    fn test_holt_winters_forecast() {
        let n = 48;
        let v: Vec<f64> = (0..n).map(|i| {
            let trend = i as f64 * 0.3;
            let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 4.0).sin() * 3.0;
            trend + seasonal
        }).collect();
        let result = holt_winters(&v, 4, 0.3, 0.1, 0.3);
        let fc = result.forecast(8, Some(4));
        assert_eq!(fc.len(), 8);
    }

    #[test]
    fn test_smoothing_mae() {
        let v = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let result = simple_exponential_smoothing(&v, 0.5);
        let mae = result.mae(&v);
        assert!(mae >= 0.0);
        // For smooth data, MAE should be small-ish
        assert!(mae < 3.0);
    }

    #[test]
    fn test_smoothing_rmse() {
        let v = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let result = simple_exponential_smoothing(&v, 0.5);
        let rmse = result.rmse(&v);
        assert!(rmse >= 0.0);
    }

    // ─── Change Point Detection ───

    #[test]
    fn test_cusum_step_change() {
        let mut v = vec![0.0; 50];
        for i in 50..100 { v.push(5.0); }
        let cps = cusum_detect(&v, 3.0);
        assert!(!cps.is_empty());
        // Should detect near index 50
        let near_50 = cps.iter().any(|cp| cp.index > 40 && cp.index < 60);
        assert!(near_50);
    }

    #[test]
    fn test_cusum_no_change() {
        let v = vec![5.0; 100];
        let cps = cusum_detect(&v, 10.0);
        assert!(cps.is_empty());
    }

    #[test]
    fn test_pelt_single_change() {
        let mut v: Vec<f64> = (0..50).map(|_| 0.0).collect();
        v.extend((0..50).map(|_| 10.0));
        let cps = pelt_detect(&v, 5.0);
        assert!(!cps.is_empty());
        // Should find a changepoint near 50
        assert!(cps.iter().any(|cp| cp.index > 30 && cp.index < 70));
    }

    #[test]
    fn test_pelt_multiple_changes() {
        let mut v: Vec<f64> = (0..30).map(|_| 0.0).collect();
        v.extend((0..30).map(|_| 5.0));
        v.extend((0..30).map(|_| 0.0));
        let cps = pelt_detect(&v, 5.0);
        assert!(cps.len() >= 1);
    }

    #[test]
    fn test_binary_segmentation() {
        let mut v: Vec<f64> = (0..40).map(|_| 0.0).collect();
        v.extend((0..40).map(|_| 10.0));
        let cps = binary_segmentation(&v, 5.0);
        assert!(!cps.is_empty());
    }

    // ─── Anomaly Detection ───

    #[test]
    fn test_zscore_detect_outliers() {
        let mut v: Vec<f64> = (0..50).map(|i| i as f64).collect();
        v.push(1000.0); // obvious outlier
        let anomalies = zscore_detect(&v, 3.0);
        assert!(!anomalies.is_empty());
        assert!(anomalies.iter().any(|a| a.value == 1000.0));
    }

    #[test]
    fn test_zscore_no_outliers() {
        let v: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let anomalies = zscore_detect(&v, 10.0);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_iqr_detect() {
        let mut v: Vec<f64> = (0..100).map(|i| i as f64).collect();
        v.push(500.0);
        let anomalies = iqr_detect(&v, 1.5);
        assert!(!anomalies.is_empty());
        assert!(anomalies.iter().any(|a| a.value == 500.0));
    }

    #[test]
    fn test_iqr_no_outliers() {
        let v: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let anomalies = iqr_detect(&v, 100.0); // very wide bounds
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_isolation_forest() {
        let mut v: Vec<f64> = (0..80).map(|i| i as f64 * 0.1).collect();
        v.push(50.0); // outlier
        v.push(55.0); // outlier
        let mut forest = IsolationForest::new(50, 10);
        forest.fit(&v);
        let anomalies = forest.detect(&v, 0.6);
        // Should detect the outliers (high score)
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_isolation_forest_score() {
        let v: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let mut forest = IsolationForest::new(20, 8);
        forest.fit(&v);
        let normal_score = forest.score(5.0);
        let outlier_score = forest.score(500.0);
        assert!(outlier_score > normal_score);
    }

    // ─── Spectral Analysis ───

    #[test]
    fn test_periodogram() {
        let v: Vec<f64> = (0..128).map(|i| (i as f64 * 2.0 * std::f64::consts::PI / 16.0).sin()).collect();
        let result = periodogram(&v);
        assert!(!result.frequencies.is_empty());
        assert!(!result.periodogram.is_empty());
    }

    #[test]
    fn test_dominant_frequency_sine() {
        // Sine wave with period 20 => frequency 1/20 = 0.05
        let v: Vec<f64> = (0..200).map(|i| (i as f64 * 2.0 * std::f64::consts::PI / 20.0).sin()).collect();
        let freq = dominant_frequency(&v);
        // Should be close to 0.05
        assert!((freq - 0.05).abs() < 0.02);
    }

    #[test]
    fn test_top_frequencies() {
        let v: Vec<f64> = (0..256).map(|i| {
            (i as f64 * 2.0 * std::f64::consts::PI / 32.0).sin()
            + 0.5 * (i as f64 * 2.0 * std::f64::consts::PI / 8.0).sin()
        }).collect();
        let top = top_frequencies(&v, 2);
        assert_eq!(top.len(), 2);
        // Should include freqs near 1/32 and 1/8
        let freqs: Vec<f64> = top.iter().map(|(f, _)| *f).collect();
        assert!(freqs.iter().any(|f| (*f - 1.0/32.0).abs() < 0.02));
    }

    #[test]
    fn test_band_power() {
        let v: Vec<f64> = (0..128).map(|i| (i as f64 * 2.0 * std::f64::consts::PI / 16.0).sin()).collect();
        let power = band_power(&v, 0.0, 0.2);
        assert!(power > 0.0);
    }

    // ─── Telemetry ───

    #[test]
    fn test_events_to_series() {
        let events = vec![
            TelemetryEvent { timestamp: 0.0, metric: MetricKind::ResponseTime, value: 100.0, labels: vec![] },
            TelemetryEvent { timestamp: 1.0, metric: MetricKind::ResponseTime, value: 200.0, labels: vec![] },
            TelemetryEvent { timestamp: 0.5, metric: MetricKind::ErrorRate, value: 0.1, labels: vec![] },
        ];
        let ts = events_to_series(&events, &MetricKind::ResponseTime);
        assert_eq!(ts.len(), 2);
    }

    #[test]
    fn test_analyze_telemetry() {
        let mut events = Vec::new();
        for i in 0..50 {
            events.push(TelemetryEvent {
                timestamp: i as f64,
                metric: MetricKind::ResponseTime,
                value: 100.0 + i as f64 * 0.5,
                labels: vec![],
            });
        }
        // Add an anomaly
        events.push(TelemetryEvent {
            timestamp: 50.0,
            metric: MetricKind::ResponseTime,
            value: 5000.0,
            labels: vec![],
        });
        let report = analyze_telemetry(&events, &MetricKind::ResponseTime);
        assert_eq!(report.count, 51);
        assert!(report.mean > 0.0);
        // Should detect trend
        assert!(report.trend_direction == TrendDirection::Up || report.trend_direction == TrendDirection::Flat);
    }

    #[test]
    fn test_telemetry_degradation() {
        let mut values = vec![100.0; 30];
        values.extend(vec![300.0; 30]); // spike
        let ts = TimeSeries::from_values("response_time", values);
        let degradations = detect_degradation(&ts, 5);
        assert!(!degradations.is_empty());
    }

    #[test]
    fn test_telemetry_empty() {
        let report = analyze_telemetry(&[], &MetricKind::CpuUsage);
        assert_eq!(report.count, 0);
    }

    // ─── Integration ───

    #[test]
    fn test_full_pipeline() {
        // Generate data: trend + seasonality + noise + outlier
        let n = 120;
        let values: Vec<f64> = (0..n).map(|i| {
            let trend = i as f64 * 0.5;
            let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin() * 5.0;
            trend + seasonal
        }).collect();
        let ts = TimeSeries::from_values("pipeline_test", values);

        // Decompose
        let decomp = decompose(&ts, 12);
        assert_eq!(decomp.trend.len(), n);

        // ACF
        let acf_vals = acf(&ts.values, 10);
        assert!((acf_vals[0] - 1.0).abs() < 1e-10);

        // Forecast
        let hw = holt_winters(&ts.values, 12, 0.3, 0.1, 0.3);
        let fc = hw.forecast(12, Some(12));
        assert_eq!(fc.len(), 12);

        // Spectral
        let freq = dominant_frequency(&ts.values);
        assert!(freq > 0.0);

        // Anomaly detection
        let anomalies = zscore_detect(&ts.values, 3.0);
        // Clean data should have few anomalies
        assert!(anomalies.len() < 10);
    }
}
