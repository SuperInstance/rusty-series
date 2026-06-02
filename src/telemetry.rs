//! Telemetry analysis: monitoring system metrics over time.

use crate::series::TimeSeries;
use crate::decomposition;
use crate::anomaly;
use crate::changepoint;
use crate::smoothing;
use crate::spectral;

/// Telemetry metric types for system monitoring.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricKind {
    ResponseTime,
    ErrorRate,
    CpuUsage,
    MemoryUsage,
    RequestCount,
    Custom(String),
}

/// A telemetry event: a timestamped metric observation.
#[derive(Debug, Clone)]
pub struct TelemetryEvent {
    pub timestamp: f64,
    pub metric: MetricKind,
    pub value: f64,
    pub labels: Vec<(String, String)>,
}

/// Aggregated telemetry analysis result.
#[derive(Debug, Clone)]
pub struct TelemetryReport {
    pub metric: MetricKind,
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub anomalies: Vec<anomaly::Anomaly>,
    pub changepoints: Vec<changepoint::ChangePoint>,
    pub trend_direction: TrendDirection,
    pub dominant_period: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Up,
    Down,
    Flat,
}

/// Convert telemetry events into a TimeSeries.
pub fn events_to_series(events: &[TelemetryEvent], kind: &MetricKind) -> TimeSeries {
    let filtered: Vec<&TelemetryEvent> = events.iter().filter(|e| e.metric == *kind).collect();
    let mut ts = TimeSeries::new(&format!("{:?}", kind));
    for e in &filtered {
        ts.push(e.timestamp, e.value);
    }
    ts
}

/// Analyze telemetry events and produce a report.
pub fn analyze_telemetry(events: &[TelemetryEvent], kind: &MetricKind) -> TelemetryReport {
    let ts = events_to_series(events, kind);
    if ts.is_empty() {
        return TelemetryReport {
            metric: kind.clone(),
            count: 0, mean: 0.0, std_dev: 0.0,
            min: 0.0, max: 0.0,
            anomalies: vec![], changepoints: vec![],
            trend_direction: TrendDirection::Flat,
            dominant_period: 0.0,
        };
    }

    let anomalies = anomaly::zscore_detect(&ts.values, 2.5);
    let changepoints = changepoint::cusum_detect(&ts.values, ts.std_dev() * 3.0);

    // Trend direction from smoothed series
    let smoothed = smoothing::simple_exponential_smoothing(&ts.values, 0.3);
    let trend_dir = if smoothed.level.len() >= 2 {
        let first = smoothed.level[smoothed.level.len() / 4];
        let last = *smoothed.level.last().unwrap();
        let diff = last - first;
        let range = ts.max() - ts.min();
        if range > 0.0 {
            if diff / range > 0.1 { TrendDirection::Up }
            else if diff / range < -0.1 { TrendDirection::Down }
            else { TrendDirection::Flat }
        } else { TrendDirection::Flat }
    } else { TrendDirection::Flat };

    let dominant_period = if ts.len() > 4 {
        1.0 / spectral::dominant_frequency(&ts.values).max(1e-10)
    } else { 0.0 };

    TelemetryReport {
        metric: kind.clone(),
        count: ts.len(),
        mean: ts.mean(),
        std_dev: ts.std_dev(),
        min: ts.min(),
        max: ts.max(),
        anomalies,
        changepoints,
        trend_direction: trend_dir,
        dominant_period,
    }
}

/// Detect response time degradation: increases in response times.
pub fn detect_degradation(ts: &TimeSeries, window: usize) -> Vec<(usize, f64)> {
    if ts.len() < window * 2 { return vec![]; }
    let smooth = decomposition::moving_average(&ts.values, window);
    let mut degradations = Vec::new();
    for i in window..ts.len() - 1 {
        if !smooth[i].is_nan() && !smooth[i - 1].is_nan() {
            let increase = smooth[i] - smooth[i - 1];
            let baseline = smooth[i - 1].abs().max(1e-6);
            if increase / baseline > 0.2 {
                degradations.push((i, increase / baseline));
            }
        }
    }
    degradations
}
