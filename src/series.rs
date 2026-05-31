use serde::{Deserialize, Serialize};
use std::ops::{Index, Range};

/// A time series: a sequence of (timestamp, value) observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub timestamps: Vec<f64>,
    pub values: Vec<f64>,
    pub name: String,
}

impl TimeSeries {
    pub fn new(name: &str) -> Self {
        Self {
            timestamps: Vec::new(),
            values: Vec::new(),
            name: name.to_string(),
        }
    }

    pub fn from_vec(name: &str, timestamps: Vec<f64>, values: Vec<f64>) -> Self {
        assert_eq!(timestamps.len(), values.len(), "timestamps and values must have same length");
        Self { timestamps, values, name: name.to_string() }
    }

    /// Create from equally-spaced values starting at t=0 with step 1.
    pub fn from_values(name: &str, values: Vec<f64>) -> Self {
        let timestamps: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
        Self { timestamps, values, name: name.to_string() }
    }

    pub fn len(&self) -> usize { self.values.len() }
    pub fn is_empty(&self) -> bool { self.values.is_empty() }

    pub fn push(&mut self, t: f64, v: f64) {
        self.timestamps.push(t);
        self.values.push(v);
    }

    // --- Statistics ---

    pub fn mean(&self) -> f64 {
        if self.is_empty() { return 0.0; }
        self.values.iter().sum::<f64>() / self.len() as f64
    }

    pub fn variance(&self) -> f64 {
        if self.len() < 2 { return 0.0; }
        let m = self.mean();
        self.values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (self.len() - 1) as f64
    }

    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }

    pub fn min(&self) -> f64 {
        self.values.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn max(&self) -> f64 {
        self.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn median(&self) -> f64 {
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    // --- Slice ---

    pub fn slice(&self, range: Range<usize>) -> TimeSeries {
        TimeSeries {
            timestamps: self.timestamps[range.clone()].to_vec(),
            values: self.values[range].to_vec(),
            name: self.name.clone(),
        }
    }

    // --- Resampling ---

    /// Resample to a new frequency by averaging within each bin.
    /// `step` is the new time step. Values within [t, t+step) are averaged.
    pub fn resample_mean(&self, step: f64) -> TimeSeries {
        if self.is_empty() || step <= 0.0 { return self.clone(); }
        let mut ts = Vec::new();
        let mut vs = Vec::new();
        let t_min = self.timestamps[0];
        let t_max = self.timestamps.last().unwrap();
        let mut bin_start = t_min;
        while bin_start <= *t_max {
            let bin_end = bin_start + step;
            let sum: f64 = self.values.iter().zip(&self.timestamps)
                .filter(|(_, t)| **t >= bin_start && **t < bin_end)
                .map(|(v, _)| *v)
                .sum();
            let count = self.timestamps.iter().filter(|t| **t >= bin_start && **t < bin_end).count();
            if count > 0 {
                ts.push(bin_start + step / 2.0);
                vs.push(sum / count as f64);
            }
            bin_start = bin_end;
        }
        TimeSeries::from_vec(&format!("{}_resampled", self.name), ts, vs)
    }

    /// Difference the series: y[t] = x[t] - x[t-1]
    pub fn diff(&self) -> TimeSeries {
        if self.len() < 2 { return TimeSeries::new(&self.name); }
        let ts = self.timestamps[1..].to_vec();
        let vs: Vec<f64> = self.values.windows(2).map(|w| w[1] - w[0]).collect();
        TimeSeries::from_vec(&format!("{}_diff", self.name), ts, vs)
    }

    /// Cumulative sum
    pub fn cumsum(&self) -> TimeSeries {
        let mut cs = Vec::with_capacity(self.len());
        let mut s = 0.0;
        for v in &self.values {
            s += v;
            cs.push(s);
        }
        TimeSeries::from_vec(&format!("{}_cumsum", self.name), self.timestamps.clone(), cs)
    }
}

impl Index<usize> for TimeSeries {
    type Output = f64;
    fn index(&self, index: usize) -> &f64 { &self.values[index] }
}
