use crate::series::TimeSeries;
use rand::Rng;

/// Anomaly detected at a specific index.
#[derive(Debug, Clone)]
pub struct Anomaly {
    pub index: usize,
    pub value: f64,
    pub score: f64,
    pub kind: AnomalyKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyKind {
    ZScore,
    IQR,
    IsolationForest,
}

/// Z-score anomaly detection.
/// Flags values with |z| > threshold.
pub fn zscore_detect(values: &[f64], threshold: f64) -> Vec<Anomaly> {
    if values.is_empty() { return vec![]; }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let std_dev = {
        let var: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        var.sqrt()
    };
    if std_dev < 1e-12 { return vec![]; }

    values.iter().enumerate()
        .filter(|(_, &v)| ((v - mean) / std_dev).abs() > threshold)
        .map(|(i, &v)| Anomaly {
            index: i,
            value: v,
            score: ((v - mean) / std_dev).abs(),
            kind: AnomalyKind::ZScore,
        })
        .collect()
}

/// IQR-based anomaly detection.
/// Flags values below Q1 - k*IQR or above Q3 + k*IQR.
pub fn iqr_detect(values: &[f64], k: f64) -> Vec<Anomaly> {
    if values.is_empty() { return vec![]; }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let q1 = sorted[n * 25 / 100];
    let q3 = sorted[n * 75 / 100];
    let iqr = q3 - q1;

    let lower = q1 - k * iqr;
    let upper = q3 + k * iqr;

    values.iter().enumerate()
        .filter(|(_, &v)| v < lower || v > upper)
        .map(|(i, &v)| Anomaly {
            index: i,
            value: v,
            score: if v < lower { (lower - v) / iqr.max(1e-12) } else { (v - upper) / iqr.max(1e-12) },
            kind: AnomalyKind::IQR,
        })
        .collect()
}

/// Simplified Isolation Forest anomaly detection.
/// Uses random splits to isolate points; anomalies need fewer splits.
pub struct IsolationForest {
    pub num_trees: usize,
    pub max_depth: usize,
    trees: Vec<IsolationTree>,
}

#[derive(Debug, Clone)]
struct IsolationTree {
    split_feature: usize, // always 0 for univariate
    split_value: f64,
    left: Option<Box<IsolationTree>>,
    right: Option<Box<IsolationTree>>,
    size: usize,
}

impl IsolationTree {
    fn build(values: &[f64], indices: &[usize], depth: usize, max_depth: usize) -> Self {
        let n = indices.len();
        if n <= 1 || depth >= max_depth {
            return IsolationTree {
                split_feature: 0, split_value: 0.0,
                left: None, right: None, size: n,
            };
        }

        let vals: Vec<f64> = indices.iter().map(|&i| values[i]).collect();
        let min_val = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-12 {
            return IsolationTree {
                split_feature: 0, split_value: min_val,
                left: None, right: None, size: n,
            };
        }

        let mut rng = rand::thread_rng();
        let split_value = min_val + rng.gen::<f64>() * (max_val - min_val);

        let left_indices: Vec<usize> = indices.iter().filter(|&&i| values[i] < split_value).cloned().collect();
        let right_indices: Vec<usize> = indices.iter().filter(|&&i| values[i] >= split_value).cloned().collect();

        if left_indices.is_empty() || right_indices.is_empty() {
            return IsolationTree {
                split_feature: 0, split_value,
                left: None, right: None, size: n,
            };
        }

        IsolationTree {
            split_feature: 0,
            split_value,
            left: Some(Box::new(Self::build(values, &left_indices, depth + 1, max_depth))),
            right: Some(Box::new(Self::build(values, &right_indices, depth + 1, max_depth))),
            size: n,
        }
    }

    fn path_length(&self, value: f64, depth: usize) -> f64 {
        if self.left.is_none() && self.right.is_none() {
            return depth as f64 + average_path_length(self.size);
        }
        if value < self.split_value {
            if let Some(ref left) = self.left {
                left.path_length(value, depth + 1)
            } else {
                depth as f64
            }
        } else {
            if let Some(ref right) = self.right {
                right.path_length(value, depth + 1)
            } else {
                depth as f64
            }
        }
    }
}

fn average_path_length(n: usize) -> f64 {
    if n <= 1 { return 0.0; }
    let n = n as f64;
    2.0 * (n.ln() + 0.5772156649) - 2.0 * (n - 1.0) / n
}

impl IsolationForest {
    pub fn new(num_trees: usize, max_depth: usize) -> Self {
        IsolationForest {
            num_trees, max_depth, trees: Vec::new(),
        }
    }

    pub fn fit(&mut self, values: &[f64]) {
        let indices: Vec<usize> = (0..values.len()).collect();
        self.trees = (0..self.num_trees)
            .map(|_| IsolationTree::build(values, &indices, 0, self.max_depth))
            .collect();
    }

    /// Anomaly score for a single value. Higher = more anomalous. Range ~ [0, 1].
    pub fn score(&self, value: f64) -> f64 {
        if self.trees.is_empty() { return 0.0; }
        let n = 256.0; // subsampling size approximation
        let avg_path: f64 = self.trees.iter()
            .map(|t| t.path_length(value, 0))
            .sum::<f64>() / self.trees.len() as f64;
        let c = average_path_length(n as usize);
        2.0_f64.powf(-avg_path / c)
    }

    /// Detect anomalies with score > threshold.
    pub fn detect(&self, values: &[f64], threshold: f64) -> Vec<Anomaly> {
        values.iter().enumerate()
            .filter(|(_, &v)| self.score(v) > threshold)
            .map(|(i, &v)| Anomaly {
                index: i,
                value: v,
                score: self.score(v),
                kind: AnomalyKind::IsolationForest,
            })
            .collect()
    }
}
