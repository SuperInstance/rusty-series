use crate::series::TimeSeries;

/// Change point detected in a time series.
#[derive(Debug, Clone)]
pub struct ChangePoint {
    pub index: usize,
    pub cost_reduction: f64,
}

/// CUSUM change point detection.
/// Detects shifts in mean by tracking cumulative sums.
pub fn cusum_detect(values: &[f64], threshold: f64) -> Vec<ChangePoint> {
    if values.is_empty() { return vec![]; }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let mut cusum_pos = 0.0;
    let mut cusum_neg = 0.0;
    let mut changepoints = Vec::new();

    for (i, &v) in values.iter().enumerate() {
        let diff = v - mean;
        cusum_pos = (cusum_pos + diff).max(0.0);
        cusum_neg = (cusum_neg - diff).max(0.0);

        if cusum_pos > threshold || cusum_neg > threshold {
            changepoints.push(ChangePoint {
                index: i,
                cost_reduction: cusum_pos.max(cusum_neg),
            });
            cusum_pos = 0.0;
            cusum_neg = 0.0;
        }
    }
    changepoints
}

/// Cost function for PELT: sum of squared deviations from segment mean.
fn segment_cost(values: &[f64], start: usize, end: usize) -> f64 {
    if end <= start { return 0.0; }
    let seg = &values[start..end];
    let mean: f64 = seg.iter().sum::<f64>() / seg.len() as f64;
    seg.iter().map(|x| (x - mean).powi(2)).sum()
}

/// PELT (Pruned Exact Linear Time) change point detection.
/// Finds optimal segmentation minimizing total cost + penalty.
pub fn pelt_detect(values: &[f64], penalty: f64) -> Vec<ChangePoint> {
    let n = values.len();
    if n < 2 { return vec![]; }

    // cost[i] = optimal cost for data[0..i]
    let mut cost = vec![f64::INFINITY; n + 1];
    cost[0] = 0.0;

    // last_cp[i] = last changepoint before i
    let mut last_cp = vec![0usize; n + 1];

    // Candidate set
    let mut candidates: Vec<usize> = vec![0];

    for tau in 1..=n {
        let mut best_cost = f64::INFINITY;
        let mut best_t = 0;
        for &t in &candidates {
            let c = cost[t] + segment_cost(values, t, tau) + penalty;
            if c < best_cost {
                best_cost = c;
                best_t = t;
            }
        }
        cost[tau] = best_cost;
        last_cp[tau] = best_t;

        // Prune: remove candidates where cost[t] + min_segment_cost >= cost[tau]
        candidates.retain(|&t| {
            cost[t] + segment_cost(values, t, tau) <= cost[tau]
        });
        candidates.push(tau);
    }

    // Backtrack to find changepoints
    let mut cps = Vec::new();
    let mut tau = n;
    while last_cp[tau] > 0 {
        let t = last_cp[tau];
        cps.push(ChangePoint {
            index: t,
            cost_reduction: segment_cost(values, t, tau),
        });
        tau = t;
    }
    cps.reverse();
    cps
}

/// Binary segmentation: simple top-down change point detector.
pub fn binary_segmentation(values: &[f64], penalty: f64) -> Vec<ChangePoint> {
    let n = values.len();
    if n < 2 { return vec![]; }
    let mut cps = Vec::new();
    binary_seg_recursive(values, 0, n, penalty, &mut cps);
    cps.sort_by_key(|cp| cp.index);
    cps
}

fn binary_seg_recursive(values: &[f64], start: usize, end: usize, penalty: f64, cps: &mut Vec<ChangePoint>) {
    if end - start < 2 { return; }
    let total_cost = segment_cost(values, start, end);
    let mut best_gain = 0.0;
    let mut best_t = start;

    for t in start + 1..end {
        let cost_left = segment_cost(values, start, t);
        let cost_right = segment_cost(values, t, end);
        let gain = total_cost - cost_left - cost_right;
        if gain > best_gain {
            best_gain = gain;
            best_t = t;
        }
    }

    if best_gain > penalty {
        cps.push(ChangePoint { index: best_t, cost_reduction: best_gain });
        binary_seg_recursive(values, start, best_t, penalty, cps);
        binary_seg_recursive(values, best_t, end, penalty, cps);
    }
}
