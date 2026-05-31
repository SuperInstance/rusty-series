use crate::series::TimeSeries;

/// Spectral analysis result.
#[derive(Debug, Clone)]
pub struct SpectralResult {
    pub frequencies: Vec<f64>,
    pub periodogram: Vec<f64>,
}

/// Compute periodogram using the Discrete Fourier Transform.
/// Returns frequencies and power spectral density.
pub fn periodogram(values: &[f64]) -> SpectralResult {
    let n = values.len();
    if n == 0 {
        return SpectralResult { frequencies: vec![], periodogram: vec![] };
    }

    let mut freqs = Vec::new();
    let mut powers = Vec::new();

    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = values.iter().map(|x| x - mean).collect();

    let half = n / 2 + 1;
    for k in 0..half {
        let freq = k as f64 / n as f64;
        let mut re = 0.0;
        let mut im = 0.0;
        for t in 0..n {
            let angle = 2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
            re += centered[t] * angle.cos();
            im -= centered[t] * angle.sin();
        }
        let power = (re * re + im * im) / n as f64;
        freqs.push(freq);
        powers.push(power);
    }

    SpectralResult { frequencies: freqs, periodogram: powers }
}

/// Find the dominant frequency (peak in periodogram, excluding DC).
pub fn dominant_frequency(values: &[f64]) -> f64 {
    let result = periodogram(values);
    if result.periodogram.len() < 2 { return 0.0; }

    // Skip DC component (index 0)
    let mut best_idx = 1;
    let mut best_power = result.periodogram[1];
    for (i, &p) in result.periodogram[1..].iter().enumerate() {
        if p > best_power {
            best_power = p;
            best_idx = i + 1;
        }
    }
    result.frequencies[best_idx]
}

/// Find top-k dominant frequencies.
pub fn top_frequencies(values: &[f64], k: usize) -> Vec<(f64, f64)> {
    let result = periodogram(values);
    let mut indexed: Vec<(usize, f64)> = result.periodogram[1..]
        .iter().enumerate().map(|(i, &p)| (i + 1, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.into_iter().take(k)
        .map(|(i, p)| (result.frequencies[i], p))
        .collect()
}

/// Power in a frequency band [f_low, f_high].
pub fn band_power(values: &[f64], f_low: f64, f_high: f64) -> f64 {
    let result = periodogram(values);
    result.frequencies.iter().zip(&result.periodogram)
        .filter(|(f, _)| **f >= f_low && **f <= f_high)
        .map(|(_, p)| *p)
        .sum()
}
