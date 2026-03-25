"""
Autocorrelation-based features for QRNG bitstream analysis.

Computes measures of temporal dependence and periodicity:
- Lag-k autocorrelation
- Autocorrelation decay rate
- Periodicity detection
"""

from __future__ import annotations

import numpy as np


def autocorr_lag1(bits: np.ndarray) -> float:
    """
    Compute the lag-1 autocorrelation of a bit stream.
    
    Measures how much each bit predicts the next bit. For random data,
    this should be close to 0.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    float
        Lag-1 autocorrelation coefficient in [-1, 1].
    """
    if len(bits) < 2:
        return 0.0
    
    # Convert to bipolar (-1, 1) for correlation analysis
    signal = 2 * bits.astype(float) - 1
    
    n = len(signal)
    
    # Compute autocorrelation at lag 1
    mean = np.mean(signal)
    var = np.var(signal)
    
    if var < 1e-10:
        return 0.0
    
    autocorr = np.sum((signal[:-1] - mean) * (signal[1:] - mean)) / ((n - 1) * var)
    
    return float(np.clip(autocorr, -1.0, 1.0))


def autocorr_lagk(bits: np.ndarray, lag: int = 1) -> float:
    """
    Compute the autocorrelation at a specified lag.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    lag : int
        Lag for autocorrelation computation.
        
    Returns
    -------
    float
        Autocorrelation coefficient at the specified lag.
    """
    if len(bits) <= lag:
        return 0.0
    
    # Convert to bipolar (-1, 1) for correlation analysis
    signal = 2 * bits.astype(float) - 1
    
    n = len(signal)
    
    # Compute autocorrelation at specified lag
    mean = np.mean(signal)
    var = np.var(signal)
    
    if var < 1e-10:
        return 0.0
    
    autocorr = np.sum((signal[:-lag] - mean) * (signal[lag:] - mean)) / ((n - lag) * var)
    
    return float(np.clip(autocorr, -1.0, 1.0))


def autocorr_lag5(bits: np.ndarray) -> float:
    """Compute the lag-5 autocorrelation."""
    return autocorr_lagk(bits, lag=5)


def autocorr_lag10(bits: np.ndarray) -> float:
    """Compute the lag-10 autocorrelation."""
    return autocorr_lagk(bits, lag=10)


def autocorr_decay_rate(bits: np.ndarray, max_lag: int = 20) -> float:
    """
    Compute the decay rate of autocorrelation over lags.
    
    For white noise, autocorrelation should be near zero for all lags > 0.
    A slow decay indicates long-range dependence or periodicity.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    max_lag : int
        Maximum lag to consider for decay estimation.
        
    Returns
    -------
    float
        Estimated decay rate (slope of log-autocorrelation vs lag).
    """
    if len(bits) <= max_lag:
        return 0.0
    
    # Compute autocorrelations for multiple lags
    autocorr_values = []
    lags = []
    
    signal = 2 * bits.astype(float) - 1
    n = len(signal)
    mean = np.mean(signal)
    var = np.var(signal)
    
    if var < 1e-10:
        return 0.0
    
    for lag in range(1, min(max_lag + 1, n // 2)):
        autocorr = np.sum((signal[:-lag] - mean) * (signal[lag:] - mean)) / ((n - lag) * var)
        if abs(autocorr) > 0.01:  # Only consider significant values
            autocorr_values.append(abs(autocorr))
            lags.append(lag)
    
    if len(autocorr_values) < 2:
        return 0.0
    
    # Fit a line to log-autocorrelation vs lag
    try:
        coeffs = np.polyfit(lags, np.log(np.maximum(autocorr_values, 1e-10)), 1)
        return float(-coeffs[0])  # Return positive decay rate
    except (RuntimeError, ValueError):
        return 0.0


def partial_autocorr_lag1(bits: np.ndarray) -> float:
    """
    Compute the partial autocorrelation at lag 1.
    
    Partial autocorrelation measures direct correlation between X_t and X_{t-k}
    after removing the effect of intermediate lags.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
        
    Returns
    -------
    float
        Partial autocorrelation at lag 1 (same as regular autocorr for lag 1).
    """
    # For lag 1, partial autocorrelation equals regular autocorrelation
    return autocorr_lag1(bits)


def run_autocorrelation_test(bits: np.ndarray, max_lag: int = 20) -> dict:
    """
    Perform a comprehensive autocorrelation test.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    max_lag : int
        Maximum lag to test.
        
    Returns
    -------
    dict
        Dictionary containing various autocorrelation metrics.
    """
    signal = 2 * bits.astype(float) - 1
    n = len(signal)
    
    # Compute all autocorrelations up to max_lag
    autocorr_values = []
    for lag in range(1, min(max_lag + 1, n // 2)):
        mean = np.mean(signal)
        var = np.var(signal)
        
        if var < 1e-10:
            autocorr_values.append(0.0)
        else:
            ac = np.sum((signal[:-lag] - mean) * (signal[lag:] - mean)) / ((n - lag) * var)
            autocorr_values.append(float(np.clip(ac, -1.0, 1.0)))
    
    # Compute test statistics
    max_acf = max(abs(v) for v in autocorr_values) if autocorr_values else 0.0
    sum_acf = sum(autocorr_values)
    sum_abs_acf = sum(abs(v) for v in autocorr_values)
    
    # Ljung-Box Q statistic (simplified)
    q_stat = n * sum([ac**2 / (n - lag) for lag, ac in enumerate(autocorr_values, 1)])
    
    return {
        "max_autocorrelation": max_acf,
        "sum_autocorrelations": sum_acf,
        "sum_abs_autocorrelations": sum_abs_acf,
        "ljung_box_q": float(q_stat),
        "autocorr_lag1": autocorr_values[0] if len(autocorr_values) > 0 else 0.0,
        "autocorr_lag2": autocorr_values[1] if len(autocorr_values) > 1 else 0.0,
        "autocorr_lag3": autocorr_values[2] if len(autocorr_values) > 2 else 0.0,
    }


def spectral_density_estimate(bits: np.ndarray, num_bins: int = 64) -> float:
    """
    Estimate the spectral density at low frequencies.
    
    Low-frequency power indicates long-range correlations or trends.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    num_bins : int
        Number of frequency bins for estimation.
        
    Returns
    -------
    float
        Ratio of low-frequency to total power.
    """
    # Convert to bipolar
    signal = 2 * bits.astype(float) - 1
    
    # Compute FFT
    fft_result = np.fft.rfft(signal)
    power = np.abs(fft_result) ** 2
    
    if len(power) < num_bins:
        return 0.5
    
    # Low-frequency power (first quarter of spectrum)
    low_freq_power = np.sum(power[:num_bins // 4])
    total_power = np.sum(power)
    
    if total_power < 1e-10:
        return 0.5
    
    return float(low_freq_power / total_power)


def compute_feature_vector(bits: np.ndarray, max_lag: int = 20) -> dict:
    """
    Compute all autocorrelation-based features for a bit window.
    
    Parameters
    ----------
    bits : np.ndarray
        Array of binary values (0 or 1).
    max_lag : int
        Maximum lag for autocorrelation tests.
        
    Returns
    -------
    dict
        Dictionary of feature names to computed values.
    """
    acf_test = run_autocorrelation_test(bits, max_lag=max_lag)
    
    return {
        "autocorr_lag1": acf_test["autocorr_lag1"],
        "autocorr_lag2": acf_test["autocorr_lag2"],
        "autocorr_lag3": acf_test["autocorr_lag3"],
        "max_autocorrelation": acf_test["max_autocorrelation"],
        "sum_abs_autocorrelations": acf_test["sum_abs_autocorrelations"],
        "spectral_low_freq_ratio": spectral_density_estimate(bits),
    }