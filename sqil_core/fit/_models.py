import numpy as np


def lorentzian(x, A, x0, fwhm, y0):
    r"""
    L(x) = A * (|FWHM| / 2) / ((x - x0)^2 + (FWHM^2 / 4)) + y0

    $$L(x) = A \frac{\left| \text{FWHM} \right|}{2} \frac{1}{(x - x_0)^2 + \frac{\text{FWHM}^2}{4}} + y_0$$
    """
    return A * (np.abs(fwhm) / 2.0) / ((x - x0) ** 2.0 + fwhm**2.0 / 4.0) + y0


def gaussian(x, A, x0, sigma, y0):
    r"""
    G(x) = A / (|σ| * sqrt(2π)) * exp(- (x - x0)^2 / (2σ^2)) + y0

    $$G(x) = A \frac{1}{\left| \sigma \right| \sqrt{2\pi}} \exp\left( -\frac{(x - x_0)^2}{2\sigma^2} \right) + y_0$$
    """
    return (
        A
        * (1 / (np.abs(sigma) * np.sqrt(2.0 * np.pi)))
        * np.exp(-((x - x0) ** 2.0) / (2.0 * sigma**2.0))
        + y0
    )


def decaying_exp(x, A, tau, y0):
    r"""
    f(x) = A * exp(-x / τ) + y0

    $$f(x) = A \exp\left( -\frac{x}{\tau} \right) + y_0$$
    """
    return A * np.exp(-x / tau) + y0


def qubit_relaxation_qp(x, A, T1R, y0, T1QP, nQP):
    r"""
    f(x) = A * exp(|nQP| * (exp(-x / T1QP) - 1)) * exp(-x / T1R) + y0

    $$f(x) = A \exp\left( |\text{n}_{\text{QP}}| \left( \exp\left(-\frac{x}{T_{1QP}}\right)
    - 1 \right) \right) \exp\left(-\frac{x}{T_{1R}}\right) + y_0$$
    """
    return (A * np.exp(np.abs(nQP) * (np.exp(-x / T1QP) - 1)) * np.exp(-x / T1R)) + y0


def decaying_oscillations(x, A, tau, y0, phi, T):
    r"""
    f(x) = A * exp(-x / τ) * cos(2π * (x - φ) / T) + y0

    $$f(x) = A \exp\left( -\frac{x}{\tau} \right) \cos\left( 2\pi \frac{x - \phi}{T} \right) + y_0$$
    """
    return A * np.exp(-x / tau) * np.cos(2.0 * np.pi * (x - phi) / T) + y0


def skewed_lorentzian(
    f: np.ndarray, A1: float, A2: float, A3: float, A4: float, fr: float, Q_tot: float
) -> np.ndarray:
    r"""
    Computes the skewed Lorentzian function.

    This function models asymmetric resonance peaks using a skewed Lorentzian
    function, which is commonly used in spectroscopy and resonator analysis to account
    for both peak sharpness and asymmetry.

    L(f) = A1 + A2 * (f - fr) + (A3 + A4 * (f - fr)) / [1 + (2 * Q_tot * ((f / fr) - 1))²]

    $$L(f) = A_1 + A_2 \cdot (f - f_r)+ \frac{A_3 + A_4 \cdot (f - f_r)}{1
    + 4 Q_{\text{tot}}^2 \left( \frac{f - f_r}{f_r} \right)^2}$$

    Parameters
    ----------
    f : np.ndarray
        Array of frequency or independent variable values.
    A1 : float
        Baseline offset of the curve.
    A2 : float
        Linear slope adjustment, accounting for background trends.
    A3 : float
        Amplitude of the Lorentzian peak.
    A4 : float
        Skewness factor that adjusts the asymmetry of the peak.
    fr : float
        Resonance frequency or the peak position.
    Q_tot : float
        Total (or loaded) quality factor controlling the sharpness and width of the resonance peak.

    Returns
    -------
    np.ndarray
        The computed skewed Lorentzian values corresponding to each input `f`.
    """
    return (
        A1
        + A2 * (f - fr)
        + (A3 + A4 * (f - fr)) / (1 + (2 * Q_tot * (f / fr - 1)) ** 2)
    )
