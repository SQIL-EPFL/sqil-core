import numpy as np


def lorentzian(x, A, x0, fwhm, y0):
    r"""
    L(x) = A * (|FWHM| / 2) / ((x - x0)^2 + (FWHM^2 / 4)) + y0

    $$L(x) = A \frac{\left| \text{FWHM} \right|}{2} \frac{1}{(x - x_0)^2 + \frac{\text{FWHM}^2}{4}} + y_0$$
    """  # noqa: E501
    return A * (np.abs(fwhm) / 2.0) / ((x - x0) ** 2.0 + fwhm**2.0 / 4.0) + y0


def two_lorentzians_shared_x0(x_data_1, x_data_2, A1, fwhm1, y01, A2, fwhm2, y02, x0):
    r"""
    Concatenates two lorentzians with same x0.
    L_1(x) = A_1 * (|FWHM_1| / 2) / ((x - x0)^2 + (FWHM_1^2 / 4)) + y0_1
    L_2(x) = A_2 * (|FWHM_2| / 2) / ((x - x0)^2 + (FWHM_2^2 / 4)) + y0_2
    """  # noqa: E501
    y1 = lorentzian(x_data_1, A1, x0, fwhm1, y01)
    y2 = lorentzian(x_data_2, A2, x0, fwhm2, y02)
    return np.concatenate([y1, y2])


def gaussian(x, A, x0, sigma, y0):
    r"""
    G(x) = A / (|σ| * sqrt(2π)) * exp(- (x - x0)^2 / (2σ^2)) + y0

    $$G(x) = A \frac{1}{\left| \sigma \right| \sqrt{2\pi}} \exp\left( -\frac{(x - x_0)^2}{2\sigma^2} \right) + y_0$$
    """  # noqa: E501
    return (
        A
        * (1 / (np.abs(sigma) * np.sqrt(2.0 * np.pi)))
        * np.exp(-((x - x0) ** 2.0) / (2.0 * sigma**2.0))
        + y0
    )


def two_gaussians_shared_x0(x_data_1, x_data_2, A1, fwhm1, y01, A2, fwhm2, y02, x0):
    r"""
    Concatenates two gaussians with same x0.
    G_1(x) = A_1 / (|σ_1| * sqrt(2π)) * exp(- (x - x0)^2 / (2σ_1^2)) + y0_1
    G_1(x) = A_2 / (|σ_2| * sqrt(2π)) * exp(- (x - x0)^2 / (2σ_2^2)) + y0_2
    """  # noqa: E501
    y1 = gaussian(x_data_1, A1, x0, fwhm1, y01)
    y2 = gaussian(x_data_2, A2, x0, fwhm2, y02)
    return np.concatenate([y1, y2])


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
    """  # noqa: E501
    return (A * np.exp(np.abs(nQP) * (np.exp(-x / T1QP) - 1)) * np.exp(-x / T1R)) + y0


def decaying_oscillations(x, A, tau, y0, phi, T):
    r"""
    f(x) = A * exp(-x / τ) * cos(2π * (x - φ) / T) + y0

    $$f(x) = A \exp\left( -\frac{x}{\tau} \right) \cos\left( 2\pi \frac{x - \phi}{T} \right) + y_0$$
    """  # noqa: E501
    return A * np.exp(-x / tau) * np.cos(2.0 * np.pi * (x - phi) / T) + y0


def many_decaying_oscillations(t, *params):
    r"""
    f(x) = SUM_i A_i * exp(-x / τ_i) * cos(2π * (x - φ_i) / T_i) + y0

    $$f(x) = \sum_i A_i \cdot e^{-x/\tau_i} \cdot \cos\left(\frac{2\pi (x - \phi_i)}{T_i}\right) + y_0$$
    """  # noqa: E501
    n = (len(params) - 1) // 4  # Each oscillation has 4 params: A, tau, phi, T
    offset = params[-1]
    result = np.zeros_like(t)
    for i in range(n):
        A = params[4 * i]
        tau = params[4 * i + 1]
        phi = params[4 * i + 2]
        T = params[4 * i + 3]
        result += A * np.exp(-t / tau) * np.cos(2 * np.pi * T * t + phi)
    return result + offset


def oscillations(x, A, y0, phi, T):
    r"""
    f(x) = A * cos(2π * (x - φ) / T) + y0

    $$f(x) = A \cos\left( 2\pi \frac{x - \phi}{T} \right) + y_0$$
    """
    return A * np.cos(2.0 * np.pi * (x - phi) / T) + y0


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
        Total (or loaded) quality factor controlling the sharpness and width of the
        resonance peak.

    Returns
    -------
    np.ndarray
        The computed skewed Lorentzian values corresponding to each input `f`.
    """  # noqa: E501
    return (
        A1
        + A2 * (f - fr)
        + (A3 + A4 * (f - fr)) / (1 + (2 * Q_tot * (f / fr - 1)) ** 2)
    )
