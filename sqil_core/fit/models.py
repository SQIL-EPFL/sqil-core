import numpy as np


def skewed_lorentzian(
    f: np.ndarray, A1: float, A2: float, A3: float, A4: float, fr: float, Q_tot: float
) -> np.ndarray:
    """
    Computes the skewed Lorentzian function.

    This function models asymmetric resonance peaks using a skewed Lorentzian
    function, which is commonly used in spectroscopy and resonator analysis to account
    for both peak sharpness and asymmetry.

    The function is defined as:
        L(f) = A1 + A2 * (f - fr) + (A3 + A4 * (f - fr)) / [1 + (2 * Q_tot * ((f / fr) - 1))²]

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

    Examples
    --------
    >>> import numpy as np
    >>> f = np.linspace(0.9, 1.1, 500)
    >>> y = skewed_lorentzian_fun(f, A1=0.0, A2=0.0, A3=1.0, A4=0.0, fr=1.0, Q_tot=1000)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(f, y)
    >>> plt.show()
    """
    return (
        A1
        + A2 * (f - fr)
        + (A3 + A4 * (f - fr)) / (1 + (2 * Q_tot * (f / fr - 1)) ** 2)
    )
