import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
from scipy import stats
from scipy.linalg import norm
from scipy.optimize import curve_fit, minimize, newton
from tabulate import tabulate

import sqil_core as sqil


def guess_from_lorentzian(freq, z_data):
    y = np.abs(z_data) ** 2
    params, cov = fit_skewed_lorentzian(freq, y)
    (A1, A2, A3, A4, fr, Q_tot) = params
    return fr, Q_tot, [A1, A2, A3, A4]


# Read the data
[dBmag], [phase], [freq] = sqil.extract_h5_data(bad, ["mag", "phase", "frequency"])
linmag = 10 ** (dBmag / 20)
phase = shift_phase(np.unwrap(phase))
data = linmag * np.exp(1j * phase)

data_history = []
data_history.append(data)

fr = None
Q_tot = None

# Guess fr and Q_tot from the lorentzian fit of the magnitude
if not skip_lorentzian_guess:
    fr, Q_tot = guess_from_lorentzian(freq, data)
    # If the estimate Q_tot is < 0 ignore it
    if Q_tot < 0:
        Q_tot = None

# Estimate a rough cable delay from the first points of the phase data
[c, tau] = estimate_linear_background(freq, phase, 0.1)
tau /= 2 * np.pi
# Remove cable delay from phase
phase1 = shift_phase(phase - 2 * np.pi * tau * (freq - freq[0]))
data1 = linmag * np.exp(1j * phase1)
data_history.append(data1)

# Move the circle to the center
xc, yc, r0 = fit_circle_algebraic(data1)
data3 = data1 - xc - 1j * yc
phase3 = np.unwrap(np.angle(data3))
data_history.append(data3)

# Perform the phase vs frequency fit
theta0, Q_tot, fr = fit_phase_vs_freq_taketo(freq, phase3, Q_tot=Q_tot, f_r=fr)

# Find the off-resonant point => find a and alpha
p_offres = (xc + 1j * yc) + r0 * np.exp(1j * (theta0 + np.pi))
a = np.abs(p_offres)
alpha = np.angle(p_offres)
# Adjust the data
linmag5 = linmag / a
phase5 = phase1 - alpha
data5 = linmag5 * np.exp(1j * phase5)
data_history.append(data5)

# Find the impedence mismatch
xc, yc, r0 = fit_circle_algebraic(data5)
phi0 = -np.arcsin(yc / r0)
# Calculate Q_ext and Q_int
Q_ext = Q_tot / (r06 * np.exp(-1j * phi0))  # REFLECTION ONLY
Q_int = 1 / (1 / Q_tot - 1 / np.real(Q_ext))

# Calculate the final fit
fit = S11_reflection(freq, a, alpha, tau, Q_tot, Q_ext, fr, phi0)

# Get a global phase factor
theta0 = phase[0] - np.unwrap(np.angle(fit))[0]
