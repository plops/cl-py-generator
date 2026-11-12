# Automatisch generiert mit cl-py-generator
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.ndimage as ndi

# 1. Verrauschtes Gauss-Signal + exakte Ableitung
rng = np.random.default_rng(42)
x = np.linspace(-5.0, 5.0, 501)
dx = (x[1]) - (x[0])
sigma_sig = 1.0
noise_std = 0.05
y_clean = np.exp(-0.5 * np.square(x / sigma_sig))
dy_true = ((-1.0 * x) / np.square(sigma_sig)) * y_clean
y_noisy = y_clean + rng.normal(0.0, noise_std, len(x))
# 2x2 Subplots + Unpacking via ,@(loop ...)
fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        13,
        8.5,
    ),
)
ax00 = axes[0][0]
ax01 = axes[0][1]
ax10 = axes[1][0]
ax11 = axes[1][1]
# 2. SavGol-Sweep (Bias-Varianz-Tradeoff)
windows = np.arange(5, 103, 2)
best_rmse = np.inf
best_win = 5
best_poly = 1
for p in [1, 3, 5]:
    errs = []
    for w in windows:
        dy_tmp = sig.savgol_filter(
            y_noisy, window_length=int(w), polyorder=p, deriv=1, delta=dx
        )
        rmse = np.sqrt(np.mean(np.square(dy_tmp - dy_true)))
        errs.append(rmse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_win = int(w)
            best_poly = p
    ax10.plot(windows, errs, label="SavGol poly={}".format(p))
dy_savgol = sig.savgol_filter(
    y_noisy, window_length=best_win, polyorder=best_poly, deriv=1, delta=dx
)
# 3. Alternativen: finite Differenz & optimierter Gauss-Ableitungsfilter
dy_grad = np.gradient(y_noisy, dx)
rmse_grad = np.sqrt(np.mean(np.square(dy_grad - dy_true)))
best_rmse_g = np.inf
best_sigma_g = 0.5
for sg in np.linspace(0.05, 1.5, 60):
    dy_g = ndi.gaussian_filter1d(y_noisy, sigma=sg / dx, order=1) / dx
    rmse_g = np.sqrt(np.mean(np.square(dy_g - dy_true)))
    if rmse_g < best_rmse_g:
        best_rmse_g = rmse_g
        best_sigma_g = sg
dy_gauss = ndi.gaussian_filter1d(y_noisy, sigma=best_sigma_g / dx, order=1) / dx
# 4. Filter-Eigenschaften: Impulsantworten, Noise-Gain, Momente & FFT
imp = np.zeros(201)
imp[100] = 1.0
h_sg = sig.savgol_coeffs(best_win, best_poly, deriv=1, delta=dx)
h_gauss = ndi.gaussian_filter1d(imp, sigma=best_sigma_g / dx, order=1) / dx
h_grad = np.array([(-0.5) / dx, 0.0, 0.5 / dx])
half_w = int((best_win - 1) / 2)
k_grid = np.arange(-1 * half_w, half_w + 1)
dc_sg = np.sum(h_sg)
slope_sg = np.sum(-1.0 * k_grid * dx * h_sg)
n_fft = 4096
freqs = np.fft.rfftfreq(n_fft, d=dx)
amp_ideal = 2.0 * np.pi * freqs
NG_sg = np.sqrt(np.sum(np.square(H_sg)))
AMP_sg = np.abs(np.fft.rfft(H_sg, n=n_fft))
NG_gauss = np.sqrt(np.sum(np.square(H_gauss)))
AMP_gauss = np.abs(np.fft.rfft(H_gauss, n=n_fft))
NG_grad = np.sqrt(np.sum(np.square(H_grad)))
AMP_grad = np.abs(np.fft.rfft(H_grad, n=n_fft))
mask_pos = freqs > 0.0
rel_sg = (amp_sg[mask_pos]) / (amp_ideal[mask_pos])
f_3db_sg = freqs[mask_pos][np.argmax(rel_sg < (1.0 / np.sqrt(2.0)))]
print(
    "RMSE      -> SavGol(w={},p={}): {:.5f} | Gauss(s={:.3f}): {:.5f} | Grad: {:.5f}".format(
        best_win, best_poly, best_rmse, best_sigma_g, best_rmse_g, rmse_grad
    )
)
print(
    "NoiseGain -> SavGol: {:.3f} | Gauss: {:.3f} | Grad: {:.3f}".format(
        ng_sg, ng_gauss, ng_grad
    )
)
print(
    "SavGol    -> DC-Summe: {:.2e} | Rampen-Steigung: {:.6f} | f_-3dB: {:.2f} Hz".format(
        dc_sg, slope_sg, f_3db_sg
    )
)
# 5. Plots
ax00.plot(x, y_noisy, label="Verrauscht", alpha=0.5)
ax00.plot(x, y_clean, label="Sauberer Gauss", color="k", linewidth=2)
ax01.plot(x, dy_grad, label="np.gradient", alpha=0.35, color="gray")
ax01.plot(x, dy_true, label="Exakt g'(x)", color="k", linewidth=2)
ax01.plot(
    x, dy_savgol, label="SavGol (w={}, p={})".format(best_win, best_poly), linewidth=2
)
ax01.plot(
    x,
    dy_gauss,
    label="Gauss-Deriv (s={:.2f})".format(best_sigma_g),
    linestyle="--",
    linewidth=2,
)
ax10.axvline(best_win, color="r", linestyle=":", label="Optimum w={}".format(best_win))
f_mask = freqs <= 6.0
ax11.plot(
    freqs[f_mask], amp_ideal[f_mask], label="Ideal: 2*pi*f", color="k", linestyle=":"
)
ax11.plot(freqs[f_mask], amp_grad[f_mask], label="Zentrale Diff.", linestyle="-")
ax11.plot(freqs[f_mask], amp_sg[f_mask], label="SavGol Opt.", linestyle="-")
ax11.plot(freqs[f_mask], amp_gauss[f_mask], label="Gauss-Deriv Opt.", linestyle="--")
ax11.axvline(
    f_3db_sg, color="r", linestyle=":", label="SavGol -3dB ({:.2f} Hz)".format(f_3db_sg)
)
ax00.set_title("Eingangssignal")
ax00.set_xlabel("x")
ax00.set_ylabel("y")
ax00.legend()
ax00.grid(alpha=0.3)
ax01.set_title("Vergleich der 1. Ableitung")
ax01.set_xlabel("x")
ax01.set_ylabel("dy/dx")
ax01.legend()
ax01.grid(alpha=0.3)
ax10.set_title("SavGol Optimierung (Bias-Varianz)")
ax10.set_xlabel("window_length")
ax10.set_ylabel("RMSE vs. exakt")
ax10.legend()
ax10.grid(alpha=0.3)
ax11.set_title("Frequenzgang |H(f)|")
ax11.set_xlabel("Frequenz [1/x]")
ax11.set_ylabel("|H(f)|")
ax11.legend()
ax11.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("savgol_analyse.png", dpi=150)
plt.show()
