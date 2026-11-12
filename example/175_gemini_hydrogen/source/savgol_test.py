# Automatisch generiert mit cl-py-generator
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.ndimage as ndi

# 1. Signal + Monte-Carlo-Ensemble (n_mc x n_x)
rng = np.random.default_rng(42)
x = np.linspace(-5.0, 5.0, 501)
dx = (x[1]) - (x[0])
sigma_sig = 1.0
noise_std = 0.05
n_mc = 40
y_clean = np.exp(-0.5 * np.square(x / sigma_sig))
dy_true = ((-1.0 * x) / np.square(sigma_sig)) * y_clean
y_noisy_mc = y_clean + rng.normal(
    0.0,
    noise_std,
    (
        n_mc,
        len(x),
    ),
)
fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        13.5,
        9.0,
    ),
)
ax00 = axes[0][0]
ax01 = axes[0][1]
ax10 = axes[1][0]
ax11 = axes[1][1]
# 2. SavGol Monte-Carlo-Sweep (w = 5..251)
windows = np.arange(5, 253, 2)
best_rmse = np.inf
best_win = 5
best_poly = 1
for p in [1, 3, 5]:
    means = []
    stds = []
    valid_wins = windows[windows > p]
    for w in valid_wins:
        dy_tmp = sig.savgol_filter(
            y_noisy_mc, window_length=int(w), polyorder=p, deriv=1, delta=dx, axis=-1
        )
        rmses = np.sqrt(np.mean(np.square(dy_tmp - dy_true), axis=-1))
        m = np.mean(rmses)
        s = np.std(rmses)
        means.append(m)
        stds.append(s)
        if m < best_rmse:
            best_rmse = m
            best_win = int(w)
            best_poly = p
    m_arr = np.array(means)
    s_arr = np.array(stds)
    ax10.plot(valid_wins, m_arr, label="SavGol poly={}".format(p))
    ax10.fill_between(
        valid_wins, np.maximum(1.0e-6, m_arr - s_arr), m_arr + s_arr, alpha=0.18
    )
dy_savgol = sig.savgol_filter(
    y_noisy_mc, window_length=best_win, polyorder=best_poly, deriv=1, delta=dx, axis=-1
)
# 3. Alternativen (Gradient & Gauss-Ableitungsfilter)
dy_grad = np.gradient(y_noisy_mc, dx, axis=-1)
rmse_grad = np.mean(np.sqrt(np.mean(np.square(dy_grad - dy_true), axis=-1)))
best_rmse_g = np.inf
best_sigma_g = 0.5
for sg in np.linspace(0.05, 1.5, 60):
    dy_g = ndi.gaussian_filter1d(y_noisy_mc, sigma=sg / dx, order=1, axis=-1) / dx
    m_g = np.mean(np.sqrt(np.mean(np.square(dy_g - dy_true), axis=-1)))
    if m_g < best_rmse_g:
        best_rmse_g = m_g
        best_sigma_g = sg
dy_gauss = (
    ndi.gaussian_filter1d(y_noisy_mc, sigma=best_sigma_g / dx, order=1, axis=-1) / dx
)
# 4. Filter-Eigenschaften, Bias-Varianz & 2Hz-Stoertest
imp = np.zeros(1001)
imp[500] = 1.0
h_sg = sig.savgol_coeffs(best_win, best_poly, deriv=1, delta=dx)
h_gauss = ndi.gaussian_filter1d(imp, sigma=best_sigma_g / dx, order=1) / dx
h_grad = np.array([(-0.5) / dx, 0.0, 0.5 / dx])
n_fft = 4096
freqs = np.fft.rfftfreq(n_fft, d=dx)
amp_ideal = 2.0 * np.pi * freqs
ng_sg = np.sqrt(np.sum(np.square(h_sg)))
amp_sg = np.abs(np.fft.rfft(h_sg, n=n_fft))
ng_gauss = np.sqrt(np.sum(np.square(h_gauss)))
amp_gauss = np.abs(np.fft.rfft(h_gauss, n=n_fft))
ng_grad = np.sqrt(np.sum(np.square(h_grad)))
amp_grad = np.abs(np.fft.rfft(h_grad, n=n_fft))
noise_sg = ng_sg * noise_std
bias_sg = np.sqrt(np.maximum(0.0, np.square(best_rmse) - np.square(noise_sg)))
noise_gauss = ng_gauss * noise_std
bias_gauss = np.sqrt(np.maximum(0.0, np.square(best_rmse_g) - np.square(noise_gauss)))
mask_pos = freqs > 0.0
rel_sg = (amp_sg[mask_pos]) / (amp_ideal[mask_pos])
f_3db_sg = freqs[mask_pos][np.argmax(rel_sg < (1.0 / np.sqrt(2.0)))]
y_tone = y_clean + 0.05 * np.sin(2.0 * np.pi * 2.0 * x)
dy_tone_sg = sig.savgol_filter(
    y_tone, window_length=best_win, polyorder=best_poly, deriv=1, delta=dx
)
dy_tone_g = ndi.gaussian_filter1d(y_tone, sigma=best_sigma_g / dx, order=1) / dx
rmse_tone_sg = np.sqrt(np.mean(np.square(dy_tone_sg - dy_true), axis=-1))
rmse_tone_g = np.sqrt(np.mean(np.square(dy_tone_g - dy_true), axis=-1))
print(
    "MC-RMSE (n={}) -> SavGol(w={},p={}): {:.5f} | Gauss(s={:.3f}): {:.5f} | Grad: {:.5f}".format(
        n_mc, best_win, best_poly, best_rmse, best_sigma_g, best_rmse_g, rmse_grad
    )
)
print(
    "Bias / Noise     -> SavGol: {:.5f} / {:.5f} | Gauss: {:.5f} / {:.5f} | f_-3dB: {:.2f} Hz".format(
        bias_sg, noise_sg, bias_gauss, noise_gauss, f_3db_sg
    )
)
print(
    "2.0Hz-Stoertest  -> SavGol RMSE: {:.5f} | Gauss RMSE: {:.5f}".format(
        rmse_tone_sg, rmse_tone_g
    )
)
# 5. Plots
ax00.plot(x, y_noisy_mc[0], label="Verrauscht (Seed 0)", alpha=0.5)
ax00.plot(x, y_clean, label="Sauberer Gauss", color="k", linewidth=2)
ax01.plot(x, dy_grad[0], label="np.gradient", alpha=0.3, color="gray")
ax01.plot(x, dy_true, label="Exakt g'(x)", color="k", linewidth=2)
ax01.plot(
    x,
    dy_savgol[0],
    label="SavGol (w={}, p={})".format(best_win, best_poly),
    linewidth=2,
)
ax01.plot(
    x,
    dy_gauss[0],
    label="Gauss-Deriv (s={:.2f})".format(best_sigma_g),
    linestyle="--",
    linewidth=2,
)
ax10.axvline(
    best_win, color="r", linestyle=":", label="SavGol Opt. w={}".format(best_win)
)
ax10.axhline(
    best_rmse_g,
    color="k",
    linestyle="--",
    label="Gauss Opt. ({:.4f})".format(best_rmse_g),
)
f_mask = freqs <= 3.0
d_spec = freqs * np.exp(-2.0 * np.square(np.pi * sigma_sig * freqs))
d_spec = 2.5 * (d_spec / np.max(d_spec))
ax11.fill_between(
    freqs[f_mask],
    0.0,
    d_spec[f_mask],
    color="0.85",
    alpha=0.6,
    label="Spektrum |g'(f)| (norm.)",
)
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
ax00.legend(fontsize=8.5)
ax00.grid(alpha=0.3, which="both")
ax01.set_title("Vergleich der 1. Ableitung (Realisierung 0)")
ax01.set_xlabel("x")
ax01.set_ylabel("dy/dx")
ax01.legend(fontsize=8.5)
ax01.grid(alpha=0.3, which="both")
ax10.set_title("SavGol MC-Optimierung (Mean +/- 1 std)")
ax10.set_xlabel("window_length")
ax10.set_ylabel("RMSE vs. exakt")
ax10.set_yscale("log")
ax10.legend(fontsize=8.5)
ax10.grid(alpha=0.3, which="both")
ax11.set_title("Frequenzgang |H(f)| vs. Signalspektrum")
ax11.set_xlabel("Frequenz [1/x]")
ax11.set_ylabel("|H(f)|")
ax11.set_xlim(0.0, 3.0)
ax11.set_ylim(0.0, 5.5)
ax11.legend(fontsize=8.5)
ax11.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("savgol_analyse.png", dpi=150)
plt.show()
