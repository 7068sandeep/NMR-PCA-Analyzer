# ================================================
# PCA of Multiple 1H NMR Peak Lists (User Input)
# ================================================
# Author: Sandeep Bindra + ChatGPT
# Description:
#   Allows user to input multiple NMR spectra manually.
#   Simulates spectra, performs PCA, and plots results.
# ================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------- Step 1: Get NMR Data from User ----------
print("\nEnter your NMR data for multiple compounds.")
print("Example format:")
print("A: 9.34(1H), 7.90(1H), 7.81(1H), 7.42(1H), 7.31(1H), 7.11(2H), 6.99(1H), 4.81(2H), 2.25(3H)")
print("B: 9.72(1H), 7.87(1H), 7.53(2H), 7.46(1H), 7.38(3H), 6.98(1H), 3.94(2H)")
print("When finished, type 'done' and press Enter.\n")

nmr_data = {}

while True:
    line = input("Enter compound data (or 'done' to finish): ").strip()
    if line.lower() == "done":
        break
    if ":" not in line:
        print("⚠️  Please use the format 'Name: δ(value), δ(value)...'")
        continue

    name, peaks = line.split(":", 1)
    name = name.strip()
    peak_list = []
    for part in peaks.split(","):
        part = part.strip()
        if "(" in part and ")" in part:
            ppm = float(part.split("(")[0])
            hcount = int(part.split("(")[1].replace("H)", ""))
            peak_list.append((ppm, hcount))
    nmr_data[name] = peak_list

if not nmr_data:
    print("No data entered. Exiting.")
    exit()

# ---------- Step 2: Prepare simulation ----------
ppm_scale = np.linspace(0, 10, 1000)
rng = np.random.default_rng(42)

def simulate_spectrum(peaks, n_replicates=5):
    spectra = []
    for _ in range(n_replicates):
        y = np.zeros_like(ppm_scale)
        for ppm, intensity in peaks:
            shift = ppm + rng.normal(scale=0.02)
            width = 0.05 + rng.uniform(0.02, 0.05)
            amp = intensity * rng.uniform(0.9, 1.1)
            y += amp * np.exp(-0.5 * ((ppm_scale - shift) / width) ** 2)
        y += rng.normal(scale=0.005, size=y.shape)
        spectra.append(y)
    return np.array(spectra)

# ---------- Step 3: Simulate spectra ----------
spectra_list, group_labels = [], []
for name, peaks in nmr_data.items():
    simulated = simulate_spectrum(peaks, n_replicates=5)
    spectra_list.append(simulated)
    group_labels.extend([name] * 5)

spectra = np.vstack(spectra_list)
groups = np.array(group_labels)
X = spectra - spectra.mean(axis=0, keepdims=True)

# ---------- Step 4: PCA ----------
U, S, Vt = np.linalg.svd(X, full_matrices=False)
scores = U * S
explained_var_ratio = (S**2 / (len(X) - 1)) / np.sum(S**2 / (len(X) - 1))

# ---------- Step 5: PCA Plot ----------
plt.figure(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(nmr_data)))
for name, color in zip(nmr_data.keys(), colors):
    mask = groups == name
    plt.scatter(scores[mask, 0], scores[mask, 1],
                s=90, edgecolor="k", label=name, alpha=0.8, c=color)
plt.xlabel(f"PC1 ({explained_var_ratio[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained_var_ratio[1]*100:.1f}% variance)")
plt.title("PCA of 1H NMR Spectra (User Input)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Step 6: PCA Loadings ----------
loadings = Vt.T[:, :2]
plt.figure(figsize=(10, 5))
plt.plot(ppm_scale, loadings[:, 0], label="PC1 Loadings", lw=2)
plt.plot(ppm_scale, loadings[:, 1], label="PC2 Loadings", lw=2, linestyle="--")
plt.xlabel("Chemical Shift (ppm)")
plt.ylabel("Loading Weight")
plt.title("PCA Loadings (Spectral Regions Influencing Separation)")
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()