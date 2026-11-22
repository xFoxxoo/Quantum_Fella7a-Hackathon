# 🌾 Quantum Felaha — Moroccan Crop Classification Challenge
**Qiskit Fall Fest 2026 · EMSI Casablanca · Quantum Africa**

> **Team:** Quantum Felaha *(Les Agriculteurs Quantiques)*  
> **Track:** Classical vs. Quantum Comparative Study & Quantum Kernel Modeling  
> **Primary Validation Result:** **0.8707 Macro-F1** (**87.33% Accuracy**) in **< 1.0s** using a 4-Qubit Calibrated Quantum Kernel SVM — outperforming the starter QNN baseline (`0.8557`) by **+1.50 points** and cutting runtime by **~65x**.

---

## 📌 1. Project Overview & Motivation

Precision agriculture in Morocco relies heavily on accurate crop mapping to optimize irrigation and resource allocation. In this challenge, we classify **1,285 satellite-observed agricultural parcels** in the **Doukkala** region into three strategic crops:
* **Alfalfa** (*Luzerne*)
* **Maize** (*Maïs*)
* **Wheat** (*Blé*)

Rather than treating quantum machine learning as a black box, our team conducted a **controlled, hypothesis-driven comparative study** (*Comparaison Classique vs. Quantique*) to answer a core research question:

> *"Under strict qubit ($\le 8$) and training-budget constraints, why do variational QNNs stall under gradient-free optimization (COBYLA), and can a calibrated, low-depth Quantum Kernel SVM overcome convergence bottlenecks to outperform the quantum baseline?"*

---

## 🔬 2. Experimental Journey & Key Findings

We systematically evaluated **6 classical and quantum configurations** on the exact same leakage-free 4-component PCA view (fitted strictly on the 914 training parcels and evaluated on the balanced 150-parcel validation set):

| Model | Paradigm | Quantum / Model Resources | Validation Macro-F1 | Validation Accuracy | Mean Runtime |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Classical RBF SVM** *(Reference)* | Classical | 4 PCA features | **0.8797** | **88.00%** | `< 0.01 s` |
| **🏆 Calibrated Quantum Kernel SVM** *(Ours)* | **Quantum-Classical** | **4 qubits · Depth 3 · 4 `CZ` gates** | **0.8707** | **87.33%** | **`0.85 s`** |
| **Starter QNN** (`ry` encoding) | Variational QNN | 4 qubits · 8 trainable params · 3 `CX` | 0.8557 | 85.56% | `~55.0 s / seed` |
| **Quantum Kernel SVM** (`zz_feature_map`) | Quantum-Classical | 4 qubits · `reps=1` · 3 `CX` | 0.8065 | 80.67% | `0.85 s` |
| **Quantum Kernel SVM** (`zz_feature_map`) | Quantum-Classical | 6 qubits · `reps=1` · 5 `CX` | 0.7419 | 74.00% | `1.17 s` |
| **Data Re-uploading QNN** (`reupload`) | Variational QNN | 4 qubits · 12 trainable params · 3 `CX` | 0.7081 ± 0.0411 | 71.11% | `166.0 s / seed` |
| **Matched Classical Control** (`MatchedHead`) | Classical Neural | 11 trainable params (1 hidden unit) | 0.5814 | 65.11% | `0.05 s / seed` |

### 💡 Three Scientific Takeaways
1. **Why Variational QNNs Stalled (`0.8557` $\rightarrow$ `0.7081`):**  
   To satisfy the $\pm 20\%$ parameter-matching rule against the 11-parameter classical control (`MatchedHead`), we built a 12-parameter 2-layer Data Re-uploading QNN (`reupload`, `8.3%` parameter gap). While it beat the matched classical neural network (`0.7081` vs. `0.5814`), expanding the parameter space under a fixed **100-evaluation COBYLA budget** prevented convergence and increased seed variance ($\pm 0.0411$).
2. **Mitigating Hilbert-Space Kernel Concentration (`4 qubits` vs. `6 qubits`):**  
   Testing `zz_feature_map` across dimensions showed a sharp drop from 4 qubits (`0.8065`) to 6 qubits (`0.7419`), confirming that expanding to a $2^6 = 64$-dimensional Hilbert space dilutes inner-product similarities on 914 samples.
3. **Breakthrough via Smooth Bi-Axial Encoding & Angle Scaling (`scale = 0.35`):**  
   Replacing COBYLA with exact statevector fidelity $K(x, y) = |\langle \psi(x) \mid \psi(y) \rangle|^2$ and designing a smooth **4-qubit circuit (`RY` + `RZ` + circular `CZ`)** with calibrated angle scaling ($\theta_i = 0.35\pi \tanh(x_i)$) prevented periodic phase wrap-around and lifted quantum performance to **0.8707 Macro-F1**.

---

## ⚛️ 3. Winning Architecture (`QuantumKernelSVM`)

```text
4 PCA Features ──► Angle Calibration: θᵢ = 0.35 · π · tanh(xᵢ)
                        │
                        ▼
        ┌───────────────────────────────┐
        │   4-Qubit Feature Map (d = 3) │
        │  • RY(θᵢ) & RZ(0.5·θᵢ)        │
        │  • Circular CZ: (0-1-2-3-0)   │
        └───────────────┬───────────────┘
                        │  Exact Statevector Fidelity
                        ▼
        Kernel Matrix: K(x, y) = |⟨ψ(x)|ψ(y)⟩|²
                        │
                        ▼
        Balanced SVC (C = 3.0, class_weight='balanced')
                        │
                        ▼
        Crop Prediction: {alfalfa, maize, wheat}
```

### Circuit Resource Summary (Exact Local Simulation)
* **Qubits:** `4` (strictly within the $\le 8$ qubit limit)
* **Circuit Depth:** `3`
* **Entangling Gates:** `4` two-qubit `CZ` gates (circular topology)
* **Trainable Quantum Parameters:** `0` (convex dual optimization via `SVC`)
* **Determinism:** Exact `Statevector` evaluation with `random_state=0` yields **zero seed variance** across seeds `0, 1, 2`.

---

## 📂 4. Repository Structure

```text
├── challenge/
│   ├── models.py          # Implements QuantumKernelSVM, QuantumClassifier ('ry', 'zz', 'reupload'), & MatchedHead
│   ├── submission.py      # Official judging interface wired to QuantumKernelSVM (scale=0.35, C=3.0)
│   └── ...                # Data loaders, leakage-free PCA transforms, and evaluation metrics
├── notebooks/
│   └── Qiskit_QuantumAfrica_Crop_Challenge.ipynb  # Full walkthrough, ablation studies, and validation plots
├── configs/               # Baseline and experiment configuration files
├── data/                  # Prepared Doukkala crop dataset (914 train / 150 val / 221 unlabeled test)
├── results/               # Reference starter outputs
└── requirements.txt       # Pinned Python 3.12 + Qiskit 2.4.2 environment
```

---

## 🚀 5. Installation & Reproduction

### 1. Environment Setup (Python 3.12)
```bash
git clone https://github.com/walid-mk/QiskitFallFest_EMSI_QuantumAfrica.git
cd QiskitFallFest_EMSI_QuantumAfrica
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name crop-challenge --display-name "Crop challenge"
```
*(On Windows PowerShell, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.)*

### 2. Verify the Submission Pipeline
You can verify our final submission model (`challenge/submission.py`) directly from Python in **< 1 second**:
```python
from challenge import submission

sub = submission.Submission()
# Fits on the 914 training fields and predicts on validation/test views
```
Or open `notebooks/Qiskit_QuantumAfrica_Crop_Challenge.ipynb`, select the **Crop challenge** kernel, ensure `USE_FULL_TRAINING = True`, and run all cells to reproduce every metric, confusion matrix, and ablation table.

---

## 📜 6. Data Rules & Acknowledgments

* **Strict Leakage Prevention:** All PCA projections and feature scalers are fitted exclusively on the `914` training fields. Validation (`150` fields) was used strictly for model selection, and test (`221` fields) remains completely untouched for organizer evaluation.
* **Dataset Attribution:** Derived from the Moroccan agricultural parcel dataset by *Ouassanouan et al. (2026)* ([Nature Scientific Data](https://www.nature.com/articles/s41597-026-06993-y)), licensed under **CC BY 4.0**.
* **Challenge Framework:** Built upon the official Qiskit Fall Fest · EMSI Casablanca · Quantum Africa starter repository (MIT License).