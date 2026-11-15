# 🧬 `pipette-viz` // Clinical AI & Machine Learning Visualization SDK

[![Quality Control CI](https://github.com/MatthewJakubowski/pipette-viz/actions/workflows/ci.yml/badge.svg)](https://github.com/MatthewJakubowski/pipette-viz/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/release-v1.2.0-005A9C?style=flat-square)](https://github.com/MatthewJakubowski/pipette-viz/releases)
[![Main Hub](https://img.shields.io/badge/🌐_Main_Hub-mateusz--jakubowski.ai.studio-005A9C?style=flat-square)](https://mateusz-jakubowski.ai.studio/)
[![Showroom](https://img.shields.io/badge/🚀_Showroom-from--pipette--to--python-D9534F?style=flat-square)](https://from-pipette-to-python.ai.studio/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mateusz_Jakubowski-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mateuszjakubowski)
[![Kaggle](https://img.shields.io/badge/Kaggle-matthewjakubowski-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/matthewjakubowski)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-matthewjakubowski-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/matthewjakubowski)
[![X](https://img.shields.io/badge/X-@M__S__Jakubowski-000000?style=flat-square&logo=x&logoColor=white)](https://x.com/M_S_Jakubowski)
[![License: MIT](https://img.shields.io/badge/License-MIT-2A9D8F?style=flat-square)](https://opensource.org/licenses/MIT)

**Silicon Valley-grade visualization engine and personal branding toolkit** for **Matplotlib**, **Pandas**, & **Seaborn**. Engineered for high-precision diagnostic analytics, classical Machine Learning (`Scikit-Learn`), Explainable AI (`SHAP` / `LIME`), and Deep Learning training dynamics.

---

## ⚡ 1. One-Line Installation (Google Colab / Kaggle / Jupyter)

Install directly from GitHub in the top cell of your notebook:

```python
!pip install -q --upgrade git+https://github.com/MatthewJakubowski/pipette-viz.git
```

---

## 🚀 2. Single-Panel Bootstrap & Editorial Branding

`pipette-viz` configures `rcParams`, syncs **Seaborn** palettes, and provides `brand_plot()`—a Physical Inch-Space Layout Engine that adds publication-style headers and your signature `#FromPipetteToPython • mateusz-jakubowski.ai.studio` watermark to every exported chart:

```python
import matplotlib.pyplot as plt
from pipette_viz import init_viz, brand_plot, add_ref_band, about

# 1. Initialize engine & retrieve semantic color dictionary
c = init_viz(mode="clinical", dpi=130)

# 2. Plot your ML or clinical data (zero manual styling needed)
plt.plot(train_sizes, train_scores, marker="o", label="Training Score (ROC-AUC)")
plt.plot(train_sizes, val_scores, marker="s", label="Cross-Validation Score (5-Fold)")

# 3. Add target zone band & editorial branding
add_ref_band(low=0.88, high=0.95, label="Clinical Acceptance Zone (ISO 15189)")
brand_plot(
    title="Random Forest Learning Curve — Bias-Variance Tradeoff",
    subtitle="Scikit-Learn Pipeline | n = 1,450 samples | Hyperparameters: max_depth=8, n_estimators=200",
    source="Validation Experiment #FromPipetteToPython"
)

plt.xlabel("Training Set Size (m)")
plt.ylabel("ROC-AUC Score")
plt.legend()
plt.show()
```

Console output on initialization:
```text
⚡ [PipetteViz v1.2.0] Profile: 'CLINICAL' | DPI: 130 | Hub: mateusz-jakubowski.ai.studio
```

---

## 🧩 3. Multi-Panel Dashboards, df.hist() & XAI (v1.2.0 Engine)

Unlike static wrappers, `pipette-viz v1.2.0` calculates header and footer margins dynamically in **physical inches** (`fig.get_size_inches()`), guaranteeing zero text collisions across:
​- **Multi-panel dashboards** (`plt.subplots(1, 2)`, `plt.subplots(2, 2)`) while preserving local subplot titles (`A., B., C., D.`),
- **​High-dimensional Pandas feature grids** (`axes = df.hist(figsize=(14, 10))` returning `np.ndarray`),
​- **Explainable AI** (`SHAP` / `ClusterSimilarity`) charts with auxiliary `colorbar` or dual Y-axes (`ax.twinx()`).

```python
# Multi-Panel MLP / Deep Learning Dashboard Example
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))

ax1.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.7)
ax1.set_title("A. Decision Boundary (MLP Hidden Layer)")

ax2.plot(epochs, val_loss, color=c["val"], lw=2.5, label="Validation Loss")
add_ref_band(low=0.05, high=0.15, label="Target Convergence Corridor", ax=ax2)
ax2.set_title("B. Cross-Entropy Loss Convergence")
ax2.legend()

brand_plot(
    title="MLP Neural Network Architecture — Decision Space & Convergence",
    subtitle="Hidden Layers: (64, 32) | Activation: ReLU | Optimizer: Adam",
    source="Experiment #FromPipetteToPython",
    fig=fig
)
plt.show()
```

---

## 🎨 4. Dual-Domain Semantic Palette (`PALETTE`)

Every color key is mapped to both **Machine Learning / Deep Learning** workflows and **Clinical Diagnostics & XAI**:

| Key / Alias | Hex | ML & Deep Learning Role | Clinical & XAI Role |
| :--- | :--- | :--- | :--- |
| `train` / `primary` | `#005A9C` | **Training Curve** / Primary Signal | Baseline Signal / Primary Cohort |
| `val` / `cito` | `#D9534F` | **Validation Loss** / Decision Boundary | Critical Alert (CITO) / High SHAP Impact |
| `test` / `ref` | `#2A9D8F` | **Test Evaluation** / Generalization | Reference Corridor / Target Range |
| `warn` | `#E76F51` | **Regularization** / Threshold Alert | Borderline / Intermediate Risk |
| `latent` / `xai` | `#6C5B7B` | **Latent Space** / Hidden Representations | **SHAP/LIME** Feature Attribution |
| `baseline` | `#64748B` | **Dummy Classifier** / Noise Floor | Control Group / Background Distribution |

---

## 🛠️ 5. Direct Raw URL Fallback (Zero-Install Mode)

When working in locked environments without `pip` access (e.g., restricted Kaggle kernels without internet installation), inject the raw stylesheet directly:

```python
import matplotlib.pyplot as plt
plt.style.use("[https://raw.githubusercontent.com/MatthewJakubowski/pipette-viz/main/pipette_viz/clinical.mplstyle](https://raw.githubusercontent.com/MatthewJakubowski/pipette-viz/main/pipette_viz/clinical.mplstyle)")
```

---

## 🌐 6. Connect & Explore the Ecosystem

Engineered by **Mateusz Jakubowski** — bridging Laboratory Medicine & Clinical Diagnostics with Production Machine Learning and Explainable AI.

```python
from pipette_viz import about
about()
```

---
