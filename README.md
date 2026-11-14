# 🧬 `pipette-viz` // Clinical AI & Machine Learning Visualization SDK
[![Quality Control CI](https://github.com/MatthewJakubowski/pipette-viz/actions/workflows/ci.yml/badge.svg)](https://github.com/MatthewJakubowski/pipette-viz/actions/workflows/ci.yml)
[![Main Hub](https://img.shields.io/badge/🌐_Main_Hub-mateusz--jakubowski.ai.studio-005A9C?style=flat-square)](https://mateusz-jakubowski.ai.studio/)
[![Showroom](https://img.shields.io/badge/🚀_Showroom-from--pipette--to--python-D9534F?style=flat-square)](https://from-pipette-to-python.ai.studio/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mateusz_Jakubowski-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mateuszjakubowski)
[![Kaggle](https://img.shields.io/badge/Kaggle-matthewjakubowski-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/matthewjakubowski)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-matthewjakubowski-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/matthewjakubowski)
[![X](https://img.shields.io/badge/X-@M__S__Jakubowski-000000?style=flat-square&logo=x&logoColor=white)](https://x.com/M_S_Jakubowski)
[![License: MIT](https://img.shields.io/badge/License-MIT-2A9D8F?style=flat-square)](https://opensource.org/licenses/MIT)

**Silicon Valley-grade visualization engine and personal branding toolkit** for **Matplotlib** & **Seaborn**. Engineered for high-precision diagnostic analytics, classical Machine Learning (`Scikit-Learn`), Explainable AI (`SHAP`/`LIME`), and Deep Learning training dynamics.

---

## ⚡ 1. One-Line Installation (Google Colab / Kaggle / Pydroid 3)

Install directly from GitHub in the top cell of your notebook:

```python
!pip install -q git+[https://github.com/MatthewJakubowski/pipette-viz.git](https://github.com/MatthewJakubowski/pipette-viz.git)
```

---

## 🚀 2. Quick Bootstrap & Personal Branding

`pipette-viz` configures `rcParams`, syncs **Seaborn** palettes, and provides `brand_plot()`—an editorial layout helper that adds publication-style subtitles and your signature `#FromPipetteToPython • mateusz-jakubowski.ai.studio` watermark to every exported chart:

```python
import matplotlib.pyplot as plt
import seaborn as sns
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
⚡ [PipetteViz v1.1.0] Profile: 'CLINICAL' | DPI: 130 | Hub: mateusz-jakubowski.ai.studio
```

---

## 🎨 3. Dual-Domain Semantic Palette (`PALETTE`)

Every color key is mapped to both **Machine Learning / Deep Learning** workflows and **Clinical Diagnostics**:

| Key / Alias | Hex | ML & Deep Learning Role | Clinical & XAI Role |
| :--- | :--- | :--- | :--- |
| `train` / `primary` | `#005A9C` | **Training Curve** / Primary Model | Baseline Signal / Primary Cohort |
| `val` / `cito` | `#D9534F` | **Validation Loss** / Decision Boundary | Critical Alert (CITO) / Anomaly |
| `test` / `ref` | `#2A9D8F` | **Test Evaluation** / Target Metric | Reference Range (Norm) / Validated |
| `warn` | `#E76F51` | **Regularization** / Hyperparameter Shift | Borderline Zone / Warning |
| `latent` / `xai` | `#6C5B7B` | **Latent Space** / Neural Layer Activations | SHAP Feature Impact / Interactions |
| `baseline` | `#64748B` | **Dummy Classifier** / Noise Floor | Control Group / Unadjusted Baseline |

---

## 🛠️ 4. Direct Raw URL Fallback (Zero-Install Mode)

When working in locked environments without `pip` access (e.g., restricted Kaggle kernels without internet installation), inject the raw stylesheet directly:

```python
import matplotlib.pyplot as plt
plt.style.use("[https://raw.githubusercontent.com/MatthewJakubowski/pipette-viz/main/pipette_viz/clinical.mplstyle](https://raw.githubusercontent.com/MatthewJakubowski/pipette-viz/main/pipette_viz/clinical.mplstyle)")
```

---

## 🌐 Connect & Explore the Ecosystem

* **Primary Portfolio & AI Hub:** [mateusz-jakubowski.ai.studio](https://mateusz-jakubowski.ai.studio/)
* **#FromPipetteToPython Showroom:** [from-pipette-to-python.ai.studio](https://from-pipette-to-python.ai.studio/)
* **LinkedIn (Newsletter & Insights):** [linkedin.com/in/mateuszjakubowski](https://www.linkedin.com/in/mateuszjakubowski)
* **GitHub (Open-Source MedTech & ML):** [github.com/MatthewJakubowski](https://github.com/MatthewJakubowski)
* **Kaggle (Notebooks & Competitions):** [kaggle.com/matthewjakubowski](https://www.kaggle.com/matthewjakubowski)
* **Hugging Face (Models & Spaces):** [huggingface.co/matthewjakubowski](https://huggingface.co/matthewjakubowski)
* **X / Twitter:** [@M_S_Jakubowski](https://x.com/M_S_Jakubowski)
* **After-Hours (Oenology & Wine Analytics):** [Vivino @mateusz.jakubowski](http://www.vivino.com/users/mateusz.jakubowski/)
