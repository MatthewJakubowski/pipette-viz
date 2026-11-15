import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pipette_viz import (
    init_viz,
    brand_plot,
    add_ref_band,
    PALETTE,
    ECOSYSTEM_LINKS,
    __version__,
)


def test_version_and_links():
    assert __version__ == "1.2.0"
    assert ECOSYSTEM_LINKS["Main Hub"] == "https://mateusz-jakubowski.ai.studio"


def test_init_viz_engine():
    colors = init_viz(mode="clinical", dpi=130, quiet=True)
    assert isinstance(colors, dict)
    assert colors["train"] == "#005A9C"
    assert colors["cito"] == "#D9534F"
    assert colors["xai"] == "#6C5B7B"
    assert plt.rcParams["figure.dpi"] == 130
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False


def test_single_panel_brand_plot_and_ref_band():
    init_viz(quiet=True)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [0.80, 0.88, 0.93])

    add_ref_band(low=0.85, high=0.95, label="Target Zone", ax=ax)
    brand_plot(
        title="Test Krzywej Uczenia",
        subtitle="Walidacja CI/CD | n = 100",
        source="Automated Pytest Suite",
        ax=ax,
    )

    assert ax.get_title(loc="left") == "Test Krzywej Uczenia"
    plt.close(fig)


def test_single_panel_with_colorbar_not_treated_as_multipanel():
    init_viz(quiet=True)
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2, 3], [1, 2, 3], c=[10, 20, 30])
    fig.colorbar(sc, ax=ax)

    brand_plot(title="ClusterSimilarity Projection", subtitle="KMeans + RBF Kernel")

    assert ax.get_title(loc="left") == "ClusterSimilarity Projection"
    assert fig._suptitle is None
    plt.close(fig)


def test_single_panel_with_twinx_not_treated_as_multipanel():
    init_viz(quiet=True)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot([1, 2, 3], [0.5, 0.3, 0.1])
    ax2.plot([1, 2, 3], [0.7, 0.85, 0.95])

    brand_plot(title="Precision vs Recall Curve", subtitle="Dual Y-Axis Check")

    assert ax1.get_title(loc="left") == "Precision vs Recall Curve"
    assert fig._suptitle is None
    plt.close(fig)


def test_brand_plot_idempotency():
    init_viz(quiet=True)
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])

    brand_plot(title="First Title", subtitle="Sub 1", source="Src 1")
    brand_plot(title="Updated Title", subtitle="Sub 2", source="Src 2")

    tagged_fig_texts = [t for t in fig.texts if getattr(t, "_pipette_viz_tag", False)]
    tagged_ax_texts = [t for t in ax.texts if getattr(t, "_pipette_viz_tag", False)]

    assert len(tagged_fig_texts) == 2
    assert len(tagged_ax_texts) == 1
    assert ax.get_title(loc="left") == "Updated Title"
    plt.close(fig)


def test_multipanel_subplots_preserves_subtitles_and_targets_ax2():
    init_viz(quiet=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))

    ax1.plot([0, 1], [0, 1])
    ax1.set_title("A. Granica decyzyjna MLPClassifier")

    ax2.plot([1, 2, 3], [0.5, 0.2, 0.1])
    ax2.set_title("B. Krzywa spadku funkcji straty (Loss)")

    initial_patches_ax1 = len(ax1.patches)
    initial_patches_ax2 = len(ax2.patches)

    returned_ax = add_ref_band(low=0.05, high=0.15, label="Target Loss Corridor", ax=ax2)
    assert returned_ax is ax2
    assert len(ax1.patches) == initial_patches_ax1
    assert len(ax2.patches) == initial_patches_ax2 + 1

    brand_plot(
        title="Architektura Sieci Neuronowej MLP — Analiza Konwergencji",
        subtitle="Hidden Layers: (64, 32) | Activation: ReLU | Optimizer: Adam",
        source="Stress-Test #4",
        fig=fig,
    )

    assert ax1.get_title() == "A. Granica decyzyjna MLPClassifier"
    assert ax2.get_title() == "B. Krzywa spadku funkcji straty (Loss)"
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Architektura Sieci Neuronowej MLP — Analiza Konwergencji"
    plt.close(fig)


def test_dense_feature_histograms_ndarray_support():
    """Symuluje zachowanie df.hist() zwracającego macierz np.ndarray 9 podwykresów (3x3)."""
    init_viz(quiet=True)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for idx, ax in enumerate(axes.ravel()):
        ax.hist(np.random.randn(50), bins=10)
        ax.set_title(f"Feature_{idx + 1}")

    # Przekazujemy bezpośrednio macierz axes (jak z axes = df.hist())
    brand_plot(
        title="Rozkład Atrybutów Wejściowych — Walidacja Wielowymiarowa (3x3)",
        subtitle="Automatyczna inspekcja skośności i wartości odstających przed transformacją",
        source="EDA Pipeline #FromPipetteToPython",
        ax=axes,
    )

    assert fig._suptitle is not None
    assert axes[0, 0].get_title() == "Feature_1"
    assert axes[2, 2].get_title() == "Feature_9"
    plt.close(fig)
