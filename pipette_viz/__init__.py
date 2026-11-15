import warnings
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

__version__ = "1.2.0"
__author__ = "Mateusz Jakubowski"
__homepage__ = "https://mateusz-jakubowski.ai.studio/"

ECOSYSTEM_LINKS = {
    "Main Hub": "https://mateusz-jakubowski.ai.studio/",
    "Project Showroom": "https://from-pipette-to-python.ai.studio/",
    "LinkedIn": "https://www.linkedin.com/in/mateuszjakubowski",
    "GitHub": "https://github.com/MatthewJakubowski",
    "Kaggle": "https://www.kaggle.com/matthewjakubowski",
    "Hugging Face": "https://huggingface.co/matthewjakubowski",
    "X (Twitter)": "https://x.com/M_S_Jakubowski",
    "Vivino": "http://www.vivino.com/users/mateusz.jakubowski/",
}

PALETTE = {
    "train": "#005A9C",     # Cobalt Blue - zbiór treningowy / główny sygnał
    "primary": "#005A9C",   # Alias dla analiz ogólnych
    "val": "#D9534F",       # Crimson Red - zbiór walidacyjny / CITO / granica decyzyjna
    "cito": "#D9534F",      # Alias kliniczny
    "test": "#2A9D8F",      # Clinical Teal - zbiór testowy / zakres referencyjny (norma)
    "ref": "#2A9D8F",       # Alias kliniczny
    "warn": "#E76F51",      # Amber Coral - ostrzeżenie / próg pośredni / regularyzacja
    "latent": "#6C5B7B",    # Deep Violet - cechy ukryte / SHAP / warstwy sieci neuronowych
    "xai": "#6C5B7B",       # Alias XAI
    "baseline": "#64748B",  # Cool Slate - model bazowy (Dummy) / szum / grupa kontrolna
}


def init_viz(mode: str = "clinical", dpi: int = 130, quiet: bool = False) -> dict:
    """Inicjalizuje silnik wizualizacji PipetteViz dla Matplotlib oraz Seaborn."""
    style_path = Path(__file__).parent / f"{mode}.mplstyle"

    if not style_path.exists():
        raise FileNotFoundError(f"[ERROR] Style profile '{mode}.mplstyle' not found.")

    sns.reset_orig()
    plt.style.use(str(style_path))
    plt.rcParams["figure.dpi"] = dpi

    palette_hex = [f"#{c.lstrip('#')}" for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
    sns.set_palette(palette_hex)

    if not quiet:
        print(
            f"⚡ [PipetteViz v{__version__}] Profile: '{mode.upper()}' | DPI: {dpi} | "
            f"Hub: mateusz-jakubowski.ai.studio"
        )

    return PALETTE


def _get_primary_axes(fig: plt.Figure) -> List[plt.Axes]:
    """Odfiltrowuje colorbary oraz osie twinx()."""
    primary_axes: List[plt.Axes] = []
    seen_bboxes = []

    for a in fig.axes:
        label = a.get_label()
        if label == "<colorbar>" or hasattr(a, "_colorbar") or "colorbar" in label.lower():
            continue

        pos = a.get_position()
        bbox_key = (round(pos.x0, 3), round(pos.y0, 3), round(pos.width, 3), round(pos.height, 3))

        if bbox_key not in seen_bboxes:
            seen_bboxes.append(bbox_key)
            primary_axes.append(a)

    return primary_axes


def _clear_previous_branding(fig: plt.Figure, ax: plt.Axes) -> None:
    """Usuwa poprzednie napisy wygenerowane przez brand_plot (idempotentność)."""
    for txt in list(fig.texts):
        if getattr(txt, "_pipette_viz_tag", False):
            txt.remove()
    for txt in list(ax.texts):
        if getattr(txt, "_pipette_viz_tag", False):
            txt.remove()


def brand_plot(
    title: str,
    subtitle: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    watermark: str = "#FromPipetteToPython • mateusz-jakubowski.ai.studio",
    source: Optional[str] = None
) -> plt.Axes:
    """Nadaje wykresowi układ redakcyjny (Dashboard / Single-Panel z obsługą twinx i colorbar)."""
    if fig is None:
        fig = ax.get_figure() if ax is not None else plt.gcf()

    primary_axes = _get_primary_axes(fig)
    is_multipanel = len(primary_axes) > 1

    if ax is None:
        ax = primary_axes[0] if primary_axes else plt.gca()

    _clear_previous_branding(fig, ax)

    if is_multipanel:
        # Podnosimy główny tytuł wyżej (y=0.99) oraz podtytuł (y=0.93), 
        # żeby zrobić czystą przestrzeń nad tytulikami A. i B. podwykresów.
        fig.suptitle(
            title, x=0.01, y=0.99, ha="left",
            fontsize=14, fontweight="bold", color="#0F172A"
        )
        if subtitle:
            sub_txt = fig.text(
                0.01, 0.93, subtitle, ha="left",
                fontsize=10.5, fontweight="normal", color="#475569"
            )
            sub_txt._pipette_viz_tag = True

    else:
        if subtitle:
            ax.set_title(title, loc="left", pad=24, fontsize=14, fontweight="bold", color="#0F172A")
            sub_txt = ax.text(
                x=0.0, y=1.03, s=subtitle,
                transform=ax.transAxes,
                fontsize=10.5, fontweight="normal", color="#475569"
            )
            sub_txt._pipette_viz_tag = True
        else:
            ax.set_title(title, loc="left", pad=15, fontsize=14, fontweight="bold", color="#0F172A")

    if source:
        src_txt = fig.text(
            0.01, -0.03, f"Source: {source}",
            fontsize=8.5, color="#94A3B8", ha="left", style="italic"
        )
        src_txt._pipette_viz_tag = True

    if watermark:
        wm_txt = fig.text(
            0.99, -0.03, watermark,
            fontsize=8.5, color="#64748B", ha="right", fontweight="bold"
        )
        wm_txt._pipette_viz_tag = True

    return ax


def add_ref_band(
    low: float,
    high: float,
    label: str = "Reference / Target Zone",
    axis: str = "y",
    color: str = "#2A9D8F",
    alpha: float = 0.12,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Rysuje pasmo referencyjne na wskazanej osi (np. ax=ax2)."""
    target_ax = ax if ax is not None else plt.gca()

    if axis.lower() == "y":
        target_ax.axhspan(low, high, color=color, alpha=alpha, label=label, zorder=1)
    else:
        target_ax.axvspan(low, high, color=color, alpha=alpha, label=label, zorder=1)

    return target_ax


def about() -> None:
    """Wyświetla w konsoli wizytówkę autora i mapę linków ekosystemu."""
    print(f"=== 🧬 PipetteViz SDK v{__version__} // {__author__} ===")
    for name, url in ECOSYSTEM_LINKS.items():
        print(f"  • {name:<18}: {url}")
