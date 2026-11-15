import warnings
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__version__ = "1.2.0"
__author__ = "Mateusz Jakubowski"
__homepage__ = "https://mateusz-jakubowski.ai.studio"

ECOSYSTEM_LINKS = {
    "Main Hub": "https://mateusz-jakubowski.ai.studio",
    "Project Showroom": "https://from-pipette-to-python.ai.studio",
    "LinkedIn": "https://linkedin.com/in/mateuszjakubowski",
    "GitHub": "https://github.com/MatthewJakubowski",
    "Kaggle": "https://kaggle.com/matthewjakubowski",
    "Hugging Face": "https://huggingface.co/matthewjakubowski",
    "X (Twitter)": "https://x.com/M_S_Jakubowski",
    "Vivino": "https://www.vivino.com/users/mateusz.jakubowski",
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
    "xai": "#6C5B7B",       # Alias XAI (SHAP / LIME)
    "baseline": "#64748B",  # Cool Slate - model bazowy (Dummy) / szum / grupa kontrolna
}


def init_viz(mode: str = "clinical", dpi: int = 130, quiet: bool = False) -> dict:
    """
    Inicjalizuje silnik wizualizacji PipetteViz dla Matplotlib, Pandas oraz Seaborn.
    Synchronizuje cykl kolorów dla klasycznego ML (Scikit-Learn), XAI oraz Deep Learningu.
    """
    style_path = Path(__file__).parent / f"{mode}.mplstyle"

    if not style_path.exists():
        raise FileNotFoundError(f"[ERROR] Style profile '{mode}.mplstyle' not found.")

    sns.reset_orig()
    plt.style.use(str(style_path))
    # Gwarantuje, że ax.set_title() i ax.get_title() na podwykresach i df.hist()
    # używają standardowego slotu 'center', a brand_plot() sam ustawia loc='left' dla single-panel
    plt.rcParams["axes.titlelocation"] = "center"
    plt.rcParams["figure.dpi"] = dpi

    palette_hex = [f"#{c.lstrip('#')}" for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
    sns.set_palette(palette_hex)

    if not quiet:
        print(
            f"⚡ [PipetteViz v{__version__}] Profile: '{mode.upper()}' | DPI: {dpi} | "
            f"Hub: mateusz-jakubowski.ai.studio"
        )

    return PALETTE


def _resolve_figure_and_ax(
    ax: Optional[Any] = None,
    fig: Optional[plt.Figure] = None
) -> tuple[plt.Figure, Optional[plt.Axes]]:
    """
    Uniwersalny ekstraktor obsługujący:
    - pojedynczy plt.Axes
    - tablicę np.ndarray osi zwracaną przez df.hist() lub plt.subplots(2, 2)
    - obiekty Seaborn (FacetGrid, JointGrid)
    """
    resolved_ax: Optional[plt.Axes] = None

    if ax is not None:
        if isinstance(ax, plt.Axes):
            resolved_ax = ax
        elif isinstance(ax, (np.ndarray, list, tuple)):
            flat = np.asarray(ax, dtype=object).ravel()
            for item in flat:
                if isinstance(item, plt.Axes):
                    resolved_ax = item
                    break
        elif hasattr(ax, "fig"):  # Seaborn FacetGrid / JointGrid
            fig = ax.fig
            if getattr(ax, "axes", None) is not None:
                flat = np.asarray(ax.axes, dtype=object).ravel()
                if len(flat) > 0 and isinstance(flat[0], plt.Axes):
                    resolved_ax = flat[0]

    if fig is None:
        if resolved_ax is not None:
            fig = resolved_ax.get_figure()
        else:
            fig = plt.gcf()

    return fig, resolved_ax


def _get_primary_axes(fig: plt.Figure) -> List[plt.Axes]:
    """
    Izoluje główne panele wykresu od osi pomocniczych:
    - Ignoruje paski kolorów (plt.colorbar(), wykresy gęstości i SHAP).
    - Ignoruje osie bliźniacze ax.twinx() / ax.twiny() (identyczny bounding box).
    """
    primary_axes: List[plt.Axes] = []
    seen_bboxes = []

    for a in fig.axes:
        label = (a.get_label() or "").lower()
        if label == "<colorbar>" or "colorbar" in label or hasattr(a, "_colorbar"):
            continue

        pos = a.get_position()
        # Dodatkowa heurystyka: bardzo wąskie osie boczne (np. colorbar z SHAP bez etykiety)
        if pos.width < 0.04 or pos.height < 0.04:
            continue

        bbox_key = (round(pos.x0, 3), round(pos.y0, 3), round(pos.width, 3), round(pos.height, 3))
        if bbox_key not in seen_bboxes:
            seen_bboxes.append(bbox_key)
            primary_axes.append(a)

    return primary_axes


def _clear_previous_branding(fig: plt.Figure) -> None:
    """Usuwa poprzednie elementy tekstowe PipetteViz (100% idempotentność)."""
    for txt in list(fig.texts):
        if getattr(txt, "_pipette_viz_tag", False):
            txt.remove()
    for a in fig.axes:
        for txt in list(a.texts):
            if getattr(txt, "_pipette_viz_tag", False):
                txt.remove()


def brand_plot(
    title: str,
    subtitle: Optional[str] = None,
    ax: Optional[Any] = None,
    fig: Optional[plt.Figure] = None,
    watermark: str = "#FromPipetteToPython • mateusz-jakubowski.ai.studio",
    source: Optional[str] = None
) -> plt.Axes:
    """
    Silnik kompozycji i brandingu (Physical Inch-Space Layout Engine):
    - Działa identycznie na pojedynczym wykresie, dashboardzie 1x2, siatce 2x2,
      wielowymiarowym df.hist() (np. 12 cech) oraz wizualizacjach XAI (SHAP).
    - Przelicza marginesy w fizycznych calach, gwarantując zero nachodzenia tekstu
      na lokalne tytuły podwykresów niezależnie od rozmiaru figury (figsize).
    """
    fig, resolved_ax = _resolve_figure_and_ax(ax=ax, fig=fig)
    primary_axes = _get_primary_axes(fig)
    is_multipanel = len(primary_axes) > 1

    if resolved_ax is None:
        resolved_ax = primary_axes[0] if primary_axes else plt.gca()

    _clear_previous_branding(fig)

    fig_w, fig_h = fig.get_size_inches()
    fig_h = max(float(fig_h), 3.5)

    if is_multipanel:
        # Fizyczna rezerwacja miejsca w calach od góry i od dołu (niezależna od figsize!)
        header_inches = 0.82 if subtitle else 0.52
        footer_inches = 0.38 if (source or watermark) else 0.15

        top_limit = max(0.55, 1.0 - (header_inches / fig_h))
        bottom_limit = min(0.25, footer_inches / fig_h)

        # Zapisujemy lokalne tytuły przed operacją layoutu (ochrona przed ich utratą)
        saved_titles = [
            (a, a.get_title(loc="center"), a.get_title(loc="left"), a.get_title(loc="right"))
            for a in primary_axes
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                fig.tight_layout(rect=[0.0, bottom_limit, 1.0, top_limit])
            except Exception:
                fig.subplots_adjust(top=top_limit - 0.04, bottom=bottom_limit + 0.04)

        # Przywrócenie lokalnych tytułów podwykresów (np. "A. ...", "B. ..." lub nazw kolumn z df.hist())
        for a, t_center, t_left, t_right in saved_titles:
            if t_center:
                a.set_title(t_center, loc="center")
            if t_left:
                a.set_title(t_left, loc="left")
            if t_right:
                a.set_title(t_right, loc="right")

        # Wyznaczamy lewą krawędź wyrównaną dokładnie do pierwszego panelu siatki
        left_x = min(a.get_position().x0 for a in primary_axes)
        right_x = max(a.get_position().x1 for a in primary_axes)

        title_y = 1.0 - (0.14 / fig_h)
        sub_y = 1.0 - (0.44 / fig_h)
        footer_y = max(0.008, (bottom_limit * 0.35))

        sup = fig.suptitle(
            title, x=left_x, y=title_y, ha="left", va="top",
            fontsize=14, fontweight="bold", color="#0F172A"
        )
        sup._pipette_viz_tag = True

        if subtitle:
            sub_txt = fig.text(
                left_x, sub_y, subtitle, ha="left", va="top",
                fontsize=10.5, fontweight="normal", color="#475569"
            )
            sub_txt._pipette_viz_tag = True

        if source:
            src_txt = fig.text(
                left_x, footer_y, f"Source: {source}",
                fontsize=8.5, color="#94A3B8", ha="left", va="bottom", style="italic"
            )
            src_txt._pipette_viz_tag = True

        if watermark:
            wm_txt = fig.text(
                right_x, footer_y, watermark,
                fontsize=8.5, color="#64748B", ha="right", va="bottom", fontweight="bold"
            )
            wm_txt._pipette_viz_tag = True

    else:
        # Tryb Single-Panel (w tym pojedynczy wykres z colorbar lub twinx)
        if subtitle:
            resolved_ax.set_title(
                title, loc="left", pad=24, fontsize=14, fontweight="bold", color="#0F172A"
            )
            sub_txt = resolved_ax.text(
                x=0.0, y=1.03, s=subtitle,
                transform=resolved_ax.transAxes,
                fontsize=10.5, fontweight="normal", color="#475569"
            )
            sub_txt._pipette_viz_tag = True
        else:
            resolved_ax.set_title(
                title, loc="left", pad=15, fontsize=14, fontweight="bold", color="#0F172A"
            )

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

    return resolved_ax


def add_ref_band(
    low: float,
    high: float,
    label: str = "Reference / Target Zone",
    axis: str = "y",
    color: str = "#2A9D8F",
    alpha: float = 0.12,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Rysuje pasmo referencyjne / korytarz konwergencji na wskazanej osi (lub wszystkich osiach).
    """
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
