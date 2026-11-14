from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

__version__ = "1.1.0"
__author__ = "Mateusz Jakubowski"
__homepage__ = "https://mateusz-jakubowski.ai.studio/"

# Centralny rejestr Twojego ekosystemu zaszyty w bibliotece
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

# Semantyczny słownik kolorów pod Klasyczny ML (Scikit-Learn), Deep Learning i Diagnostykę
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
    """
    Inicjalizuje silnik wizualizacji PipetteViz dla Matplotlib oraz Seaborn.
    Synchronizuje cykl kolorów dla klasycznego ML (Scikit-Learn) oraz Deep Learningu.
    """
    style_path = Path(__file__).parent / f"{mode}.mplstyle"
    
    if not style_path.exists():
        raise FileNotFoundError(f"[ERROR] Style profile '{mode}.mplstyle' not found.")

    # 1. Reset do czystego stanu i narzucenie Twojego arkusza .mplstyle
    sns.reset_orig()
    plt.style.use(str(style_path))
    plt.rcParams["figure.dpi"] = dpi
    
    # 2. Synchronizacja palety Seaborna z plikiem .mplstyle
    palette_hex = [f"#{c.lstrip('#')}" for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
    sns.set_palette(palette_hex)
    
    if not quiet:
        print(
            f"⚡ [PipetteViz v{__version__}] Profile: '{mode.upper()}' | DPI: {dpi} | "
            f"Hub: mateusz-jakubowski.ai.studio"
        )
        
    return PALETTE


def brand_plot(
    title: str,
    subtitle: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    watermark: str = "#FromPipetteToPython • mateusz-jakubowski.ai.studio",
    source: Optional[str] = None
) -> plt.Axes:
    """
    Nadaje wykresowi układ redakcyjny (Silicon Valley / Publication Style):
    - Główny pogrubiony tytuł po lewej stronie
    - Techniczny podtytuł z metrykami lub opisem zbioru danych poniżej
    - Dyskretny znak wodny (Personal Branding) kierujący na mateusz-jakubowski.ai.studio
    """
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()

    # 1. Tytuł i podtytuł
    if subtitle:
        ax.set_title(title, loc="left", pad=24, fontsize=14, fontweight="bold", color="#0F172A")
        ax.text(
            x=0.0, y=1.03, s=subtitle,
            transform=ax.transAxes,
            fontsize=10.5, fontweight="normal", color="#475569"
        )
    else:
        ax.set_title(title, loc="left", pad=15, fontsize=14, fontweight="bold", color="#0F172A")

    # 2. Stopka brandingowa (Watermark + opcjonalne źródło danych)
    footer_left = f"Source: {source}" if source else ""
    
    if footer_left:
        fig.text(0.01, -0.03, footer_left, fontsize=8.5, color="#94A3B8", ha="left", style="italic")
    if watermark:
        fig.text(0.99, -0.03, watermark, fontsize=8.5, color="#64748B", ha="right", fontweight="bold")

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
    """
    Rysuje zacieniowane pasmo na wykresie (np. zakres referencyjny w laboratorium
    lub docelowy korytarz metryki / przedział ufności w Machine Learningu).
    """
    if ax is None:
        ax = plt.gca()
        
    if axis.lower() == "y":
        ax.axhspan(low, high, color=color, alpha=alpha, label=label, zorder=1)
    else:
        ax.axvspan(low, high, color=color, alpha=alpha, label=label, zorder=1)
        
    return ax


def about() -> None:
    """Wyświetla w konsoli wizytówkę autora i pełną mapę linków ekosystemu."""
    print(f"=== 🧬 PipetteViz SDK v{__version__} // {__author__} ===")
    for name, url in ECOSYSTEM_LINKS.items():
        print(f"  • {name:<18}: {url}")
