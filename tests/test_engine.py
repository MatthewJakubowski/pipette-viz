import matplotlib
# Wymuszenie trybu bezokienkowego (Headless) dla serwerów CI/CD na GitHubie
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from pipette_viz import init_viz, brand_plot, add_ref_band, PALETTE, ECOSYSTEM_LINKS, __version__


def test_version_and_links():
    """Weryfikuje integralność metadanych i głównego hubu."""
    assert __version__ == "1.1.0"
    assert ECOSYSTEM_LINKS["Main Hub"] == "https://mateusz-jakubowski.ai.studio/"


def test_init_viz_engine():
    """Sprawdza, czy arkusz .mplstyle poprawnie nadpisuje parametry Matplotliba."""
    colors = init_viz(mode="clinical", dpi=130, quiet=True)
    assert isinstance(colors, dict)
    assert colors["train"] == "#005A9C"
    assert colors["cito"] == "#D9534F"
    assert plt.rcParams["figure.dpi"] == 130
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False


def test_brand_plot_and_ref_band():
    """Testuje renderowanie wykresu z podpisem #FromPipetteToPython i pasmem normy."""
    init_viz(quiet=True)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [0.80, 0.88, 0.93])
    
    add_ref_band(low=0.85, high=0.95, label="Target Zone", ax=ax)
    brand_plot(
        title="Test Krzywej Uczenia",
        subtitle="Walidacja CI/CD | n = 100",
        source="Automated Pytest Suite",
        ax=ax
    )
    
    assert ax.get_title(loc="left") == "Test Krzywej Uczenia"
    plt.close(fig)
