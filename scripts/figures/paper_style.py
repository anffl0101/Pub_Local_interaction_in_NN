"""Plotting style of the paper figures (APS rules): 17.8 cm wide, panels (a), (b), ..., inward ticks, 600 dpi, Okabe-Ito colours.
Use:  from paper_style import STYLE, TWO_COL, panel_label, ...;  plt.rcParams.update(STYLE)
"""
TWO_COL = 7.0      # inches

STYLE = {"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
         "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5,
         "ytick.minor.width": 0.5, "xtick.direction": "in", "ytick.direction": "in",
         "xtick.top": True, "ytick.right": True,
         "xtick.minor.visible": True, "ytick.minor.visible": True,      # small ticks between the numbered ones
         "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
         "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
         "savefig.pad_inches": 0.02}

# Okabe-Ito colours; each meaning keeps its colour in every figure
METHOD = {"AdamW": "#D55E00", "SGLD": "#0072B2"}
LAYER = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]          # layers 1-5
DISTANCE = {1: "#0072B2", 2: "#D55E00", 3: "#009E73", 4: "#E69F00"}      # layers 1, 2, 3, 4 apart
INK = "#555555"


def panel_label(ax, label, dx=-26, dy=4):
    """Write '(a)' at the top-left corner of a panel, just outside the axes."""
    ax.annotate(label, (0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points",
                ha="left", va="bottom", fontsize=9)


def save(fig, path):
    """Save as PDF, keeping axis labels wider than their panel (a tight bounding box ignores their width)."""
    from pathlib import Path
    labels = [a.xaxis.label for a in fig.axes] + [a.yaxis.label for a in fig.axes]
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight", bbox_extra_artists=fig.get_default_bbox_extra_artists() + labels)


def step_number(n):
    """A step count as APS writes numbers: 5000, 38 850 (thin space from five digits on), true minus sign."""
    n = int(round(n))                              # math text: the thin space survives in the PDF (a Unicode one does not)
    s = f"{abs(n):,}".replace(",", r"\,") if abs(n) >= 10_000 else str(abs(n))
    return f"${'-' if n < 0 else ''}{s}$"


def full_steps(axis):
    """Tick labels in full step numbers for an axis plotted in thousands of steps (APS: no power of ten as a unit)."""
    from matplotlib.ticker import FuncFormatter
    axis.set_major_formatter(FuncFormatter(lambda v, _: step_number(1000 * v)))
