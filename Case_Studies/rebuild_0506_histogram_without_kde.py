"""Rebuild Fig. 2(b) from the Plot workspace PDF, without KDE."""

from pathlib import Path
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = ROOT / "Plot/0506_Cost_All.pdf"
OUTPUT_PDF = ROOT / "[paper]MDP-LTL-Opacity/pics/r1/hist/0506_Cost_All.pdf"
SVG = ROOT / ".qa_tmp/0506_cost_all_vector.svg"

COLORS = {
    "78.822327%,61.959839%,54.901123%": ("#C99E8C", r"$\pi$ in opaque run"),
    "27.450562%,36.862183%,39.607239%": ("#465E65", r"$\gamma$ in opaque run"),
    "34.117126%,76.469421%,76.077271%": ("#57C3C2", r"$\pi$ in non-opaque run"),
    "99.606323%,27.058411%,40.391541%": ("#FE4567", r"$\gamma$ in non-opaque run"),
}


def pdf_to_svg() -> None:
    SVG.parent.mkdir(parents=True, exist_ok=True)
    if SVG.exists():
        return
    subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-22.04", "--", "bash", "-lc",
         "pdftocairo -svg "
         f"'{str(SOURCE_PDF).replace('D:', '/mnt/d').replace(chr(92), '/')}' "
         f"'{str(SVG).replace('D:', '/mnt/d').replace(chr(92), '/')}'"],
        check=True,
    )


def extract_bars(svg_text: str, baseline: float, x_scale, y_scale):
    bars = []
    for rgb, (color, label) in COLORS.items():
        pattern = re.compile(
            r'<path style="fill-rule:nonzero;fill:rgb\(' + re.escape(rgb) +
            r'\);fill-opacity:0\.5;[^>]*d="([^"]+)"'
        )
        for match in pattern.finditer(svg_text):
            nums = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', match.group(1))]
            points = list(zip(nums[0::2], nums[1::2]))
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if abs(min(ys) - baseline) < 0.02 and max(ys) > baseline + 0.02:
                bars.append((x_scale(min(xs)), x_scale(max(xs)), y_scale(max(ys)), color, label))
    return bars


def draw_bars(ax, bars):
    used = set()
    for x0, x1, height, color, label in bars:
        shown_label = label if label not in used else None
        used.add(label)
        ax.add_patch(Rectangle((x0, 0), x1 - x0, height,
                               facecolor=color, edgecolor=color,
                               linewidth=0.9, alpha=0.5, label=shown_label))


def main() -> None:
    if not SOURCE_PDF.exists():
        raise FileNotFoundError(f"Plot source does not exist: {SOURCE_PDF}")
    pdf_to_svg()
    text = SVG.read_text(encoding="utf-8")

    main_bars = extract_bars(
        text, 85.564376,
        lambda x: (x - 100.818953) / 205.236994,
        lambda y: (y - 85.564376) / 259.208432,
    )
    inset_bars = extract_bars(
        text, 380.337779,
        lambda x: 0.95 + (x - 102.176454) / 1167.60766,
        lambda y: (y - 380.337779) / 98.679006,
    )
    if not main_bars or not inset_bars:
        raise RuntimeError("Could not recover histogram rectangles from the source PDF")

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(7.2, 5.76), dpi=300)
    draw_bars(ax, main_bars)
    ax.set_xlim(-0.02, 2.96)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Cost", fontsize=20)
    ax.set_ylabel("Probability", fontsize=20)
    ax.tick_params(labelsize=14)
    ax.grid(True, color="#b0b0b0", linewidth=1.1)
    ax.legend(loc="upper right", fontsize=13, framealpha=0.85)

    inset = ax.inset_axes([0.025, 0.62, 0.23, 0.36])
    draw_bars(inset, inset_bars)
    inset.set_xlim(0.95, 1.067)
    inset.set_ylim(0, 1.0)
    inset.set_xticks([0.95, 1.00, 1.05])
    inset.set_yticks([0, 0.5, 1.0])
    inset.set_yticklabels([])
    inset.tick_params(labelsize=10)
    inset.set_title("Zoom near Cost = 1", fontsize=9, pad=2)

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.15, top=0.96)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF)
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
