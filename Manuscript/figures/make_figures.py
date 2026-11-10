"""Generate the figures for manuscript.tex.

Run from any directory:  python make_figures.py
Inputs : ../../Data/table_A1_arrival_rates.csv (Table A1 of Wang et al.'s supplement; medium = 0.5x, large = 1.0x)
         sensitivity_planning.csv, sensitivity_scheduling.csv (HGASA sensitivity results)
Outputs: PDF figures next to this script, plus graphical-abstract.pdf/.tif for the C&IE submission.
Regenerate the sensitivity figures after updating the two CSV files.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "Data"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#52514e",
    "axes.linewidth": 0.6,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "grid.color": "#d9d8d4",
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
})

# Categorical palette, fixed order (validated: all-pairs CVD/normal-vision checks pass).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, FILL, EDGE = "#0b0b0b", "#52514e", "#f3f2ef", "#52514e"


# --------------------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, text, fc=FILL, ec=EDGE, lw=0.8, fs=8, bold=False, dashed=False, round_=True):
    style = "round,pad=0.0,rounding_size=0.06" if round_ else "square,pad=0.0"
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw,
                                ls="--" if dashed else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def diamond(ax, x, y, w, h, text, fs=7.5):
    ax.add_patch(Polygon([(x - w / 2, y), (x, y + h / 2), (x + w / 2, y), (x, y - h / 2)],
                         closed=True, fc="white", ec=EDGE, lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.15)


def arrow(ax, p, q, text=None, tpos=0.5, toff=(0, 0.12), fs=7, rad=0.0):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK2, shrinkA=0, shrinkB=0,
                                mutation_scale=8, connectionstyle=f"arc3,rad={rad}"))
    if text:
        mx = p[0] + (q[0] - p[0]) * tpos + toff[0]
        my = p[1] + (q[1] - p[1]) * tpos + toff[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=INK2)


def polyline(ax, pts):
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=INK2, lw=0.8)
    arrow(ax, pts[-2], pts[-1])


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


def fig_framework():
    fig, ax = canvas(7.0, 4.2, (0, 17.5), (0, 10.5))
    fs = 6.5
    ax.text(0.1, 8.0, "Phase 1\nStaff\nplanning\n(daily)", fontsize=7.5, fontweight="bold",
            va="center", ha="left", color=INK)
    ax.text(0.1, 2.7, "Phase 2\nStaff\nscheduling\n(weekly)", fontsize=7.5,
            fontweight="bold", va="center", ha="left", color=INK)
    ax.plot([0.1, 17.4], [5.35, 5.35], color="#bdbbb5", lw=0.8, ls="--")
    # Phase 1 inputs
    box(ax, 4.6, 9.0, 3.8, 1.5, "Deterministic inputs\nrooms, shift structure,\nweights, waiting targets", fs=fs)
    box(ax, 4.6, 6.8, 3.8, 1.5, "Stochastic inputs\nhourly arrival rates,\nservice times, routing", fs=fs)
    # Phase 1 simulation-based optimization
    ax.add_patch(FancyBboxPatch((7.1, 5.8), 6.2, 4.4, boxstyle="round,pad=0.0,rounding_size=0.1",
                                fc="white", ec=EDGE, lw=0.9))
    ax.text(10.2, 9.75, "Simulation-based optimization (SO-HGASA)", ha="center", va="center", fontsize=fs,
            fontweight="bold", color=INK)
    box(ax, 10.2, 8.85, 5.4, 0.85, "HGASA search over staffing levels", fs=fs)
    box(ax, 10.2, 6.55, 5.4, 0.85, "Discrete-event simulation of the ED", fs=fs)
    arrow(ax, (8.4, 8.42), (8.4, 6.98), "candidate\nplan", toff=(0.85, 0), fs=6)
    arrow(ax, (12.0, 6.98), (12.0, 8.42), r"$\widehat{\mathbb{E}}[WT_i]$," + "\n" + r"$\widehat{\mathbb{E}}[DT_i]$",
          toff=(-0.95, 0), fs=6)
    arrow(ax, (6.5, 9.0), (7.5, 8.85))
    arrow(ax, (6.5, 6.8), (7.5, 6.55))
    box(ax, 15.5, 8.0, 3.4, 2.8, "Output\nnurses and physicians\nper full-time and\npart-time shift,\nfor each day", fs=fs)
    arrow(ax, (13.3, 8.0), (13.8, 8.0))
    # Phase 2
    box(ax, 4.6, 4.0, 3.8, 1.4, "Rostering rules\n" + r"$W_1, W_2, E_1, E_2$, rest rules", fs=fs)
    box(ax, 4.6, 1.7, 3.8, 1.4, "Shift demands\nfrom Phase 1", fs=fs)
    ax.add_patch(FancyBboxPatch((7.1, 0.5), 6.2, 4.4, boxstyle="round,pad=0.0,rounding_size=0.1",
                                fc="white", ec=EDGE, lw=0.9))
    ax.text(10.2, 4.45, "Roster construction", ha="center", va="center", fontsize=fs, fontweight="bold",
            color=INK)
    box(ax, 10.2, 3.45, 5.4, 0.85, "Exact integer program (MILP)", fs=fs, dashed=True)
    box(ax, 10.2, 1.6, 5.4, 0.85, "HGASA heuristic with staff elimination", fs=fs, dashed=True)
    arrow(ax, (6.5, 4.0), (7.5, 3.45))
    arrow(ax, (6.5, 1.7), (7.5, 1.6))
    box(ax, 15.5, 2.7, 3.4, 2.2, "Output\nweekly roster and\nnumber of staff\nrequired", fs=fs)
    arrow(ax, (13.3, 2.7), (13.8, 2.7))
    polyline(ax, [(15.5, 6.6), (15.5, 5.6), (2.35, 5.6), (2.35, 1.7), (2.7, 1.7)])
    fig.savefig(HERE / "framework.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_patient_flow():
    fig, ax = canvas(7.0, 3.1, (0, 17.5), (0, 7.75))
    fs = 7
    box(ax, 1.15, 6.4, 2.1, 1.0, "Arrival\n(hourly NHPP)", fs=fs)
    box(ax, 4.15, 6.4, 2.9, 1.0, "Triage*\n(one nurse, FCFS)", fs=fs)
    box(ax, 7.95, 6.4, 4.0, 1.0, "First consultation\n(one physician + one nurse)", fs=fs)
    diamond(ax, 12.0, 6.4, 3.0, 1.6, "Examination\nrequired?", fs=fs)
    box(ax, 16.1, 6.4, 2.4, 1.0, "Discharge", fs=fs)
    arrow(ax, (2.35, 6.4), (2.7, 6.4)); arrow(ax, (5.6, 6.4), (5.95, 6.4)); arrow(ax, (9.95, 6.4), (10.5, 6.4))
    arrow(ax, (13.5, 6.4), (14.9, 6.4), "no (40%)", toff=(0.0, 0.3), fs=6.5)
    tests = ["Laboratory test", "Ultrasound", "X-ray", "CT scan"]
    xs = [3.6, 6.6, 9.6, 12.6]
    ax.add_patch(FancyBboxPatch((1.9, 2.95), 12.4, 1.35, boxstyle="round,pad=0.0,rounding_size=0.1",
                                fc="white", ec="#bdbbb5", lw=0.8, ls="--"))
    for x, t in zip(xs, tests):
        box(ax, x, 3.62, 2.5, 0.8, t, fc=FILL, fs=fs)
    ax.text(7.6, 4.55, "tests in order of shortest current queue; fixed examination and result turnaround times",
            ha="center", fontsize=6.5, color=INK2)
    polyline(ax, [(12.0, 5.6), (12.0, 4.95), (13.6, 4.95), (13.6, 4.3)])
    ax.text(12.15, 5.15, "yes (60%)", fontsize=6.5, color=INK2, ha="left")
    box(ax, 8.1, 1.05, 4.6, 1.0, "Revisit consultation, IP queue\n(one physician + one nurse)", fs=fs)
    polyline(ax, [(8.1, 2.95), (8.1, 1.55)])
    polyline(ax, [(10.4, 1.05), (16.1, 1.05), (16.1, 5.9)])
    ax.text(0.15, 1.6, "Consultation queues: Slack-k\npolicy, Category III first.\n*Nurses: this study's extension.",
            fontsize=6.5, color=INK2, va="center")
    fig.savefig(HERE / "patient-flow.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_hgasa():
    fig, ax = canvas(6.4, 4.2, (0, 12.8), (0, 8.4))
    ax.text(3.2, 8.1, "Genetic algorithm (global search)", ha="center", fontsize=8, fontweight="bold")
    ax.text(9.6, 8.1, "Simulated annealing (local search)", ha="center", fontsize=8, fontweight="bold")
    ga = ["Generate feasible initial population\nof size $N_{pop}$; evaluate fitness",
          "$\\kappa$-way tournament selection\nof two parents",
          "One-point crossover (prob. $p_c$)\nand swap mutation (prob. $p_m$)",
          "Infeasible child $\\Rightarrow$ random chromosome\n(Phase 1) or mutation undone (Phase 2)",
          "Evaluate children; keep the best $N_{pop}$\n(steady-state replacement)"]
    ys = [7.3, 5.95, 4.6, 3.25, 1.9]
    for y, t in zip(ys, ga):
        box(ax, 3.2, y, 5.4, 0.95, t, fs=7.5)
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, (3.2, a - 0.48), (3.2, b + 0.48))
    diamond(ax, 3.2, 0.6, 3.2, 0.9, "$G$ generations done?")
    arrow(ax, (3.2, 1.42), (3.2, 1.05))
    polyline(ax, [(1.6, 0.6), (0.2, 0.6), (0.2, 5.95), (0.5, 5.95)])
    ax.text(0.85, 0.78, "no", fontsize=7, color=INK2)
    sa = ["Start from best GA solution;\n$T \\leftarrow T_0$",
          "Draw a neighbour with a randomly\nchosen structure $\\mathcal{N}_1,\\dots,\\mathcal{N}_4$",
          "Infeasible $\\Rightarrow$ random restart (Phase 1)\nor move redrawn (Phase 2); evaluate fitness",
          "Accept if $\\Delta\\leq 0$, otherwise with\nprobability $\\exp(-\\Delta/T)$; update best",
          "Cool: $T \\leftarrow \\lambda T$"]
    for y, t in zip(ys, sa):
        box(ax, 9.6, y, 5.4, 0.95, t, fs=7.5)
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, (9.6, a - 0.48), (9.6, b + 0.48))
    diamond(ax, 9.6, 0.6, 2.6, 0.9, "$T < T_f$ ?")
    arrow(ax, (9.6, 1.42), (9.6, 1.05))
    polyline(ax, [(10.9, 0.6), (12.6, 0.6), (12.6, 5.95), (12.3, 5.95)])
    ax.text(11.5, 0.78, "no", fontsize=7, color=INK2)
    polyline(ax, [(4.8, 0.6), (6.3, 0.6), (6.3, 7.3), (6.9, 7.3)])
    ax.text(5.3, 0.78, "yes", fontsize=7, color=INK2)
    ax.text(9.6, -0.1, "yes: decode the best solution", fontsize=7, color=INK2, ha="center")
    fig.savefig(HERE / "hgasa-flow.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# --------------------------------------------------------------------------- charts
def fig_arrivals():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=False)
    table = pd.read_csv(DATA / "table_A1_arrival_rates.csv")
    for ax, (factor, title) in zip(axes, [(0.5, "Medium instance"), (1.0, "Large instance")]):
        df = factor * table[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]
        hours = range(24)
        series = [("Monday", df["Monday"], BLUE, "o", "-"),
                  ("Tuesday-Friday (mean)", df[["Tuesday", "Wednesday", "Thursday", "Friday"]].mean(axis=1),
                   ORANGE, "s", "--"),
                  ("Saturday-Sunday (mean)", df[["Saturday", "Sunday"]].mean(axis=1), AQUA, "^", ":")]
        for lab, y, c, m, ls in series:
            ax.plot(hours, y, color=c, lw=1.6, ls=ls, marker=m, ms=3.2, label=lab)
        ax.set_title(f"{title} ({df.sum().mean():.0f} patients/day on average)", loc="left")
        ax.set_xlabel("Hour of day")
        ax.set_xticks(range(0, 25, 3)); ax.set_xlim(-0.5, 23.5)
        ax.grid(axis="y")
    axes[0].set_ylabel("Arrival rate (patients/hour)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(HERE / "arrival-rates.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


LABELS = {"population_size": ("Population size", 50), "generations": ("Number of generations", 200),
          "initial_temperature": ("Initial temperature $T_0$", 100),
          "final_temperature": ("Final temperature $T_f$", 1), "cooling_rate": ("Cooling rate $\\lambda$", None)}


def fig_sensitivity(csv, out, cooling_base):
    df = pd.read_csv(HERE / csv)
    params = list(LABELS)
    fig, axes = plt.subplots(2, 5, figsize=(7.0, 3.3), sharey="row")
    for j, p in enumerate(params):
        d = df[df.parameter == p].sort_values("value")
        base_val = LABELS[p][1] if LABELS[p][1] is not None else cooling_base
        base = d[d.value == base_val].iloc[0]
        avg = 100 * (d.avg_objective - base.avg_objective) / base.avg_objective
        mn = 100 * (d.min_objective - base.min_objective) / base.min_objective
        rt = 100 * (d.runtime_s - base.runtime_s) / base.runtime_s
        x = d.value.astype(float)
        ax = axes[0, j]
        ax.axhline(0, color="#bdbbb5", lw=0.6)
        ax.plot(x, avg, color=BLUE, marker="o", ms=3.2, lw=1.4, label="Mean objective (10 runs)")
        ax.plot(x, mn, color=ORANGE, marker="s", ms=3.2, lw=1.4, ls="--", label="Best objective (10 runs)")
        ax.set_title(LABELS[p][0], fontsize=8)
        ax.grid(axis="y")
        ax = axes[1, j]
        ax.axhline(0, color="#bdbbb5", lw=0.6)
        ax.plot(x, rt, color=AQUA, marker="^", ms=3.2, lw=1.4, label="Runtime")
        ax.grid(axis="y")
        for a in axes[:, j]:
            a.axvline(base_val, color=INK2, lw=0.6, ls=":")
    axes[0, 0].set_ylabel("Objective change (%)")
    axes[1, 0].set_ylabel("Runtime change (%)")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(HERE / out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# --------------------------------------------------------------------------- graphical abstract
def trunc_exp_mean(scale, c=3.5):
    """Mean of an exponential distribution with this scale, truncated at c times the scale."""
    import math
    return scale * (1 - c * math.exp(-c) / (1 - math.exp(-c)))


def fig_graphical_abstract():
    """Graphical abstract for C&IE: 13 cm x 5.2 cm (h:w = 1:2.5), designed to be readable at that size.

    Chart data: Monday arrival rates of the large instance (Table A1) times the expected physician minutes
    per patient (first consultation plus 0.6 x revisit, truncated exponential, Table 4), and the best Monday
    plan of the large instance (Table 8): 7/7/3 full-time physicians plus two part-time physicians at 20:00.
    """
    import numpy as np
    fig = plt.figure(figsize=(13.0 / 2.54, 5.2 / 2.54))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 25); ax.set_ylim(0, 10); ax.axis("off")
    fs, fs_small = 8, 7
    arr = r"$\rightarrow$ "

    def titled_box(x, y, w, h, title, body):
        box(ax, x, y, w, h, "", fs=fs)
        ax.text(x, y + h / 2 - 0.35, title, ha="center", va="top", fontsize=fs, fontweight="bold", color=INK)
        ax.text(x, y + h / 2 - 1.15, body, ha="center", va="top", fontsize=fs, color=INK, linespacing=1.2)

    # left: the two phases
    titled_box(3.85, 7.15, 7.3, 5.3, "Phase 1: SO-HGASA",
               "Simulated ED: every\nconsultation needs a\nphysician and a nurse\n" + arr + "full- and part-time\nstaff per shift")
    titled_box(3.85, 1.75, 7.3, 2.9, "Phase 2: MILP + HGASA", arr + "weekly roster with\nthe fewest staff")
    arrow(ax, (3.85, 4.48), (3.85, 3.22))
    arrow(ax, (7.52, 7.35), (8.55, 7.35))

    # middle: what Phase 1 does with part-time shifts
    table = pd.read_csv(DATA / "table_A1_arrival_rates.csv")
    rate = table["Monday"].to_numpy()
    mu_phy = trunc_exp_mean(9) + 0.6 * trunc_exp_mean(15)
    load = rate * mu_phy / 60
    ft = np.array([3] * 7 + [7] * 8 + [7] * 7 + [3] * 2)
    pt = np.array([0] * 20 + [2, 2, 2] + [0])
    hrs = np.arange(25)
    step = lambda a: np.append(a, a[-1])
    cx = fig.add_axes([0.385, 0.20, 0.27, 0.62])
    cx.fill_between(hrs, 0, step(load), step="post", color="#d9d8d4", lw=0)
    cx.fill_between(hrs, step(ft), step(ft + pt), step="post", color=ORANGE, lw=0, alpha=0.9)
    cx.step(hrs, step(ft), where="post", color=BLUE, lw=1.3)
    cx.set_xlim(0, 24); cx.set_ylim(0, 10)
    cx.set_xticks([0, 6, 12, 18, 24]); cx.set_yticks([0, 3, 6, 9])
    cx.tick_params(labelsize=fs_small, length=2, pad=1.5)
    cx.set_xlabel("Hour of day", fontsize=fs_small, labelpad=1)
    cx.set_ylabel("Physicians", fontsize=fs_small, labelpad=1)
    cx.set_title("Large instance, Monday", fontsize=fs, pad=3)
    cx.text(3.4, 6.1, "full-time", color=BLUE, fontsize=fs_small, ha="center")
    cx.text(10.0, 0.9, "mean workload", color=INK2, fontsize=fs_small, ha="center")
    cx.text(15.4, 9.1, "part-time\n20:00", color=ORANGE, fontsize=fs_small, ha="center", va="center",
            linespacing=1.0)

    # right: key results
    results = ["Optimal plan found\nin all ten enumerated\ncases",
               "Part-time shifts cut\nwaiting by 41%\nat full load",
               "Optimal rosters in\nseconds; HGASA within\n2.5% at 8" + r"$\times$" + " the size"]
    for k, (y, t) in enumerate(zip([8.25, 5.0, 1.75], results)):
        ax.add_patch(FancyBboxPatch((17.15, y - 1.3), 7.6, 2.6, boxstyle="round,pad=0.0,rounding_size=0.12",
                                    fc="white", ec=EDGE, lw=0.8))
        ax.add_patch(FancyBboxPatch((17.15, y - 1.3), 0.35, 2.6, boxstyle="square,pad=0.0",
                                    fc=[BLUE, ORANGE, AQUA][k], lw=0))
        ax.text(21.1, y, t, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.2)
    arrow(ax, (16.35, 5.0), (17.0, 5.0))
    fig.savefig(HERE / "graphical-abstract.pdf", pad_inches=0)
    fig.savefig(HERE / "graphical-abstract.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    fig_graphical_abstract()
    fig_framework()
    fig_patient_flow()
    fig_hgasa()
    fig_arrivals()
    fig_sensitivity("sensitivity_planning.csv", "sensitivity-planning.pdf", cooling_base=0.95)
    fig_sensitivity("sensitivity_scheduling.csv", "sensitivity-scheduling.pdf", cooling_base=0.99)
    print("figures written to", HERE)
