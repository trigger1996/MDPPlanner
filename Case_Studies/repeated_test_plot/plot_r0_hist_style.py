#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate repeated-test histograms with the paper r0 visual style."""

import argparse
import contextlib
import io
import os
import random
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, "-m",
             "Case_Studies.repeated_test_plot.plot_r0_hist_style"] + sys.argv[1:])

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Plot" / "0506_Cost_All_single_graph.pdf"
BATCH_OUTPUT_DIR = ROOT / "Plot" / "r0_hist_rebuild"

PARAM_GROUPS = [
    (0.125, 5, 0.05, 5),
    (0.125, 5, 0.1, 5),
    (0.25, 7, 0.25, 7),
    (0.5, 15, 0.5, 15),
]

SERIES_STYLE = [
    ("pi_opaque", "#C99E8C", r"$\pi$ in opaque run", 0.48),
    ("gamma_opaque", "#465E65", r"$\gamma$ in opaque run", 0.50),
    ("pi_nonopaque", "#57C3C2", r"$\pi$ in non-opaque run", 0.48),
    ("gamma_nonopaque", "#FE4567", r"$\gamma$ in non-opaque run", 0.50),
]


def _cost_values(cost_list, is_average=True):
    return [item[0] / item[2] if is_average else item[0] for item in sorted(cost_list, key=lambda x: x[0])]


def _flatten_cost_groups(cost_groups):
    data = {key: [] for key, _, _, _ in SERIES_STYLE}
    repeated = []
    for group in cost_groups:
        (pi_opaque, gamma_opaque), (pi_nonopaque, gamma_nonopaque) = group
        repeated.append({
            "pi_opaque": _cost_values(pi_opaque),
            "gamma_opaque": _cost_values(gamma_opaque),
            "pi_nonopaque": _cost_values(pi_nonopaque),
            "gamma_nonopaque": _cost_values(gamma_nonopaque),
        })
        data["pi_opaque"].extend(repeated[-1]["pi_opaque"])
        data["gamma_opaque"].extend(repeated[-1]["gamma_opaque"])
        data["pi_nonopaque"].extend(repeated[-1]["pi_nonopaque"])
        data["gamma_nonopaque"].extend(repeated[-1]["gamma_nonopaque"])
    return data, repeated


def _draw_hist_series(ax, data, keys, bins=25, legend=True):
    """Draw per-bin probabilities, matching the 0506 trial figures.

    KDE is intentionally disabled: seaborn scales a KDE as probability density,
    so overlaying it on ``stat="probability"`` can produce values above one and
    gives the shared y-axis two incompatible meanings.
    """
    for key, color, label, alpha in SERIES_STYLE:
        if key not in keys:
            continue
        sns.histplot(
            data[key],
            bins=bins,
            kde=False,
            stat="probability",
            color=color,
            edgecolor=color,
            linewidth=1.0,
            alpha=alpha,
            label=label if legend else None,
            ax=ax,
        )


def _style_main_axis(ax, show_ylabel=True):
    ax.set_xlim(-0.02, 2.96)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Cost", fontsize=30)
    ax.set_ylabel("Probability" if show_ylabel else "", fontsize=30)
    ax.tick_params(axis="both", labelsize=22, width=1.4, length=5)
    ax.grid(True, color="#b0b0b0", linewidth=1.35)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_r0_single_graph(data, output_path, keys=None, with_inset=True):
    keys = keys or [key for key, _, _, _ in SERIES_STYLE]
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(7, 5.6), dpi=150)
    _draw_hist_series(ax, data, keys)
    _style_main_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        fontsize=20,
        frameon=True,
        framealpha=0.84,
        handlelength=1.8,
        borderpad=0.45,
        labelspacing=0.55,
    )

    if with_inset:
        inset = ax.inset_axes([0.01, 0.635, 0.33, 0.355])
        _draw_hist_series(inset, data, keys, legend=False)
        inset.set_xlim(0.95, 1.067)
        inset.set_ylim(0, 1.0)
        inset.set_xlabel("")
        inset.set_ylabel("")
        inset.set_xticks([0.95, 1.00, 1.05])
        inset.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        inset.set_yticklabels([])
        inset.tick_params(axis="both", labelsize=12, width=1.0, length=4)
        inset.grid(False)
        for spine in inset.spines.values():
            spine.set_linewidth(0.9)
        mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="#4c4c4c", lw=2.4)

    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.16, top=0.97)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_r0_repeated_graph(repeated_data, output_path):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })
    fig, axs = plt.subplots(1, len(repeated_data), figsize=(7 * len(repeated_data), 5), squeeze=False, dpi=150)
    handles, labels = None, None
    keys = [key for key, _, _, _ in SERIES_STYLE]
    for idx, data in enumerate(repeated_data):
        ax = axs[0][idx]
        _draw_hist_series(ax, data, keys)
        ax.set_xlim(-0.02, 2.96)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="both", labelsize=16, width=1.2, length=4)
        ax.grid(True, color="#b0b0b0", linewidth=1.2)
        ax.set_axisbelow(True)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if idx != 0:
            ax.tick_params(left=False, labelleft=False)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend().remove()
    fig.legend(handles, labels, loc="center right", fontsize=19, frameon=True, framealpha=0.84)
    fig.text(0.5, 0.04, "Cost", ha="center", fontsize=24)
    fig.text(0.035, 0.5, "Probability", va="center", rotation="vertical", fontsize=24)
    fig.tight_layout(rect=[0.06, 0.08, 0.86, 1])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def collect_0506_cost_groups(seed=1, quiet=True):
    from Case_Studies.repeated_test_plot import repeated_example_20250506_single_agent_main as ex
    from Case_Studies.example_20250506_single_agent_main import (
        execute_example_4_product_mdp3,
        execute_example_in_origin_product_mdp,
        obtain_all_aps_from_team_mdp,
    )
    from Map.example_20250506_grid_single_agent import construct_single_agent_mdp
    from MDP_TG.dra import Dra
    from User.team_dra3 import product_team_mdp3

    ex.ltl_formula = "GF (gather -> (!gather U drop))"
    ex.opt_prop = "gather"
    ltl_formula_converted = "G F | ! gather U ! gather drop"
    results = []
    # The trial generator retains the final MDP/initial state after synthesis
    # and uses those values for every Monte Carlo group. Mirror that behavior
    # exactly so a rebuild made with seed 1 is data-consistent with seed_01.
    trial_mdp = trial_initial_node = trial_initial_label = None
    for param_group in PARAM_GROUPS:
        mdp, initial_node, initial_label, _ = construct_single_agent_mdp(is_visualize=True)
        ap_list = obtain_all_aps_from_team_mdp(mdp)
        best_plan_opq, best_plan_non_opq, prod_dra_pi = ex.run_one_param_group(
            mdp, ltl_formula_converted, ap_list, param_group, max_attempt=1)
        results.append((mdp, initial_node, initial_label, best_plan_opq, best_plan_non_opq, prod_dra_pi))
        trial_mdp, trial_initial_node, trial_initial_label = mdp, initial_node, initial_label

    random.seed(seed)
    np.random.seed(seed)
    cost_groups = []
    for _mdp, _initial_node, _initial_label, best_plan_opq, best_plan_non_opq, prod_dra_pi in results:
        dra = Dra(ltl_formula_converted)
        cost_list_pi, cost_list_gamma, _ = execute_example_4_product_mdp3(
            500, 150, prod_dra_pi, best_plan_opq,
            [trial_initial_node], [trial_initial_label], ex.opt_prop, best_plan_opq[3][0], attr="Opaque")
        prod_dra = product_team_mdp3(trial_mdp, dra)
        cost_list_pi_p, cost_list_gamma_p = execute_example_in_origin_product_mdp(
            500, 150, prod_dra, best_plan_non_opq,
            [trial_initial_node], [trial_initial_label], ex.opt_prop, best_plan_opq[3][0], attr="Non-Opaque")
        cost_groups.append([[cost_list_pi, cost_list_gamma], [cost_list_pi_p, cost_list_gamma_p]])
    return cost_groups


def collect_0426_cost_groups(seed=1):
    from Case_Studies.repeated_test_plot import repeated_example_20250426_main as ex
    from Case_Studies.example_20250426_team_mdp_main import (
        execute_example_4_product_mdp3,
        execute_example_in_origin_product_mdp,
        obtain_all_aps_from_team_mdp,
    )
    from Map.example_20250426_team_mdp import construct_team_mdp
    from MDP_TG.dra import Dra
    from User.team_dra3 import product_team_mdp3

    ex.ltl_formula = "GF (gather -> (!gather U drop))"
    ex.opt_prop = "gather"
    ltl_formula_converted = "G F | ! gather U ! gather drop"
    team_mdp, initial_node, initial_label = construct_team_mdp()
    ap_list = obtain_all_aps_from_team_mdp(team_mdp)
    results = []
    for param_group in PARAM_GROUPS:
        best_plan_opq, best_plan_non_opq, prod_dra_pi = ex.run_one_param_group(
            team_mdp, ltl_formula_converted, ap_list, param_group, max_attempt=1)
        results.append((best_plan_opq, best_plan_non_opq, prod_dra_pi))

    random.seed(seed)
    np.random.seed(seed)
    cost_groups = []
    for best_plan_opq, best_plan_non_opq, prod_dra_pi in results:
        dra = Dra(ltl_formula_converted)
        cost_list_pi, cost_list_gamma, _ = execute_example_4_product_mdp3(
            500, 150, prod_dra_pi, best_plan_opq,
            [initial_node], [initial_label], ex.opt_prop, best_plan_opq[3][0], attr="Opaque")
        prod_dra = product_team_mdp3(team_mdp, dra)
        cost_list_pi_p, cost_list_gamma_p = execute_example_in_origin_product_mdp(
            500, 150, prod_dra, best_plan_non_opq,
            [initial_node], [initial_label], ex.opt_prop, best_plan_opq[3][0], attr="Non-Opaque")
        cost_groups.append([[cost_list_pi, cost_list_gamma], [cost_list_pi_p, cost_list_gamma_p]])
    return cost_groups


def build_outputs_for_prefix(cost_groups, prefix, output_dir):
    data, repeated = _flatten_cost_groups(cost_groups)
    plot_r0_single_graph(data, output_dir / f"{prefix}_Cost_All.pdf")
    plot_r0_single_graph(data, output_dir / f"{prefix}_Cost_Opaque.pdf",
                         keys=["pi_opaque", "gamma_opaque"], with_inset=False)
    plot_r0_single_graph(data, output_dir / f"{prefix}_Cost_NonOpaque.pdf",
                         keys=["pi_nonopaque", "gamma_nonopaque"], with_inset=False)
    plot_r0_repeated_graph(repeated, output_dir / ("0506_Single_Repeated.pdf" if prefix == "0506" else "0426_Team_Repeated.pdf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output path for the default 0506 single-graph PDF")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--all-reference-hist", action="store_true",
                        help="Also rebuild r0/hist-style 0426 and 0506 PDFs under Plot/r0_hist_rebuild")
    args = parser.parse_args()

    quiet_context = contextlib.redirect_stdout(io.StringIO()) if args.quiet else contextlib.nullcontext()
    with quiet_context:
        cost_groups_0506 = collect_0506_cost_groups(seed=args.seed)
    data_0506, repeated_0506 = _flatten_cost_groups(cost_groups_0506)
    plot_r0_single_graph(data_0506, args.output)
    print(f"Saved {Path(args.output).resolve()}")

    if args.all_reference_hist:
        build_outputs_for_prefix(cost_groups_0506, "0506", BATCH_OUTPUT_DIR)
        quiet_context = contextlib.redirect_stdout(io.StringIO()) if args.quiet else contextlib.nullcontext()
        with quiet_context:
            cost_groups_0426 = collect_0426_cost_groups(seed=args.seed)
        build_outputs_for_prefix(cost_groups_0426, "0426", BATCH_OUTPUT_DIR)
        print(f"Saved batch histograms under {BATCH_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
