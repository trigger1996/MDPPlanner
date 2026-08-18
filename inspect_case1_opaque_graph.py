"""Run Case Study 1 synthesis and print its opaque graph and observer MEC."""
from collections import deque
from pathlib import Path
from pprint import pformat

import matplotlib.pyplot as plt
import networkx as nx
from Map.example_20250506_grid_single_agent import (
    construct_single_agent_mdp,
    observation_func_0506,
    control_observable_dict,
)
from MDP_TG.dra import Dra
from User.dra3 import product_mdp3, project_sync_mec_3_2_observer_mec_3


def obtain_all_aps_from_team_mdp(mdp):
    aps = set()
    for state in mdp.nodes:
        for labels in mdp.nodes[state]["label"]:
            aps.update(labels)
    aps.discard("")
    return sorted(aps)


def compact_state(state):
    def pstate(x):
        q, labels, dra = x
        return f"({q},{'|'.join(sorted(labels)) or '-'},s{dra})"

    actual, gamma, other = state
    return (
        f"[{pstate(actual)}; "
        f"G={{{','.join(pstate(x) for x in sorted(gamma, key=str))}}}; "
        f"R={{{','.join(pstate(x) for x in sorted(other, key=str))}}}]"
    )


def edge_actions(data):
    actions = set()
    for attrs in data.values():
        actions.update(attrs.get("prop", {}).keys())
    return ",".join(sorted(map(str, actions))) or "-"


def readable_actions(data):
    actions = set()
    for attrs in data.values():
        for action in attrs.get("prop", {}):
            if isinstance(action, tuple) and len(action) == 1:
                action = action[0]
            actions.add(str(action))
    return ", ".join(sorted(actions)) or "-"


def export_remake_transition_check(graph, output_path):
    """Export all internal and external outgoing edges for states drawn in the remake PDF."""
    specifications = [
        ("R01", "13", 2), ("R02", "12", 3), ("R03", "17", 1),
        ("R04", "12", 1), ("R05", "18", 1), ("R06", "17", 2),
        ("R07", "19", 0), ("R08", "18", 3), ("R09", "16", 1),
    ]
    selected = {}
    for state_id, q, dra in specifications:
        matches = sorted((node for node in graph if node[0][0] == q and node[0][2] == dra), key=str)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one state for {state_id}=(q{q},s{dra}), found {len(matches)}")
        selected[state_id] = matches[0]
    id_by_node = {node: state_id for state_id, node in selected.items()}

    internal_edges, external_edges = [], []
    for source_id, source in selected.items():
        for target in sorted(graph.successors(source), key=str):
            action = readable_actions(graph[source][target])
            if target in id_by_node:
                internal_edges.append((source_id, action, id_by_node[target]))
            else:
                external_edges.append((source_id, action, compact_state(target)))

    lines = [
        "REMAKE PDF TRANSITION CHECK",
        "===========================",
        "",
        "State IDs (full joint information states)",
        "-----------------------------------------",
    ]
    lines.extend(f"{state_id} = {compact_state(node)}" for state_id, node in selected.items())
    lines.extend(["", "Edges whose source and target are both drawn", "-------------------------------------------"])
    lines.extend(f"{source} --{action}--> {target}" for source, action, target in sorted(internal_edges))
    lines.extend(["", "Outgoing edges to states not drawn in the PDF", "----------------------------------------------"])
    lines.extend(f"{source} --{action}--> ... {target}" for source, action, target in sorted(external_edges))
    lines.extend([
        "", "Typo / consistency check", "------------------------",
        "1. R05=(q18,empty,s1): the PDF shows (q13,{gather},s0) in X_gamma^obs; code output is (q13,{gather},s1).",
        "2. R09=(q16,{investigate},s1): X_gamma^obs in the PDF omits (q21,empty,s1).",
        "3. The drawn arrow R03=(q17,empty,s1) -> R04=(q12,{gather},s1) has no action label; see the internal-edge list above.",
        "4. The filename says '13states', but the remake PDF contains 9 state boxes.",
    ])
    output_path = Path(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Remake transition check saved to: {output_path}")


def draw_local_subgraph(graph, observer_states, initial_states, output_path, transition_path, count=13):
    """Draw a deterministic connected neighborhood around the opaque q12 state."""
    observer_states = set(observer_states)
    centers = [n for n in graph if n[0][0] == "12" and n[0][2] == 1 and n in observer_states]
    if not centers:
        raise RuntimeError("Could not find the q12/s1 observer state")
    center = sorted(centers, key=str)[0]
    undirected = graph.to_undirected()
    selected = [center]
    initial_states = [state for state in initial_states if state in graph]
    required = [("12", 3), ("13", 2), ("18", 1), ("18", 3)]
    for q_required, dra_required in required:
        matches = sorted(
            (n for n in graph if n[0][0] == q_required and n[0][2] == dra_required),
            key=lambda n: (n not in observer_states, str(n)),
        )
        if not matches:
            continue
        path = nx.shortest_path(undirected, center, matches[0])
        for node in path:
            if node not in selected:
                selected.append(node)
    for initial_state in sorted(initial_states, key=str):
        for node in nx.shortest_path(undirected, center, initial_state):
            if node not in selected:
                selected.append(node)
    seen = set(selected)
    queue = deque(selected)
    while queue and len(selected) < count:
        node = queue.popleft()
        neighbors = sorted(
            undirected.neighbors(node),
            key=lambda n: (n not in observer_states, n[0][0] not in {"2", "7", "12", "13", "17"}, str(n)),
        )
        for neighbor in neighbors:
            if neighbor not in seen:
                seen.add(neighbor)
                selected.append(neighbor)
                queue.append(neighbor)
                if len(selected) >= count:
                    break
    if len(selected) < count:
        raise RuntimeError(f"Only found {len(selected)} connected states")

    selected = selected[:count]
    sub = graph.subgraph(selected).copy()
    pos = nx.spring_layout(sub, seed=17, k=2.45, iterations=900, weight=None, scale=1.75)
    fig = plt.figure(figsize=(40, 26))
    ax = fig.add_axes([0.02, 0.065, 0.75, 0.87])
    ax.set_title(
        "Case Study 1: synchronized product-state pairs in the opaque subgraph",
        fontsize=27, weight="bold", pad=34,
    )
    node_colors = ["#BFE5D8" if n in observer_states else "#FAD9A8" for n in sub]
    node_edges = ["#7A3DB8" if n in initial_states else ("#168A88" if n in observer_states else "#D98219") for n in sub]
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=25500, node_shape="s", node_color=node_colors,
                           edgecolors=node_edges, linewidths=3.6)
    nx.draw_networkx_edges(sub, pos, ax=ax, edge_color="#526572", width=2.0,
                           arrows=True, arrowsize=27, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.14",
                           min_source_margin=86, min_target_margin=86)
    labels = {}
    for i, node in enumerate(selected, 1):
        q, labels_t, dra = node[0]
        ap = "/".join(sorted(x for x in labels_t if x)) or "-"
        obs = observation_func_0506(q)
        actual = f"(q{q},{ap},s{dra})"
        region = "observer MEC" if node in observer_states else "opaque prefix only"
        initial_mark = "  INITIAL" if node in initial_states else ""
        info_class = "X_op" if node[2] else "X_bar_op"

        def format_component(state):
            sq, slabels, sdra = state
            sap = "/".join(sorted(x for x in slabels if x)) or "-"
            return f"(q{sq},{sap},s{sdra})"

        def wrap_set(name, states, per_line=2):
            values = [format_component(state) for state in sorted(states, key=str)]
            if not values:
                return [f"{name} = empty"]
            lines = []
            for start in range(0, len(values), per_line):
                chunk = ", ".join(values[start:start + per_line])
                prefix = f"{name} = {{" if start == 0 else "       "
                suffix = "}" if start + per_line >= len(values) else ""
                lines.append(prefix + chunk + suffix)
            return lines

        labels[node] = {
            "meta": f"x{i:02d} [{region}; {info_class}]{initial_mark}   O = z{obs}",
            "actual": f"x_pi = {actual}",
            "gamma": wrap_set("X_gamma^obs", node[1]),
            "third": wrap_set("X_3", node[2]),
        }
    # Print the joint information state exactly as (x_pi, X_gamma^obs, X_3).
    for node, parts in labels.items():
        x, y = pos[node]
        colored_lines = (
            [(parts["meta"], "#263746", 8.5, "bold"),
             ("joint state: (", "#263746", 8.0, "normal"),
             (parts["actual"], "#C7472F", 8.4, "bold")]
            + [(line, "#2E75B6", 7.5, "semibold") for line in parts["gamma"]]
            + [(line, "#7A3DB8", 7.5, "semibold") for line in parts["third"]]
            + [(")", "#263746", 8.0, "normal")]
        )
        step = 0.029
        start_y = y + step * (len(colored_lines) - 1) / 2
        for line_index, (line, color, fontsize, weight) in enumerate(colored_lines):
            ax.text(x, start_y - line_index * step, line, ha="center", va="center",
                    fontsize=fontsize, color=color, weight=weight)
    edge_labels = {}
    for u, v in sub.edges():
        actions = edge_actions(sub[u][v]).replace("('", "").replace("',)", "")
        edge_labels[(u, v)] = actions
    nx.draw_networkx_edge_labels(sub, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=11, rotate=False, label_pos=0.52,
                                 bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#D8DEE2", "alpha": 0.94})
    # Show real outgoing transitions whose target is outside this 10-state view.
    center_xy = pos[center]
    for node in selected:
        outside_actions = set()
        for target in graph.successors(node):
            if target not in sub:
                outside_actions.update(edge_actions(graph[node][target]).split(","))
        if not outside_actions:
            continue
        x, y = pos[node]
        dx, dy = x - center_xy[0], y - center_xy[1]
        norm = max((dx * dx + dy * dy) ** 0.5, 0.2)
        dx, dy = 0.38 * dx / norm, 0.38 * dy / norm
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x + 0.19 * dx, y + 0.19 * dy),
                    arrowprops={"arrowstyle": "-|>", "mutation_scale": 20, "lw": 2.0, "color": "#667984"})
        actions = ",".join(sorted(a for a in outside_actions if a and a != "-"))
        ax.text(x + 1.20 * dx, y + 1.20 * dy, f"{actions}  ...", fontsize=11,
                color="#586873", weight="bold", ha="center", va="center")
    ax.set_axis_off()

    table_ax = fig.add_axes([0.79, 0.075, 0.195, 0.85])
    table_ax.set_axis_off()
    table_ax.text(0, 1.02, "Information-state index", fontsize=17, weight="bold", color="#263746")
    rows = []
    for i, node in enumerate(selected, 1):
        q, labels_t, dra = node[0]
        ap = "/".join(sorted(x for x in labels_t if x)) or "-"
        region = "MEC" if node in observer_states else "prefix"
        rows.append([f"x{i:02d}", f"(q{q},s{dra}) {ap}", f"z{observation_func_0506(q)}", str(len(node[1])), str(len(node[2])), region])
    table = table_ax.table(cellText=rows, colLabels=["ID", "$x_\\pi$", "$\\mathcal{O}$", "$|G|$", "$|R|$", "region"],
                           cellLoc="center", colLoc="center", loc="upper left",
                           colWidths=[0.11, 0.34, 0.12, 0.10, 0.10, 0.19])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 2.05)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD3D8")
        if row == 0:
            cell.set_facecolor("#263746")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif rows[row - 1][-1] == "MEC":
            cell.set_facecolor("#E3F3ED")
        else:
            cell.set_facecolor("#FFF0D9")
    fig.text(0.03, 0.022,
             "Each box prints the joint information state exactly as (x_pi, X_gamma^obs, X_3): red = x_pi, "
             "blue = X_gamma^obs, purple = X_3. Green: observer MEC; orange: opaque-prefix only; purple border: initial state. "
             "All observations, states, and action-labeled edges are extracted from the implementation.",
             fontsize=13.5, color="#263746")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Machine-checkable incoming/outgoing transition dictionary.
    id_by_node = {node: f"x{i:02d}" for i, node in enumerate(selected, 1)}
    transition_dict = {}
    for node in selected:
        incoming, outgoing = [], []
        for source in graph.predecessors(node):
            incoming.append({
                "source": id_by_node.get(source, "..."),
                "source_state": compact_state(source),
                "action": edge_actions(graph[source][node]),
            })
        for target in graph.successors(node):
            outgoing.append({
                "target": id_by_node.get(target, "..."),
                "target_state": compact_state(target),
                "action": edge_actions(graph[node][target]),
            })
        transition_dict[id_by_node[node]] = {
            "state": compact_state(node),
            "observation": f"z{observation_func_0506(node[0][0])}",
            "region": "observer_mec" if node in observer_states else "opaque_prefix_only",
            "initial": node in initial_states,
            "incoming": sorted(incoming, key=lambda x: (x["source"], x["action"], x["source_state"])),
            "outgoing": sorted(outgoing, key=lambda x: (x["target"], x["action"], x["target_state"])),
        }
    transition_path = Path(transition_path)
    transition_path.write_text(
        '"""Generated from the Case Study 1 opaque full graph.\n'
        'Each entry records incoming edges, the information state, and outgoing edges.\n"""\n\n'
        "STATE_TRANSITIONS = " + pformat(transition_dict, width=140, sort_dicts=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n{count}-state figure saved to: {output_path}")
    print(f"Transition dictionary saved to: {transition_path}")
    print("Selected state IDs:")
    for i, node in enumerate(selected, 1):
        print(f"x{i:02d} {'MEC' if node in observer_states else 'PREFIX'} {compact_state(node)}")


def main():
    mdp, _, _, _ = construct_single_agent_mdp(is_visualize=False)
    pi, gamma = "gather", "recharge"
    # Native prefix syntax accepted by the bundled ltl2dstar/ltl2ba tools.
    task_pi = "& G F i gather U ! gather drop G F gather"
    task_gamma = "& G F i gather U ! gather drop G F recharge"
    prod_pi = product_mdp3(mdp, Dra(task_pi))
    prod_pi.compute_S_f()
    prod_gamma = product_mdp3(mdp, Dra(task_gamma))
    prod_gamma.compute_S_f()

    candidates = []
    for sf_pi in prod_pi.Sf:
        for mec_pi in sf_pi:
            for sf_gamma in prod_gamma.Sf:
                for mec_gamma in sf_gamma:
                    ok = prod_pi.re_synthesize_sync_amec(
                        pi, gamma, mec_pi, mec_gamma, prod_gamma,
                        observation_func=observation_func_0506,
                        ctrl_obs_dict=control_observable_dict,
                    )
                    if not ok or not prod_pi.sync_amec_set:
                        continue
                    sync_graph = prod_pi.sync_amec_set[prod_pi.current_sync_amec_index]
                    sync_mec = prod_pi.project_sync_amec_back_to_mec_pi(sync_graph, mec_pi)
                    if not sync_mec[1]:
                        continue
                    prefix, initial = prod_pi.construct_opaque_subgraph_2_amec(
                        prod_gamma, sync_mec, sync_graph, mec_pi, mec_gamma,
                        pi, gamma, observation_func_0506, control_observable_dict,
                    )
                    observer = project_sync_mec_3_2_observer_mec_3(prefix, sync_mec)
                    full = prod_pi.construct_fullgraph_4_amec(
                        prefix, prod_gamma, sync_graph, mec_pi, mec_gamma,
                        pi, gamma, observation_func_0506, control_observable_dict,
                    )
                    candidates.append((prefix, full, observer, initial))

    if not candidates:
        raise RuntimeError("No opaque-subgraph candidate was constructed")
    print("\n=== CANDIDATES ===")
    for i, (prefix, full, observer, _) in enumerate(candidates, 1):
        print(f"C{i}: prefix={prefix.number_of_nodes()}/{prefix.number_of_edges()}, "
              f"full={full.number_of_nodes()}/{full.number_of_edges()}, observer MEC={len(observer[0])}")
    opaque_prefix, opaque_full, observer, initial_states = min(
        candidates,
        key=lambda c: (abs(c[1].number_of_nodes() - 42), abs(len(c[2][0]) - 41)),
    )
    observer_states, observer_ip, _ = observer
    observer_states = set(observer_states)

    print("\n=== GRAPH SUMMARY ===")
    print(f"opaque prefix subgraph: {opaque_prefix.number_of_nodes()} nodes, {opaque_prefix.number_of_edges()} edges")
    print(f"opaque full graph:      {opaque_full.number_of_nodes()} nodes, {opaque_full.number_of_edges()} edges")
    print(f"observer MEC:           {len(observer_states)} nodes")
    print(f"observer accepting set: {len(observer_ip)} nodes")

    print("\n=== OBSERVER MEC NODES ===")
    for i, node in enumerate(sorted(observer_states, key=str), 1):
        print(f"M{i:02d} {compact_state(node)}")

    print("\n=== OPAQUE FULL GRAPH NODES ===")
    for i, node in enumerate(sorted(opaque_full.nodes, key=str), 1):
        mark = "MEC" if node in observer_states else "PREFIX"
        print(f"B{i:02d} [{mark}] {compact_state(node)}")

    print("\n=== OPAQUE FULL GRAPH EDGES ===")
    for u, v in sorted(opaque_full.edges(), key=lambda e: (str(e[0]), str(e[1]))):
        print(f"{compact_state(u)} --{edge_actions(opaque_full[u][v])}--> {compact_state(v)}")

    draw_local_subgraph(
        opaque_full,
        observer_states,
        initial_states,
        "[paper]MDP-LTL-Opacity/pics/opaque_subgraph_case1_local_13states.pdf",
        "[paper]MDP-LTL-Opacity/pics/opaque_subgraph_case1_local_13states_transitions.py",
        count=13,
    )
    export_remake_transition_check(
        opaque_full,
        "[paper]MDP-LTL-Opacity/pics/opaque_subgraph_case1_local_13states_remake_transitions.txt",
    )


if __name__ == "__main__":
    main()
