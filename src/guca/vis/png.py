from __future__ import annotations

import math
import time
import colorsys
import zlib
from pathlib import Path
from typing import Any, Iterable, Tuple

__all__ = ["save_png"]


# =============================================================================
# Color & palette helpers (generalized for arbitrary node states)
# =============================================================================

# Matches your M1 convention for label contrast:
# for (state % 16) in this set, use white text; else black.
_DARK_INDICES = {2, 3, 5, 7, 0}


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """H in [0,1], S in [0,1], L in [0,1] -> '#RRGGBB' (via colorsys, which uses HLS)."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r * 255 + 0.5):02x}{int(g * 255 + 0.5):02x}{int(b * 255 + 0.5):02x}"


def _mix_hex(c1: str, c2: str, w: float = 0.5) -> str:
    """Linear mix of two hex colors."""
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 * (1 - w) + r2 * w + 0.5)
    g = int(g1 * (1 - w) + g2 * w + 0.5)
    b = int(b1 * (1 - w) + b2 * w + 0.5)
    return f"#{r:02x}{g:02x}{b:02x}"


def _nx_id(n: Any) -> Any:
    """Prefer n.id if present; otherwise use n itself."""
    return getattr(n, "id", n)


def _state_label(n: Any) -> str:
    """Human-readable label to draw on node (first character of state/label if textual)."""
    for attr in ("state", "State", "status", "Status", "label"):
        if hasattr(n, attr):
            v = getattr(n, attr)
            s = str(v)
            return s[:1].upper() if s else "?"
    try:
        return str(n)[:1].upper()
    except Exception:
        return "?"


def _state_index(n: Any) -> int:
    """
    Stable integer index for palette selection.
    Priority:
      - numeric .state / label (int or numeric string)
      - first letter A..Z -> 0..25
      - stable CRC32 of string repr (positive int)
    """
    # 1) numeric state if possible
    for attr in ("state", "State", "status", "Status", "label"):
        if hasattr(n, attr):
            v = getattr(n, attr)
            try:
                return int(v)
            except Exception:
                s = str(v).strip()
                if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                    return int(s)
                if s:
                    ch = s[0]
                    if ch.isalpha():
                        return ord(ch.upper()) - ord("A")
    # 2) fallback: first char of str
    s = str(n)
    if s:
        ch = s[0]
        if ch.isalpha():
            return ord(ch.upper()) - ord("A")
    # 3) stable hash
    return zlib.crc32(s.encode("utf-8"))


def _tone_for_index(idx: int) -> tuple[str, str]:
    """
    Return (node_fill, text_color) for a tone index using a 16-hue wheel.
    Text is white for indices {2,3,5,7,0}; black otherwise (per your M1 rule).
    """
    i16 = idx % 16
    h = i16 / 16.0  # 16 evenly spaced hues
    s = 0.80
    l_node = 0.35 if i16 in _DARK_INDICES else 0.73  # darker/lighter nodes
    node_fill = _hsl_to_hex(h, s, l_node)
    text_color = "#ffffff" if i16 in _DARK_INDICES else "#000000"
    return node_fill, text_color


# =============================================================================
# Graph adapters
# =============================================================================

def _edge_endpoints(e: Any) -> Tuple[Any, Any] | None:
    """Extract endpoints for edge objects with various shapes."""
    for (a, b) in (
        ("u", "v"),
        ("source", "target"),
        ("src", "dst"),
        ("a", "b"),
        ("from_", "to"),
        ("From", "To"),
        ("Source", "Target"),
    ):
        if hasattr(e, a) and hasattr(e, b):
            return getattr(e, a), getattr(e, b)
    for m in ("endpoints", "tuple", "as_tuple"):
        if hasattr(e, m) and callable(getattr(e, m)):
            try:
                pair = getattr(e, m)()
                if isinstance(pair, (tuple, list)) and len(pair) >= 2:
                    return pair[0], pair[1]
            except Exception:
                pass
    return None


def _collect_nodes(graph: Any) -> Iterable[Any]:
    for attr in ("nodes", "Nodes", "Vertices", "vertices"):
        if hasattr(graph, attr):
            obj = getattr(graph, attr)
            return obj() if callable(obj) else obj
    return []


def _collect_edges(graph: Any) -> Iterable[Any]:
    for attr in ("edges", "Edges", "adjacent_edges", "AdjacentEdges"):
        if hasattr(graph, attr):
            obj = getattr(graph, attr)
            return obj() if callable(obj) else obj
    return []


# =============================================================================
# Layout helpers (robust to missing SciPy)
# =============================================================================

def _safe_layout(G, seed=None):
    """
    Try spring_layout; if it needs SciPy or fails, fall back to SciPy-free layouts.
    Order: spring -> fruchterman_reingold -> kamada_kawai -> shell(BFS) -> circular.
    """
    import networkx as nx

    try:
        return nx.spring_layout(G, seed=seed)
    except Exception:
        pass

    try:
        return nx.fruchterman_reingold_layout(G, seed=seed)
    except Exception:
        pass

    try:
        return nx.kamada_kawai_layout(G)
    except Exception:
        pass

    try:
        if len(G) > 0:
            center = max(G.degree, key=lambda t: t[1])[0]
            shells = []
            seen = {center}
            cur = [center]
            while cur:
                shells.append(cur)
                nxt = []
                for u in cur:
                    for v in G.neighbors(u):
                        if v not in seen:
                            seen.add(v)
                            nxt.append(v)
                cur = nxt
            return nx.shell_layout(G, nlist=shells)
    except Exception:
        pass

    return nx.circular_layout(G)


def _auto_params(n: int, base_px: int = 1000):
    """
    Choose canvas/node/font/edge sizes based on node count n.
    - Canvas grows ~ sqrt(n)
    - Nodes, fonts, edges shrink with sqrt(n)
    - Layout spacing k increases slightly for larger graphs
    """
    sf = max((n / 150.0) ** 0.5, 1.0)             # scale factor vs a 150-node baseline
    px = int(min(base_px * sf, 5000))             # canvas up to 5k x 5k
    node_size = int(max(160, min(700, 700 / sf))) # node size (points^2)
    font_size = int(max(6, min(14, 14 / sf)))     # font size (points)
    edge_width = max(0.5, 1.5 / sf)               # edge width (points)
    label_nodes_max = 300 if n <= 600 else 0      # skip labels when huge
    k = 1.6 / math.sqrt(n) if n > 0 else 1.0      # more spacing than default
    iters = 100 if n <= 800 else 200              # more iterations for larger graphs
    return px, node_size, font_size, edge_width, label_nodes_max, k, iters


# =============================================================================
# Public API
# =============================================================================

def save_png(
    graph: Any,
    out_path: Path,
    *,
    layout_seed: int | None = 42,
    figsize_px: int | None = None,     # auto if None
    dpi: int = 150,
    node_size: int | None = None,      # auto if None
    font_size: int | None = None,      # auto if None
    label_nodes_max: int | None = None # auto if None
) -> float:
    """
    Render the graph to a PNG with:
      - black background
      - state-colored nodes (generalized 16-tone wheel)
      - centered node labels (auto-disabled for very large graphs)
      - edges tinted as a mix of endpoint colors
      - auto-sized canvas and style parameters for large graphs

    Returns plotting time in seconds.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    t0 = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build networkx graph + keep originals for labels/states
    G = nx.Graph()
    orig_by_id: dict[Any, Any] = {}

    for n in _collect_nodes(graph):
        nid = _nx_id(n)
        orig_by_id[nid] = n
        G.add_node(nid)

    for e in _collect_edges(graph):
        if isinstance(e, (tuple, list)) and len(e) >= 2:
            u, v = e[0], e[1]
        else:
            pair = _edge_endpoints(e)
            if pair is None:
                continue
            u, v = pair
        uid = _nx_id(u)
        vid = _nx_id(v)
        if uid not in orig_by_id:
            orig_by_id[uid] = u
            G.add_node(uid)
        if vid not in orig_by_id:
            orig_by_id[vid] = v
            G.add_node(vid)
        G.add_edge(uid, vid)

    n_nodes = len(G)

    # ---- auto params ---------------------------------------------------------
    px_auto, ns_auto, fs_auto, ew_auto, label_max_auto, k_auto, it_auto = _auto_params(n_nodes)
    if figsize_px is None:
        figsize_px = px_auto
    if node_size is None:
        node_size = ns_auto
    if font_size is None:
        font_size = fs_auto
    if label_nodes_max is None:
        label_nodes_max = label_max_auto
    edge_width = ew_auto

    # ---- layout --------------------------------------------------------------
    if n_nodes > 0:
        try:
            pos = nx.spring_layout(G, seed=layout_seed, k=k_auto, iterations=it_auto)
        except Exception:
            pos = _safe_layout(G, seed=layout_seed)
    else:
        pos = {}

    # ---- per-node styles -----------------------------------------------------
    nodes_order = list(G.nodes())
    labels = {nid: _state_label(orig_by_id[nid]) for nid in nodes_order}
    indices = {nid: _state_index(orig_by_id[nid]) for nid in nodes_order}
    node_fill: dict[Any, str] = {}
    text_color: dict[Any, str] = {}
    for nid in nodes_order:
        fill, txt = _tone_for_index(indices[nid])
        node_fill[nid] = fill
        text_color[nid] = txt

    # per-edge colors = mix of endpoints
    edge_colors = []
    for (u, v) in G.edges():
        edge_colors.append(_mix_hex(node_fill[u], node_fill[v], 0.5))

    # ---- draw ----------------------------------------------------------------
    figsize_in = (figsize_px / dpi, figsize_px / dpi)
    fig = plt.figure(figsize=figsize_in, dpi=dpi, facecolor="black")
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")
    ax.axis("off")

    if n_nodes > 0:
        # edges first
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_width, alpha=0.9)

        # nodes on top
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=[node_fill[nid] for nid in nodes_order],
            node_size=node_size,
            linewidths=0.7,
            edgecolors="#e6e6e6",
            ax=ax,
            alpha=0.98,
        )

        # centered labels (only if graph is not too large)
        if n_nodes <= label_nodes_max:
            for nid, (x, y) in pos.items():
                ax.text(
                    x,
                    y,
                    labels[nid],
                    fontsize=font_size,
                    fontweight="bold",
                    color=text_color[nid],
                    ha="center",
                    va="center",
                )
    else:
        ax.text(0.05, 0.95, "empty graph", va="top", color="#ffffff", transform=ax.transAxes)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return time.perf_counter() - t0
