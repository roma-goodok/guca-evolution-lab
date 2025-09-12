import itertools
from collections import Counter
import networkx as nx

def _sorted_edge(u, v):
    return (u, v) if u <= v else (v, u)

def _edges_in_cycle(cycle):
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        yield _sorted_edge(u, v)

def _canonical_rotation(seq):
    """Return the lexicographically smallest rotation of seq (as a tuple)."""
    n = len(seq)
    best = min(range(n), key=lambda i: tuple(seq[i:] + seq[:i]))
    return tuple(seq[best:] + seq[:best])

def _canonical_cycle(cyc):
    """Canonicalize a cycle up to rotation and reversal."""
    if len(cyc) > 1 and cyc[0] == cyc[-1]:
        cyc = cyc[:-1]
    if not cyc:
        return tuple()
    fwd = _canonical_rotation(list(cyc))
    rev = _canonical_rotation(list(reversed(cyc)))
    return fwd if fwd <= rev else rev

def _unique_cycles(cycles):
    seen = set()
    out = []
    for c in cycles:
        cc = _canonical_cycle(c)
        if cc and cc not in seen:
            seen.add(cc)
            out.append(list(cc))
    return out

def planar_faces(G):
    """Return (embedding, faces, outer_idx) using a topological embedding."""
    is_planar, emb = nx.check_planarity(G, counterexample=False)
    if not is_planar:
        raise nx.NetworkXException("Graph is not planar.")
    faces = []
    visited = set()
    for u in emb:
        for v in emb[u]:
            if (u, v) not in visited:
                cyc = emb.traverse_face(u, v, mark_half_edges=visited)
                if len(cyc) > 1 and cyc[0] == cyc[-1]:
                    cyc = cyc[:-1]
                faces.append(cyc)
    # Heuristic: choose the face with the most distinct vertices as outer
    outer_idx = max(range(len(faces)), key=lambda i: len(set(faces[i])))
    return emb, faces, outer_idx

def _decompose_face_by_allowed_chords(face, G):
    """
    For the given face boundary 'face' (cycle of vertices), split it into
    smaller chordless cycles using only graph edges between nonconsecutive
    boundary vertices. Purely topological; no geometry.
    """
    n = len(face)
    if n <= 3:
        return [face[:]]

    edges_set = {_sorted_edge(*e) for e in G.edges}
    idxs = list(range(n))

    def solve(indices):
        m = len(indices)
        if m <= 3:
            return [[face[i] for i in indices]]

        # Eligible chords are nonconsecutive boundary pairs that exist as edges
        allowed = []
        for ai in range(m):
            for aj in range(ai + 2, m):
                if ai == 0 and aj == m - 1:
                    continue  # skip wrap-around adjacency
                u = face[indices[ai]]
                v = face[indices[aj]]
                if _sorted_edge(u, v) in edges_set:
                    allowed.append((ai, aj))

        if not allowed:
            return [[face[i] for i in indices]]

        # Choose a chord that balances sub-polygons
        def score(ai, aj):
            left = aj - ai + 1
            right = m - (aj - ai) + 1
            return (max(left, right), min(left, right))

        ai, aj = min(allowed, key=lambda p: score(*p))
        left = indices[ai:aj + 1]
        right = indices[aj:] + indices[:ai + 1]
        return solve(left) + solve(right)

    return solve(idxs)

def minimal_faces_and_shell(G):
    """
    Build a minimal inner-face set (deduplicated, chordless cycles) and a
    maximal outer shell cycle (longest cycle on edges incident to exactly one
    inner face). Also return the outer face from the initial embedding.
    """
    emb, faces, outer_idx = planar_faces(G)

    # Start from the embedding’s inner faces and split them by chords.
    raw_minimal = []
    for i, f in enumerate(faces):
        if i == outer_idx:
            continue
        raw_minimal.extend(_decompose_face_by_allowed_chords(f, G))

    # **DEDUPLICATE** cycles up to rotation + reversal
    minimal_inner = _unique_cycles(raw_minimal)

    # Edge usage across minimal inner faces
    cnt = Counter()
    for f in minimal_inner:
        for e in _edges_in_cycle(f):
            cnt[e] += 1

    # Shell edges = edges on exactly one minimal inner face
    shell_edges = [e for e, c in cnt.items() if c == 1]

    # A “maximal outer shell” = the longest cycle in the shell-edge subgraph
    H = nx.Graph()
    H.add_edges_from(shell_edges)
    shell_cycles = nx.cycle_basis(H)
    outer_shell_cycle = max(shell_cycles, key=len) if shell_cycles else []

    # Also expose the embedding’s own outer face (may be shorter)
    outer_face_from_embedding = faces[outer_idx]

    return {
        "minimal_inner_faces": minimal_inner,
        "outer_shell_cycle": outer_shell_cycle,               # maximal shell
        "shell_vertices": sorted(set(outer_shell_cycle)),
        "outer_face_from_embedding": outer_face_from_embedding  # the embedding’s outer
    }

def is_triangle_mesh(G):
    info = minimal_faces_and_shell(G)
    tri = all(len(c) == 3 for c in info["minimal_inner_faces"])
    return tri, info


def _g_7_tri_pie():
    G = nx.Graph()
    G.add_edges_from([
        (3,4), (4,5), (3,6), (4,6), (5,6),
        (4,7), (5,7), (5,8), (7,8),
        (7,9), (8,9), (9,10), (10,11),
        (7,10), (7,11)
    ])
    return G


def G(edges):
    g = nx.Graph(); g.add_edges_from(edges); return g
#@case("10 triangles (2x6-deg vertices are inside the shell) ")
def tri_line4():
    return G([(0,1), (1,2), (2,3), (3,4), (4,5),
              (5,0), (0,6), (1,6), (2,6), (3,6), (4,6),
              (5,6), (3,7), (3,8), (3,9), (2,7),
              (7,8), (8,9), (9,4)])

#@case("10 triangles as pie (2x6-deg  on the shell) ")
def tri_line5():
    return G([(0,1), (1,2), (2,3), (3,4), (4,5),
              (0,6), (1,6), (2,6), (3,6), (4,6),
              (5,6), (4,7), (5,7), (8,7), (9,7), 
              (10,7), (11,7), (5,8), (8,9), (9,10), 
              (10,11)])

#@case("8 triangles as pie (1x6-deg on the shell) ")
def tri_line6():
    return G([(2,3), (3,4), (4,5),
              (2,6), (3,6), (4,6),
              (5,6), (4,7), (5,7), (8,7), (9,7), 
              (10,7), (11,7), (5,8), (8,9), (9,10), 
              (10,11)])

G = tri_line4()
tri, info = is_triangle_mesh(G)
inner_faces = info["minimal_inner_faces"]
outer_shell = info["outer_shell_cycle"]
shell_vertices = info["shell_vertices"]
outer_face_embed = info["outer_face_from_embedding"]

print("All inner faces minimal & triangular?", tri)
print("Inner minimal faces:", inner_faces)
print("Max outer shell cycle:", outer_shell)
print("Shell vertices:", shell_vertices)
print("Outer face (from embedding):", outer_face_embed)

v = G.number_of_nodes()
e = G.number_of_edges()
f = len(inner_faces) + 1  # +1 for the (single) outer face
print("v,e,f =", v, e, f, "  v-e+f =", v - e + f)


