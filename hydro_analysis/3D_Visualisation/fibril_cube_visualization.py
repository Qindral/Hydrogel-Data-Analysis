import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Helpers: geometry + WLC-like centerline
# -----------------------------
def _orthonormal_frame(u):
    """Given unit vector u, return two unit vectors v,w perpendicular to u."""
    u = u / np.linalg.norm(u)
    # pick a vector not parallel to u
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(u, a)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    w /= np.linalg.norm(w)
    return v, w

def wlc_like_segment(A, B, Lp=0.2, n=80, noise_scale=1.0, max_tries=20, cube_min=0.0, cube_max=1.0, rng=None):
    """
    Create a smooth, correlated random curve from A to B that stays in the cube.
    Uses OU-correlated transverse noise with a sin(pi t) envelope => exact endpoints.
    """
    if rng is None:
        rng = np.random.default_rng()

    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    d = B - A
    L = np.linalg.norm(d)
    if L == 0:
        raise ValueError("A and B must be different points.")
    u = d / L
    v, w = _orthonormal_frame(u)

    # Parameter along the chord
    t = np.linspace(0.0, 1.0, n)
    base = A[None, :] + t[:, None] * d[None, :]

    # OU correlation along arc-length-ish parameter
    # ds chosen as chord length/(n-1); correlation alpha ~ exp(-ds/Lp)
    ds = L / max(n - 1, 1)
    alpha = np.exp(-ds / max(Lp, 1e-9))

    # Target amplitude: proportional to sqrt(ds/Lp) * L
    # (heuristic; increase/decrease via noise_scale)
    amp0 = noise_scale * L * np.sqrt(max(ds / max(Lp, 1e-9), 0.0))

    envelope = np.sin(np.pi * t)  # 0 at endpoints, smooth maximum in the middle

    # Try to keep inside cube by reducing amplitude if needed
    amp = amp0
    for _ in range(max_tries):
        # correlated 1D OU for two transverse components
        xi = np.zeros(n)
        eta = np.zeros(n)
        # OU innovation variance to keep stationary var ~ 1
        sigma = np.sqrt(max(1.0 - alpha**2, 1e-12))
        for i in range(1, n):
            xi[i] = alpha * xi[i-1] + sigma * rng.normal()
            eta[i] = alpha * eta[i-1] + sigma * rng.normal()

        disp = (amp * envelope)[:, None] * (xi[:, None] * v[None, :] + eta[:, None] * w[None, :])
        curve = base + disp

        # Enforce exact endpoints (numerically already ~exact; make explicit)
        curve[0] = A
        curve[-1] = B

        if np.all(curve >= cube_min - 1e-12) and np.all(curve <= cube_max + 1e-12):
            return curve
        amp *= 0.7  # shrink if outside

    # If we never fit, return the base line (safe fallback)
    return base

def fibril_centerline_through_three_points(p1, p3, p2, Lp=0.2, n1=90, n2=90, noise_scale=1.0, rng=None):
    """Piecewise curve p1->p3 and p3->p2; concatenated into one continuous polyline."""
    if rng is None:
        rng = np.random.default_rng()
    seg1 = wlc_like_segment(p1, p3, Lp=Lp, n=n1, noise_scale=noise_scale, rng=rng)
    seg2 = wlc_like_segment(p3, p2, Lp=Lp, n=n2, noise_scale=noise_scale, rng=rng)
    return np.vstack([seg1[:-1], seg2])  # drop duplicate midpoint

# -----------------------------
# Collision checks
# -----------------------------
def min_polyline_distance(P, Q):
    """Approximate minimum distance between two polylines by point sampling."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    diff = P[:, None, :] - Q[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return float(np.sqrt(np.min(d2)))

def self_intersection_distance(P, min_skip=5):
    """Approximate minimum distance within a single polyline, skipping neighbors."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if n <= min_skip + 1:
        return np.inf
    min_d2 = np.inf
    for i in range(n - min_skip - 1):
        diff = P[i] - P[i + min_skip + 1 :]
        d2 = np.sum(diff * diff, axis=1)
        local_min = np.min(d2)
        if local_min < min_d2:
            min_d2 = local_min
    return float(np.sqrt(min_d2))

def generate_fibril_no_overlap(
    p1,
    p3,
    p2,
    existing_curves,
    radius,
    Lp,
    n1,
    n2,
    noise_scale,
    rng,
    clearance_factor=1.05,
    max_tries=40,
):
    """Generate a fibril curve that keeps a clearance from existing curves."""
    min_clearance = 2.0 * radius * clearance_factor
    current_noise = noise_scale

    for _ in range(max_tries):
        curve = fibril_centerline_through_three_points(
            p1, p3, p2, Lp=Lp, n1=n1, n2=n2, noise_scale=current_noise, rng=rng
        )

        # Avoid self-intersections when possible
        if self_intersection_distance(curve) < min_clearance:
            current_noise *= 0.85
            continue

        if not existing_curves:
            return curve

        dmin = min(min_polyline_distance(curve, other) for other in existing_curves)
        if dmin >= min_clearance:
            return curve

        current_noise *= 0.85

    # Fallback: return last candidate (may overlap)
    return curve

# -----------------------------
# Tube mesh from polyline (parallel transport-ish frames)
# -----------------------------
def tube_mesh_from_polyline(P, radius=0.025, n_theta=18):
    """
    Build a triangulated tube mesh around polyline P (N x 3).
    Returns vertices (V x 3) and triangle indices (F x 3).
    """
    P = np.asarray(P, dtype=float)
    N = P.shape[0]
    if N < 2:
        raise ValueError("Polyline must have at least 2 points.")

    # tangents
    T = np.diff(P, axis=0)
    T_norm = np.linalg.norm(T, axis=1, keepdims=True)
    T = T / np.clip(T_norm, 1e-12, None)

    # initial normal
    t0 = T[0]
    v0, w0 = _orthonormal_frame(t0)
    Nrm = np.zeros((N, 3))
    Bin = np.zeros((N, 3))
    Nrm[0] = v0
    Bin[0] = w0

    # propagate frame by projecting previous normal onto new normal plane
    for i in range(1, N):
        ti = T[i-1] if i-1 < T.shape[0] else T[-1]
        tprev = T[i-2] if i-2 >= 0 else T[0]
        # Use the tangent at segment i-1; this is fine for polyline tubes
        # Re-orthonormalize:
        ni = Nrm[i-1] - np.dot(Nrm[i-1], ti) * ti
        if np.linalg.norm(ni) < 1e-10:
            # fallback if nearly parallel
            vi, wi = _orthonormal_frame(ti)
            ni = vi
        ni /= np.linalg.norm(ni)
        bi = np.cross(ti, ni)
        bi /= np.linalg.norm(bi)
        Nrm[i] = ni
        Bin[i] = bi

    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n_theta,2)

    # vertices
    verts = []
    for i in range(N):
        ring = P[i][None, :] + radius * (circle[:, 0:1] * Nrm[i][None, :] + circle[:, 1:2] * Bin[i][None, :])
        verts.append(ring)
    verts = np.vstack(verts)  # (N*n_theta, 3)

    # faces
    faces = []
    for i in range(N - 1):
        for j in range(n_theta):
            a = i * n_theta + j
            b = i * n_theta + (j + 1) % n_theta
            c = (i + 1) * n_theta + j
            d = (i + 1) * n_theta + (j + 1) % n_theta
            faces.append([a, c, b])
            faces.append([b, c, d])
    faces = np.array(faces, dtype=int)
    return verts, faces

# -----------------------------
# Scene construction (cube + fibrils)
# -----------------------------
def cube_edges_traces():
    # 12 edges of a unit cube
    corners = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1]
    ], dtype=float)
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    xs, ys, zs = [], [], []
    for i,j in edges:
        xs += [corners[i,0], corners[j,0], None]
        ys += [corners[i,1], corners[j,1], None]
        zs += [corners[i,2], corners[j,2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="Cube",
                       line=dict(width=4))

def add_fibril_mesh(fig, centerline, radius=0.025, name="Fibril"):
    V, F = tube_mesh_from_polyline(centerline, radius=radius, n_theta=20)
    fig.add_trace(go.Mesh3d(
        x=V[:,0], y=V[:,1], z=V[:,2],
        i=F[:,0], j=F[:,1], k=F[:,2],
        name=name,
        opacity=0.95,
        flatshading=False,
        showscale=False
    ))

def add_points(fig, pts, name="Points"):
    pts = np.asarray(pts, dtype=float)
    fig.add_trace(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode="markers+text",
        text=[name]*len(pts),
        textposition="top center",
        marker=dict(size=4),
        showlegend=False
    ))

def main(seed=7, Lp=0.1, tube_diameter=0.05, noise_scale=0.15):
    rng = np.random.default_rng(seed)
    r = tube_diameter / 2.0
    n1 = 90
    n2 = 90

    # --- Your constraints (interpreting: 3 fibrils total) ---
    f1_p1 = (0.2, 0.0, 0.4)
    f1_p2 = (0.6, 1.0, 0.4)
    f1_p3 = (0.8, 0.5, 0.7)

    f2_p1 = (0.1, 0.0, 0.7)
    f2_p2 = (0.8, 1.0, 0.1)
    f2_p3 = (0.7, 0.7, 0.6)

    f3_p1 = (1.0, 0.5, 0.6)
    f3_p2 = (0.0, 0.4, 0.5)
    f3_p3 = (0.5, 0.6, 0.4)

    f4_p1 = (.8, 0.4, 0.)
    f4_p2 = (0.2, 1, 0.4)
    f4_p3 = (0.4, 0.5, 0.5)

    curves = []
    c1 = generate_fibril_no_overlap(f1_p1, f1_p3, f1_p2, curves, r, Lp, n1, n2, noise_scale, rng)
    curves.append(c1)
    c2 = generate_fibril_no_overlap(f2_p1, f2_p3, f2_p2, curves, r, Lp, n1, n2, noise_scale, rng)
    curves.append(c2)
    c3 = generate_fibril_no_overlap(f3_p1, f3_p3, f3_p2, curves, r, Lp, n1, n2, noise_scale, rng)
    curves.append(c3)
    c4 = generate_fibril_no_overlap(f4_p1, f4_p3, f4_p2, curves, r, Lp, n1, n2, noise_scale, rng)
    curves.append(c4)

    fig = go.Figure()
    fig.add_trace(cube_edges_traces())

    add_fibril_mesh(fig, c1, radius=r, name="Fibril 1")
    add_fibril_mesh(fig, c2, radius=r, name="Fibril 2")
    add_fibril_mesh(fig, c3, radius=r, name="Fibril 3")
    add_fibril_mesh(fig, c4, radius=r, name="Fibril 4")

    # show constraint points (optional)
    add_points(fig, np.array([f1_p1, f1_p3, f1_p2]), name="F1")
    add_points(fig, np.array([f2_p1, f2_p3, f2_p2]), name="F2")
    add_points(fig, np.array([f3_p1, f3_p3, f3_p2]), name="F3")
    add_points(fig, np.array([f4_p1, f4_p3, f4_p2]), name="F4")

    fig.update_layout(
        title="Unit Cube with 4 WLC-like Fibrils (tube diameter 0.05, Lp=0.2)",
        scene=dict(
            xaxis=dict(range=[0,1], title="x", backgroundcolor="white"),
            yaxis=dict(range=[0,1], title="y", backgroundcolor="white"),
            zaxis=dict(range=[0,1], title="z", backgroundcolor="white"),
            aspectmode="cube",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()

if __name__ == "__main__":
    main()
