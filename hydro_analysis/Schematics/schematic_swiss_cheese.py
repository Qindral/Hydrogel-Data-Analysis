"""Swiss-cheese analogy for effective pore size (hand-drawn brush style, no text).

Row 1: two slices far apart  -> a sphere only has to pass one hole at a time,
        so the largest sphere = the hole itself.
Row 2: two slices (almost) touching -> a sphere must pass both holes at once,
        so the largest sphere = largest inscribed circle of the hole overlap
        (computed numerically); a hole without partner is blocked.
Holes are slightly irregular convex blobs; overlaps point in different directions
and give different sphere sizes.
Left: front view (slices offset, back slice grey). Right: side view (holes indicated).
Styleguide: full width (6.30 in), PDF, no bbox='tight'."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path as MPath
from scipy.ndimage import gaussian_filter1d, distance_transform_edt

HERE = Path(__file__).parent
plt.rcParams.update({"savefig.dpi": 600, "savefig.pad_inches": 0.02})

INK = "#111111"
GREY = "#717b8b"            # back-slice pen: muted ochre
BACK_FILL = "#689C7C"       # back slice: deeper cheese yellow
FRONT_FILL = "#31B363"      # front slice: light cheese yellow
FRONT_ALPHA = 0.72          # front slice translucent -> back holes show through
SPHERE_FILL = "#DD7632"     # particles: styleguide teal, lightened
SPHERE_EDGE = "#8A2D2A"     # styleguide dark teal
SIDE_FRONT_FILL = FRONT_FILL
W, H = 6.30, 4.90
RNG = np.random.default_rng(11)
fig = plt.figure(figsize=(W, H))


# =============================================================== brush engine
def _scales(ax):
    pos = ax.get_position()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    return pos.width * W / (x1 - x0), pos.height * H / (y1 - y0)


def brush(ax, x, y, w=2.0, color=INK, wobble=0.8, closed=False, zorder=3):
    """Felt-pen stroke along (x, y); w = nominal width in pt."""
    sx, sy = _scales(ax)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if closed:
        x, y = np.append(x, x[0]), np.append(y, y[0])
    X, Y = x * sx * 72, y * sy * 72
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    L = s[-1]
    n = max(int(L / 0.5), 8)
    t = np.linspace(0, L, n)
    X, Y = np.interp(t, s, X), np.interp(t, s, Y)
    dx, dy = np.gradient(X), np.gradient(Y)
    nrm = np.hypot(dx, dy) + 1e-12
    nx, ny = -dy / nrm, dx / nrm
    wob = gaussian_filter1d(RNG.normal(0, 1, n), sigma=max(n / 6, 2),
                            mode="wrap" if closed else "nearest")
    wob *= wobble / (np.abs(wob).max() + 1e-12)
    X, Y = X + nx * wob, Y + ny * wob
    u = t / L
    if closed:
        prof = 0.9 + 0.1 * np.cos(2 * np.pi * u)
    else:
        prof = np.clip(0.55 + 2.5 * u, 0, 1.08) * np.clip(0.45 + 3.0 * (1 - u), 0, 1)
        prof[: max(n // 40, 2)] *= 1.15
    half = 0.5 * w * prof * (1 + 0.12 * gaussian_filter1d(RNG.normal(0, 1, n), sigma=3))
    dx, dy = np.gradient(X), np.gradient(Y)
    nrm = np.hypot(dx, dy) + 1e-12
    nx, ny = -dy / nrm, dx / nrm
    px = np.concatenate([X + nx * half, (X - nx * half)[::-1]]) / 72 / sx
    py = np.concatenate([Y + ny * half, (Y - ny * half)[::-1]]) / 72 / sy
    ax.add_patch(Polygon(np.column_stack([px, py]), closed=True, fc=color, ec=color,
                         lw=0.15, joinstyle="round", zorder=zorder, clip_on=False))
    if not closed:
        a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        for i in (0, -1):
            ax.add_patch(Polygon(np.column_stack([(X[i] + half[i] * np.cos(a)) / 72 / sx,
                                                  (Y[i] + half[i] * np.sin(a)) / 72 / sy]),
                                 fc=color, ec="none", zorder=zorder, clip_on=False))


def ring(xc, yc, r, k=120, cw=False):
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    if cw:
        a = a[::-1]
    return np.column_stack([xc + r * np.cos(a), yc + r * np.sin(a)])


def rect(x0, y0, x1, y1, cw=False):
    p = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return p[::-1] if cw else p


def filled_with_holes(ax, outer, holes, fc, zorder, alpha=1.0):
    """Fill `outer` (ccw) with circular holes punched out (cw) -> transparent holes."""
    verts, codes = [], []
    for poly in [outer] + holes:
        verts += list(poly) + [poly[0]]
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(poly) - 1) + [MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=fc, ec="none", zorder=zorder,
                           alpha=alpha))


def sphere(ax, xc, yc, r, zorder=6):
    """Sphere = grey disc, pen outline, small white highlight."""
    ax.add_patch(Polygon(ring(xc, yc, r), fc=SPHERE_FILL, ec="none", zorder=zorder))
    brush(ax, *ring(xc, yc, r).T, w=1.4, color=SPHERE_EDGE, closed=True, wobble=0.25,
          zorder=zorder + 0.1)
    a = np.linspace(np.deg2rad(105), np.deg2rad(165), 20)
    brush(ax, xc + 0.62 * r * np.cos(a), yc + 0.62 * r * np.sin(a), w=1.1,
          color="white", wobble=0.1, zorder=zorder + 0.2)


# =============================================================== geometry
SL_W, SL_H = 1.40, 1.70            # slice size (front slice at origin)
OFF = np.array([0.16, -0.10])      # back slice offset in the front view
T = 0.09                           # slice thickness (side view): thinner than every sphere
SHAPE_RNG = np.random.default_rng(3)


def blob(xc, yc, r0, k=160):
    """Slightly irregular but convex hole: r(t) = r0 (1 + a2 cos 2(t-p2) + a3 cos 3(t-p3)).
    Convex as long as 3*a2 < 1 and 8*a3 < 1."""
    a2, a3 = SHAPE_RNG.uniform(0.06, 0.13), SHAPE_RNG.uniform(0.0, 0.05)
    p2, p3 = SHAPE_RNG.uniform(0, np.pi, 2)
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    r = r0 * (1 + a2 * np.cos(2 * (t - p2)) + a3 * np.cos(3 * (t - p3)))
    return np.column_stack([xc + r * np.cos(t), yc + r * np.sin(t)])


# front centre, r0  |  back centre, r0  (back offsets point in different directions)
front_holes = [
    blob(0.42, 1.42, 0.18),        # P1: partner shifted right       -> medium sphere
    blob(0.98, 0.92, 0.21),        # P2: partner shifted down-left   -> large sphere
    blob(0.40, 0.40, 0.15),        # P3: partner shifted up-right    -> small sphere
    blob(1.05, 0.14, 0.10),        # no partner in the back slice    -> blocked when close
]
back_holes = [
    blob(0.57, 1.36, 0.17),
    blob(0.92, 0.84, 0.20),
    blob(0.55, 0.48, 0.15),
]
pairs = [(0, 0), (1, 1), (2, 2)]


def largest_circle(*polys, h=0.0015):
    """Largest circle inside the intersection of convex polygons (distance transform)."""
    allp = np.vstack(polys)
    x0, y0 = allp.min(0) - 0.02; x1, y1 = allp.max(0) + 0.02
    gx, gy = np.meshgrid(np.arange(x0, x1, h), np.arange(y0, y1, h))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.ones(len(pts), bool)
    for p in polys:
        inside &= MPath(p).contains_points(pts)
    dist = distance_transform_edt(inside.reshape(gx.shape)) * h
    iy, ix = np.unravel_index(dist.argmax(), dist.shape)
    return gx[iy, ix], gy[iy, ix], dist[iy, ix]


far_spheres = [largest_circle(p) for p in front_holes]
near_spheres = [largest_circle(front_holes[i], back_holes[j]) for i, j in pairs]


def closed(p):
    return np.vstack([p, p[:1]])


# =============================================================== layout
FX = (-0.12, 1.70); FY = (-0.20, 1.76)          # front-view limits
SX = (-0.30, 1.30)                              # side-view z limits
k = (H - 0.25) / (2 * (FY[1] - FY[0]))          # inches per data unit (equal aspect)
fw, sw, hh = (FX[1] - FX[0]) * k, (SX[1] - SX[0]) * k, (FY[1] - FY[0]) * k
gap = 0.55
left = (W - fw - gap - sw) / 2
rows = []
for r in range(2):
    bottom = H - 0.08 - (r + 1) * hh - r * 0.09
    axf = fig.add_axes([left / W, bottom / H, fw / W, hh / H])
    axs = fig.add_axes([(left + fw + gap) / W, bottom / H, sw / W, hh / H])
    for a, xl in ((axf, FX), (axs, SX)):
        a.set_xlim(*xl); a.set_ylim(*FY); a.axis("off")
    rows.append((axf, axs))


# =============================================================== front view
def front_view(ax, close):
    bx0, by0 = OFF
    back_outer = rect(bx0, by0, bx0 + SL_W, by0 + SL_H)
    filled_with_holes(ax, back_outer, [h[::-1] for h in back_holes], BACK_FILL, 1)
    brush(ax, *closed(back_outer).T, w=1.6, color=GREY, zorder=2)
    for h in back_holes:
        brush(ax, *h.T, w=1.2, color=GREY, closed=True, zorder=2)

    front_outer = rect(0, 0, SL_W, SL_H)
    filled_with_holes(ax, front_outer, [h[::-1] for h in front_holes], FRONT_FILL, 3,
                      alpha=FRONT_ALPHA)
    brush(ax, *closed(front_outer).T, w=2.0, zorder=4)
    for h in front_holes:
        brush(ax, *h.T, w=1.6, closed=True, zorder=4)

    for (x, y, r) in (near_spheres if close else far_spheres):
        sphere(ax, x, y, 0.94 * r)


# =============================================================== side view
def bands(holes):
    return [(h[:, 1].min(), h[:, 1].max()) for h in holes]


def slab(ax, z0, y0, y1, holes, fc, color, w):
    """Slice seen from the side: continuous bar, holes indicated as dashed bands."""
    ax.add_patch(Polygon(rect(z0, y0, z0 + T, y1), fc=fc, ec="none", zorder=1))
    for g0, g1 in bands(holes):
        ax.add_patch(Polygon(rect(z0, g0, z0 + T, g1), fc="white", ec="none", zorder=1.1))
        for yy in (g0, g1):
            ax.plot([z0, z0 + T], [yy, yy], color=color, lw=0.7, ls=(0, (2, 1.6)), zorder=2)
    brush(ax, *closed(rect(z0, y0, z0 + T, y1)).T, w=w, color=color, wobble=0.3, zorder=3)


def side_view(ax, close):
    g = 0.0 if close else 0.72
    zf, zb = 0.0, T + g
    slab(ax, zb, OFF[1], OFF[1] + SL_H, back_holes, BACK_FILL, GREY, 1.4)
    slab(ax, zf, 0.0, SL_H, front_holes, SIDE_FRONT_FILL, INK, 1.8)
    if not close:
        for (_, y, r) in far_spheres:
            sphere(ax, zf + T / 2, y, 0.94 * r)
    else:
        for (_, y, r) in near_spheres:
            sphere(ax, zf + T, y, 0.94 * r)


for r, (axf, axs) in enumerate(rows):
    front_view(axf, close=bool(r))
    side_view(axs, close=bool(r))

print("near sphere diameters:", [round(2 * 0.94 * r, 3) for *_, r in near_spheres], " T =", T)
fig.savefig(HERE / "schematic_swiss_cheese.pdf", format="pdf")
fig.savefig(HERE / "schematic_swiss_cheese_preview.png", dpi=200)
