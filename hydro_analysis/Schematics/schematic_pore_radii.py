"""Pore-size definitions on one irregular pore (hand-drawn brush style).
1  maximal inscribed sphere            (largest sphere that fits in the pore)
2  sphere of equal area                (r = sqrt(A / pi), at the pore centroid)
3  sphere of the maximal radius        (smallest circle enclosing the whole pore)
4  throat sphere                       (largest sphere that passes the constriction)
Only the numbers 1-4 as text. Styleguide: class "narrow" (4.72 x 3.32 in), PDF."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path as MPath
from scipy.ndimage import gaussian_filter1d, distance_transform_edt

HERE = Path(__file__).parent
for f in HERE.glob("OpenSans-*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"savefig.dpi": 600, "savefig.pad_inches": 0.02,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Open Sans", "Arial", "DejaVu Sans"]})

INK = "#111111"
WALL = "#6e5a45"            # pore wall pen
MATRIX = "#EFE6DA"          # gel matrix (warm, muted)
C1, C1_FILL = "#2A6666", "#86BABA"   # inscribed   - teal (styleguide cat. A)
C2 = "#2A4F66"                       # equal area  - single-series blue
C3 = "#9b5191"                       # max radius  - muted magenta
C4, C4_FILL = "#A6672D", "#E8B37A"   # throat      - orange (styleguide cat. B)
RNG = np.random.default_rng(5)
W, H = 4.72, 3.32

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


def sphere(ax, xc, yc, r, fill, edge, zorder=6):
    """Sphere = filled disc, pen outline, small white highlight."""
    ax.add_patch(Polygon(ring(xc, yc, r), fc=fill, ec="none", zorder=zorder))
    brush(ax, *ring(xc, yc, r).T, w=1.4, color=edge, closed=True, wobble=0.25,
          zorder=zorder + 0.1)
    a = np.linspace(np.deg2rad(105), np.deg2rad(165), 20)
    brush(ax, xc + 0.62 * r * np.cos(a), yc + 0.62 * r * np.sin(a), w=1.1,
          color="white", wobble=0.1, zorder=zorder + 0.2)


# =============================================================== pore geometry
# Organic pore = iso-contour of a few "metaballs": a big chamber (left), a smaller
# chamber (right), a narrow throat between them and two side pockets.
BALLS = [  # (x, y, strength, width)
    (-0.60,  0.00, 1.00, 0.36),     # big chamber
    (-0.90,  0.52, 0.75, 0.16),     # pocket top-left
    (-0.28, -0.52, 0.75, 0.15),     # pocket bottom
    ( 0.66,  0.02, 0.90, 0.28),     # small chamber
    ( 0.92,  0.36, 0.65, 0.14),     # bump on the small chamber
    ( 0.04,  0.03, 0.30, 0.13),     # throat bridge
]
LEVEL = 0.32
h = 0.004
gx, gy = np.meshgrid(np.arange(-1.9, 1.9, h), np.arange(-1.5, 1.5, h))


def field(x, y):
    return sum(s * np.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * w ** 2))
               for bx, by, s, w in BALLS)


F = field(gx, gy)
mask = F > LEVEL
_cs = plt.figure().add_subplot().contour(gx, gy, F, [LEVEL])
outline = max(_cs.allsegs[0], key=len)
plt.close()

dist = distance_transform_edt(mask) * h
area = mask.sum() * h * h
cx, cy = gx[mask].mean(), gy[mask].mean()

# 1  maximal inscribed sphere (largest sphere that fits anywhere)
iy, ix = np.unravel_index(dist.argmax(), dist.shape)
S1 = (gx[iy, ix], gy[iy, ix], dist[iy, ix])

# 2  sphere of equal area, centred on the pore centroid
S2 = (cx, cy, np.sqrt(area / np.pi))


# 3  sphere of the maximal radius = smallest circle that encloses the whole pore
def enclosing_circle(P):
    def circ2(a, b):
        c = (a + b) / 2
        return c, np.linalg.norm(a - c)

    def circ3(a, b, c):
        ax_, ay = a; bx, by = b; cx_, cy_ = c
        d = 2 * (ax_ * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
        ux = ((ax_**2 + ay**2) * (by - cy_) + (bx**2 + by**2) * (cy_ - ay) + (cx_**2 + cy_**2) * (ay - by)) / d
        uy = ((ax_**2 + ay**2) * (cx_ - bx) + (bx**2 + by**2) * (ax_ - cx_) + (cx_**2 + cy_**2) * (bx - ax_)) / d
        u = np.array([ux, uy])
        return u, np.linalg.norm(a - u)

    inside = lambda c, r, p: np.linalg.norm(p - c) <= r + 1e-9
    P = P[np.random.default_rng(0).permutation(len(P))]
    c, r = P[0], 0.0
    for i in range(1, len(P)):
        if inside(c, r, P[i]):
            continue
        c, r = P[i], 0.0
        for j in range(i):
            if inside(c, r, P[j]):
                continue
            c, r = circ2(P[i], P[j])
            for k_ in range(j):
                if not inside(c, r, P[k_]):
                    c, r = circ3(P[i], P[j], P[k_])
    return c, r


(c3x, c3y), r3 = enclosing_circle(outline[::3])
S3 = (c3x, c3y, r3)

# 4  throat sphere: largest sphere that can pass from the left to the right chamber
#    = min over x (between the chambers) of the max inscribed radius in that column
cols = np.where((gx[0] > -0.25) & (gx[0] < 0.40))[0]
colmax = dist[:, cols].max(axis=0)
jt = cols[colmax.argmin()]
it = dist[:, jt].argmax()
S4 = (gx[0, jt], gy[it, 0], dist[it, jt])

# =============================================================== figure
W_IN, H_IN = 4.72, 3.32                       # styleguide class "narrow"
W, H = W_IN, H_IN
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
YR = S3[2] + 0.10                             # enclosing circle must fit
XR = YR * W / H
X0, Y0 = S3[0], S3[1]
ax.set_xlim(X0 - XR, X0 + XR); ax.set_ylim(Y0 - YR, Y0 + YR); ax.axis("off")

# matrix (gel) block with the pore cut out
block = rect(X0 - XR + 0.04, Y0 - YR + 0.04, X0 + XR - 0.04, Y0 + YR - 0.04)
filled_with_holes(ax, block, [outline[::-1]], MATRIX, 1)
brush(ax, *outline.T, w=2.0, color=WALL, closed=True, wobble=0.5, zorder=3)

# 3 enclosing and 2 equal-area: dashed outlines over everything
for (x, y, r), col in ((S3, C3), (S2, C2)):
    a = np.linspace(0, 2 * np.pi, 300)
    ax.plot(x + r * np.cos(a), y + r * np.sin(a), color=col, lw=1.3, ls=(0, (4, 2.5)), zorder=7)
    ax.plot(x, y, "+", color=col, ms=5, mew=1.0, zorder=7)

# 1 inscribed and 4 throat: real spheres
sphere(ax, *S1, fill=C1_FILL, edge=C1)
sphere(ax, *S4, fill=C4_FILL, edge=C4)


def num(x, y, n, col):
    sx, sy = _scales(ax)
    r = 6.0 / 72
    a = np.linspace(0, 2 * np.pi, 100)
    ax.add_patch(Polygon(np.column_stack([x + r / sx * np.cos(a), y + r / sy * np.sin(a)]),
                         fc="white", ec="none", zorder=9))
    brush(ax, x + r / sx * np.cos(a), y + r / sy * np.sin(a), w=1.0, color=col,
          closed=True, wobble=0.2, zorder=10)
    ax.text(x, y, str(n), ha="center", va="center", fontsize=7, color=col, zorder=11,
            fontfamily="Open Sans")


num(S1[0] - 0.15 * S1[2], S1[1] + 0.15 * S1[2], 1, C1)
a2 = np.deg2rad(118); num(S2[0] + S2[2] * np.cos(a2), S2[1] + S2[2] * np.sin(a2), 2, C2)
a3 = np.deg2rad(40);  num(S3[0] + S3[2] * np.cos(a3), S3[1] + S3[2] * np.sin(a3), 3, C3)
num(S4[0] + 0.02, S4[1] + S4[2] + 0.13, 4, C4)

print(f"r1 inscribed = {S1[2]:.3f}, r2 equal-area = {S2[2]:.3f}, "
      f"r3 enclosing = {S3[2]:.3f}, r4 throat = {S4[2]:.3f}")
fig.savefig(HERE / "schematic_pore_radii.pdf", format="pdf")
fig.savefig(HERE / "schematic_pore_radii_preview.png", dpi=200)
