"""Hand-drawn (felt-pen / brush) schematic: repulsive (F > 0) and adhesive (F < 0)
forces when a colloidal-probe cantilever (sphere on cantilever) interacts with a
soft surface. Only stage numbers and axis labels as text.
Styleguide: full width (6.30 in), Open Sans, PDF, no bbox='tight'."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, Circle
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).parent
for f in HERE.glob("OpenSans-*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Open Sans", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Open Sans", "mathtext.it": "Open Sans:italic",
    "savefig.dpi": 600, "savefig.pad_inches": 0.02,
})

INK = "#111111"
GREY = "#8a8a8a"
W, H = 6.30, 4.0
RNG = np.random.default_rng(7)

fig = plt.figure(figsize=(W, H))


# --------------------------------------------------------------- brush engine
def _scales(ax):
    """inches per data unit in x and y (fixed layout, so this is exact)."""
    pos = ax.get_position()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    return pos.width * W / (x1 - x0), pos.height * H / (y1 - y0)


def brush(ax, x, y, w=2.2, color=INK, wobble=0.9, closed=False, zorder=3):
    """Draw a felt-pen stroke along (x, y). w = nominal width in pt."""
    sx, sy = _scales(ax)
    X, Y = np.asarray(x, float) * sx * 72, np.asarray(y, float) * sy * 72   # -> points
    seg = np.hypot(np.diff(X), np.diff(Y))
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    n = max(int(L / 0.5), 8)
    t = np.linspace(0, L, n)
    X, Y = np.interp(t, s, X), np.interp(t, s, Y)
    # low-frequency hand wobble perpendicular to the path
    dx, dy = np.gradient(X), np.gradient(Y)
    norm = np.hypot(dx, dy) + 1e-12
    nx, ny = -dy / norm, dx / norm
    wob = gaussian_filter1d(RNG.normal(0, 1, n), sigma=max(n / 6, 2), mode="wrap" if closed else "nearest")
    wob *= wobble / (np.abs(wob).max() + 1e-12)
    X, Y = X + nx * wob, Y + ny * wob
    # pressure profile: ink blob at start, slight taper at end, grainy variation
    u = t / L
    if closed:
        prof = 0.9 + 0.1 * np.cos(2 * np.pi * u)
    else:
        prof = np.clip(0.55 + 2.5 * u, 0, 1.08) * np.clip(0.45 + 3.0 * (1 - u), 0, 1)
        prof[: max(n // 40, 2)] *= 1.15
    grain = 1 + 0.12 * gaussian_filter1d(RNG.normal(0, 1, n), sigma=3)
    half = 0.5 * w * prof * grain
    dx, dy = np.gradient(X), np.gradient(Y)
    norm = np.hypot(dx, dy) + 1e-12
    nx, ny = -dy / norm, dx / norm
    lx, ly = X + nx * half, Y + ny * half
    rx, ry = X - nx * half, Y - ny * half
    px = np.concatenate([lx, rx[::-1]]) / 72 / sx
    py = np.concatenate([ly, ry[::-1]]) / 72 / sy
    ax.add_patch(Polygon(np.column_stack([px, py]), closed=True, fc=color, ec=color,
                         lw=0.15, joinstyle="round", zorder=zorder, clip_on=False))
    if not closed:   # round pen caps
        for i in (0, -1):
            ax.add_patch(Polygon(_disc(X[i], Y[i], half[i], sx, sy), fc=color, ec="none",
                                 zorder=zorder, clip_on=False))


def _disc(Xp, Yp, r, sx, sy, k=16):
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.column_stack([(Xp + r * np.cos(a)) / 72 / sx, (Yp + r * np.sin(a)) / 72 / sy])


def arrow(ax, p0, p1, w=1.8, color=INK, head=6.0, both=False, zorder=4):
    """Brush shaft + open triangular head like a pen sketch (head size in pt)."""
    sx, sy = _scales(ax)
    brush(ax, [p0[0], p1[0]], [p0[1], p1[1]], w=w, color=color, wobble=0.4, zorder=zorder)
    ends = [(p0, p1)] + ([(p1, p0)] if both else [])
    for a, b in ends:
        bx, by = b[0] * sx * 72, b[1] * sy * 72
        ax_, ay_ = a[0] * sx * 72, a[1] * sy * 72
        d = np.array([bx - ax_, by - ay_]); d /= np.linalg.norm(d)
        nrm = np.array([-d[1], d[0]])
        tip = np.array([bx, by]) + d * head * 0.35
        l = tip - d * head + nrm * head * 0.45
        r = tip - d * head - nrm * head * 0.45
        tri = np.array([l, tip, r, l])
        brush(ax, tri[:, 0] / 72 / sx, tri[:, 1] / 72 / sy, w=w * 0.9, color=color,
              wobble=0.2, zorder=zorder)


def circle(ax, xc, yc, r, w=1.8, color=INK, zorder=3, fill=None):
    a = np.linspace(0, 2 * np.pi, 120)
    if fill:
        ax.add_patch(Circle((xc, yc), r, fc=fill, ec="none", zorder=zorder - 0.5))
    brush(ax, xc + r * np.cos(a), yc + r * np.sin(a), w=w, color=color, closed=True,
          wobble=0.35, zorder=zorder)


def num(ax, x, y, n):
    """Stage number in a thin pen circle (r given in pt)."""
    sx, sy = _scales(ax)
    r = 6.0 / 72
    a = np.linspace(0, 2 * np.pi, 100)
    ax.add_patch(Polygon(np.column_stack([x + r / sx * np.cos(a), y + r / sy * np.sin(a)]),
                         fc="white", ec="none", zorder=9, clip_on=False))
    brush(ax, x + r / sx * np.cos(a), y + r / sy * np.sin(a), w=0.9, closed=True,
          wobble=0.25, zorder=10)
    ax.text(x, y, str(n), ha="center", va="center", fontsize=7, color=INK, zorder=11)


# --------------------------------------------------------------- layout (fixed)
inset_w, inset_h, top = 1.12 / W, 0.95 / H, 0.985
gap = (1 - 5 * inset_w) / 6
insets = []
for i in range(5):
    a = fig.add_axes([gap + i * (inset_w + gap), top - inset_h, inset_w, inset_h])
    a.set_xlim(-1, 1)
    half_h = (0.95 / 1.12)                       # equal aspect: y range = x range * h/w
    a.set_ylim(-0.5, -0.5 + 2 * half_h)
    a.axis("off")
    insets.append(a)
ax = fig.add_axes([0.085, 0.10, 0.88, 0.58])
ax.set_xlim(-1.08, 1.12); ax.set_ylim(-0.45, 1.15)
ax.axis("off")

# --------------------------------------------------------------- insets
xs = np.linspace(-1, 1, 200)
R = 0.15
X_CLAMP = -0.95
stages = [
    # (surface profile, sphere bottom height, cantilever deflection)
    (np.zeros_like(xs),                   0.32,  0.00),   # 1 approach
    (np.zeros_like(xs),                   0.00,  0.00),   # 2 contact
    (-0.10 * np.exp(-(xs / 0.30) ** 2),  -0.10, +0.08),   # 3 indentation: bent up
    (0.18 * np.exp(-(xs / 0.22) ** 2),    0.18, -0.10),   # 4 adhesion: bent down
    (np.zeros_like(xs),                   0.32,  0.00),   # 5 detached
]
for i, (a, (prof, bottom, defl)) in enumerate(zip(insets, stages), start=1):
    brush(a, xs * 0.95, prof, w=2.0)
    yc = bottom + R
    circle(a, 0, yc, R, w=1.7, fill="white", zorder=4)
    end = yc + R + 0.02
    xb = np.linspace(X_CLAMP, 0.05, 50)
    yb = (end - defl) + defl * ((xb - X_CLAMP) / (0 - X_CLAMP)) ** 2
    brush(a, xb, yb, w=2.4, zorder=5)
    num(a, 0.80, 0.95, i)

# --------------------------------------------------------------- force curve
x_app = np.linspace(-1.0, 1.0, 300)
F_app = np.where(x_app > 0, np.clip(x_app, 0, None) ** 1.5, 0.0)
x_min = -0.24
px = [x_min, -0.10, 0.05, 0.25, 0.55, 1.0]
py = [-0.20, -0.17, -0.07, 0.09, 0.38, 1.0]
x_ret = np.linspace(x_min, 1.0, 250)
F_ret = PchipInterpolator(px, py)(x_ret)

ax.plot([-1.05, 1.08], [0, 0], color=GREY, lw=0.6, ls=(0, (3, 3)), zorder=1)
brush(ax, x_ret, F_ret, w=2.0, color=GREY, zorder=2)
brush(ax, [x_min, x_min], [F_ret[0], 0], w=1.6, color=GREY, zorder=2)
brush(ax, x_app, F_app, w=2.4, zorder=3)

arrow(ax, (-0.80, 0.11), (-0.45, 0.11))
arrow(ax, (0.55, 0.64), (0.80, 0.86))
arrow(ax, (0.52, 0.24), (0.31, 0.04), color=GREY)
arrow(ax, (x_min - 0.08, -0.025), (x_min - 0.08, -0.175), w=1.2, head=4.5, color=GREY, both=True)

num(ax, -0.88, 0.0, 1)
num(ax, 0.0, 0.0, 2)
num(ax, 1.0, 1.0, 3)
num(ax, x_min, -0.20, 4)
num(ax, x_min, 0.0, 5)

# axes: two pen strokes with arrowheads
arrow(ax, (-1.07, -0.43), (-1.07, 1.13), w=2.0, head=7)
arrow(ax, (-1.07, -0.43), (1.11, -0.43), w=2.0, head=7)
ax.text(-1.11, 1.13, r"Force $F$", rotation=90, ha="right", va="top", fontsize=9)
ax.text(1.11, -0.50, r"Displacement $z$", ha="right", va="top", fontsize=9)

fig.savefig(HERE / "schematic_force_soft_surface_brush.pdf", format="pdf")
fig.savefig(HERE / "schematic_force_soft_surface_brush_preview.png", dpi=200)
