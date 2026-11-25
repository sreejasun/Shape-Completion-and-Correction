"""Recognize, complete and regularize hand-drawn shapes.

Pipeline for a drawing (a list of strokes, each an (N, 2) array of points):

1. ``merge_strokes``  - join strokes whose endpoints meet (e.g. a square drawn
   as four separate lines) and close strokes whose ends nearly touch.
2. ``classify``       - for each resulting stroke, try a family of geometric
   models (line, arc, circle, ellipse, polygon, regular polygon, star,
   polyline) and keep the simplest one that explains the points well.
3. Each model is *regularized*: near-horizontal lines become horizontal,
   near-square rectangles become squares, almost-regular polygons become
   regular, and so on.

Everything works in the stroke's own coordinates, and all tolerances are
relative to the stroke's size, so the same thresholds work for pixel
coordinates from a canvas and for arbitrary units from a CSV file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Tunable tolerances (all relative to stroke size)
# ---------------------------------------------------------------------------
RESAMPLE_N = 256            # points per stroke after uniform resampling
CLOSE_GAP_FRAC = 0.12       # ends closer than this * length => closed shape
MERGE_GAP_FRAC = 0.12       # endpoint gap (vs. shorter stroke) to join strokes
CORNER_WINDOW_FRAC = 0.03   # half-window (fraction of length) for corner angle
CORNER_ANGLE_DEG = 38.0     # turning angle that counts as a corner
LINE_TOL = 0.025            # mean deviation / length for a straight line
ARC_TOL = 0.04              # mean deviation / radius for a circular arc
ELLIPSE_TOL = 0.03          # mean deviation / diagonal for circle / ellipse
POLY_TOL = 0.03             # mean deviation / diagonal for a polygon
MAX_EDGE_BEND_DEG = 22.0    # a polygon edge may not bend more than this
CIRCLE_AXIS_RATIO = 0.85    # minor/major above this => circle
SNAP_ANGLE_DEG = 8.0        # snap orientations within this of "nice" angles
REGULAR_TOL_FEW = 0.08      # mean vertex offset / radius to call a 3-4-gon regular
REGULAR_TOL_MANY = 0.15     # same, for 5+ vertices and stars
DOT_FRAC = 0.02             # strokes smaller than this * drawing size are dots


@dataclass
class Shape:
    """A recognized shape: its type, clean geometry and fit parameters."""

    kind: str
    points: np.ndarray          # (M, 2) clean outline, closed shapes repeat p0
    closed: bool
    params: dict = field(default_factory=dict)
    error: float = 0.0          # normalized fit error of the chosen model

    @property
    def label(self) -> str:
        if self.kind == "regular_polygon":
            names = {3: "equilateral triangle", 4: "square", 5: "pentagon",
                     6: "hexagon", 7: "heptagon", 8: "octagon"}
            n = self.params["n"]
            return names.get(n, f"regular {n}-gon")
        if self.kind == "star":
            return f"{self.params['n']}-point star"
        if self.kind == "polygon":
            return f"{self.params['n']}-sided polygon"
        return self.kind.replace("_", " ")


# ---------------------------------------------------------------------------
# Basic geometry helpers
# ---------------------------------------------------------------------------
def _dedupe(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 2:
        return pts
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return pts[np.r_[True, step > 1e-9]]


def _cumlen(pts: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]


def path_length(pts: np.ndarray) -> float:
    return float(_cumlen(pts)[-1]) if len(pts) > 1 else 0.0


def _diag(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0


def resample(pts: np.ndarray, n: int = RESAMPLE_N, closed: bool = False) -> np.ndarray:
    """Resample a polyline to ``n`` points equally spaced along its length.

    For closed curves the result does not repeat the first point.
    """
    if closed:
        pts = np.vstack([pts, pts[:1]])
        n_out = n + 1
    else:
        n_out = n
    s = _cumlen(pts)
    if s[-1] == 0:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n_out)
    out = np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])
    return out[:-1] if closed else out


def _close(pts: np.ndarray) -> np.ndarray:
    return np.vstack([pts, pts[:1]])


def _point_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab) or 1e-12
    t = np.clip((p - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def dist_to_polyline(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point in ``p`` to the nearest segment of ``poly``."""
    d = np.full(len(p), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        d = np.minimum(d, _point_segment_dist(p, a, b))
    return d


def _rotate(pts: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    return (pts - center) @ r.T + center


def _snap_angle(angle: float, step: float, offset: float = 0.0,
                tol_deg: float = SNAP_ANGLE_DEG) -> float:
    """Snap ``angle`` to the nearest ``offset + k*step`` if within tolerance."""
    k = np.round((angle - offset) / step)
    target = offset + k * step
    return float(target) if abs(angle - target) <= np.radians(tol_deg) else angle


# ---------------------------------------------------------------------------
# Stroke preprocessing: closure detection and stroke merging
# ---------------------------------------------------------------------------
def detect_closure(pts: np.ndarray, gap_frac: float = CLOSE_GAP_FRAC):
    """Decide whether a stroke is meant to be closed.

    Handles both under-shoot (a small gap between the ends) and over-shoot
    (the pen runs past the start point), trimming the overlap.

    Returns ``(points, closed)``.
    """
    if len(pts) < 8:
        return pts, False
    L = path_length(pts)
    if L == 0:
        return pts, False
    s = _cumlen(pts)
    head = np.where(s <= 0.3 * L)[0]
    tail = np.where(s >= 0.7 * L)[0]
    d = np.linalg.norm(pts[head][:, None, :] - pts[tail][None, :, :], axis=2)
    # Prefer joins that trim little of the stroke, so that a self-crossing in
    # the middle of a shape (e.g. a pentagram) is not mistaken for the closure.
    trimmed = s[head][:, None] + (L - s[tail])[None, :]
    i, j = np.unravel_index(np.argmin(d + 0.5 * trimmed), d.shape)
    gap = d[i, j]
    i, j = head[i], tail[j]
    loop_len = s[j] - s[i]
    if loop_len > 0 and gap <= gap_frac * loop_len:
        return pts[i:j + 1], True
    return pts, False


def merge_strokes(strokes: list[np.ndarray], gap_frac: float = MERGE_GAP_FRAC):
    """Join open strokes whose endpoints (nearly) touch.

    Returns a list of ``(points, closed)`` tuples.
    """
    items = []
    for s in strokes:
        s = _dedupe(s)
        if len(s) < 2:
            continue
        pts, closed = detect_closure(s)
        items.append([pts, closed])

    while True:
        best = None
        open_idx = [k for k, (_, c) in enumerate(items) if not c]
        for a_pos, a in enumerate(open_idx):
            for b in open_idx[a_pos + 1:]:
                pa, pb = items[a][0], items[b][0]
                tol = gap_frac * min(path_length(pa), path_length(pb))
                for ea in (0, -1):
                    for eb in (0, -1):
                        d = np.linalg.norm(pa[ea] - pb[eb])
                        if d <= tol and (best is None or d < best[0]):
                            best = (d, a, b, ea, eb)
        if best is None:
            break
        _, a, b, ea, eb = best
        pa, pb = items[a][0], items[b][0]
        if ea == 0:
            pa = pa[::-1]          # make pa end at the joining endpoint
        if eb == -1:
            pb = pb[::-1]          # make pb start at the joining endpoint
        joined = _dedupe(np.vstack([pa, pb]))
        items[a] = list(detect_closure(joined))
        del items[b]

    return [(p, c) for p, c in items]


# ---------------------------------------------------------------------------
# Corner detection
# ---------------------------------------------------------------------------
def find_corners(pts: np.ndarray, closed: bool,
                 window_frac: float = CORNER_WINDOW_FRAC,
                 angle_deg: float = CORNER_ANGLE_DEG) -> list[int]:
    """Indices of sharp corners in a uniformly resampled stroke."""
    n = len(pts)
    mode = "wrap" if closed else "nearest"
    sm = np.column_stack([gaussian_filter1d(pts[:, 0], 1.5, mode=mode),
                          gaussian_filter1d(pts[:, 1], 1.5, mode=mode)])
    k = max(2, int(round(window_frac * n)))
    idx = np.arange(n)
    if closed:
        prev, nxt = sm[(idx - k) % n], sm[(idx + k) % n]
        valid = np.ones(n, bool)
    else:
        prev, nxt = sm[np.clip(idx - k, 0, n - 1)], sm[np.clip(idx + k, 0, n - 1)]
        valid = (idx >= k) & (idx < n - k)
    v1, v2 = sm - prev, nxt - sm
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = (v1 * v2).sum(axis=1)
    turn = np.abs(np.degrees(np.arctan2(cross, dot)))
    turn[~valid] = 0.0

    corners, taken = [], np.zeros(n, bool)
    for i in np.argsort(-turn):
        if turn[i] < angle_deg:
            break
        if taken[i]:
            continue
        corners.append(int(i))
        nb = np.arange(i - 2 * k, i + 2 * k + 1)
        nb = nb % n if closed else nb[(nb >= 0) & (nb < n)]
        taken[nb] = True
    return sorted(corners)


# ---------------------------------------------------------------------------
# Primitive fits
# ---------------------------------------------------------------------------
def fit_line(pts: np.ndarray):
    """Total-least-squares line. Returns (centroid, unit direction, mean dist)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    resid = np.abs((pts - c) @ np.array([-d[1], d[0]]))
    return c, d, float(resid.mean())


def fit_circle(pts: np.ndarray):
    """Geometric circle fit. Returns (center, radius, mean |dist - r|)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    c0 = np.array([-D / 2, -E / 2])
    r0 = np.sqrt(max(c0 @ c0 - F, 1e-12))

    def resid(p):
        return np.linalg.norm(pts - p[:2], axis=1) - p[2]

    sol = least_squares(resid, np.r_[c0, r0])
    c, r = sol.x[:2], abs(sol.x[2])
    return c, float(r), float(np.abs(resid(sol.x)).mean())


def fit_ellipse(pts: np.ndarray):
    """Direct least-squares ellipse fit (Fitzgibbon / Halir-Flusser).

    Returns (center, (a, b), angle) with a >= b, or None if no ellipse fits.
    """
    m = pts.mean(axis=0)
    s = np.abs(pts - m).max() or 1.0
    q = (pts - m) / s
    x, y = q[:, 0], q[:, 1]
    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
        M = S1 + S2 @ T
        M = np.array([M[2] / 2, -M[1], M[0] / 2])
        w, v = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        return None
    v = np.real(v)
    cond = 4 * v[0] * v[2] - v[1] ** 2
    ok = np.nonzero(cond > 0)[0]
    if len(ok) == 0:
        return None
    a1 = v[:, ok[0]]
    A, B, C, D, E, F = np.r_[a1, T @ a1]

    den = B * B - 4 * A * C
    if abs(den) < 1e-12:
        return None
    x0 = (2 * C * D - B * E) / den
    y0 = (2 * A * E - B * D) / den
    F0 = A * x0 ** 2 + B * x0 * y0 + C * y0 ** 2 + D * x0 + E * y0 + F
    lam, vec = np.linalg.eigh(np.array([[A, B / 2], [B / 2, C]]))
    with np.errstate(invalid="ignore", divide="ignore"):
        axes = np.sqrt(-F0 / lam)
    if not np.all(np.isfinite(axes)):
        return None
    order = np.argsort(-axes)            # major axis first
    axes, vec = axes[order], vec[:, order]
    angle = float(np.arctan2(vec[1, 0], vec[0, 0]))
    center = np.array([x0, y0]) * s + m
    return center, (float(axes[0] * s), float(axes[1] * s)), angle


def ellipse_points(center, axes, angle, n: int = 200) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a, b = axes
    pts = np.column_stack([a * np.cos(t), b * np.sin(t)])
    return _close(_rotate(pts, angle, np.zeros(2)) + center)


def _mean_dist_to(pts: np.ndarray, outline: np.ndarray) -> float:
    return float(dist_to_polyline(pts, outline).mean())


# ---------------------------------------------------------------------------
# Polygon extraction and regularization
# ---------------------------------------------------------------------------
def _line_intersection(c1, d1, c2, d2):
    m = np.column_stack([d1, -d2])
    if abs(np.linalg.det(m)) < 1e-9:
        return None
    t = np.linalg.solve(m, c2 - c1)
    return c1 + t[0] * d1


def _segment_lines(pts, corners, closed):
    """Fit a line to the middle part of each run between corners."""
    n = len(pts)
    bounds = list(corners)
    if closed:
        bounds = bounds + [bounds[0] + n]
    else:
        bounds = [0] + bounds + [n - 1]
    lines = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        span = b - a
        trim = int(span * 0.15) if span > 8 else 0
        idx = np.arange(a + trim, b - trim + 1) % n
        if len(idx) < 2:
            idx = np.arange(a, b + 1) % n
        c, d, err = fit_line(pts[idx])
        # bend: angle between the first and second half of the edge
        # (only measured on edges long enough for it to be reliable)
        h = len(idx) // 2
        if h >= 12:
            d1, d2 = fit_line(pts[idx[:h]])[1], fit_line(pts[idx[h:]])[1]
            bend = float(np.degrees(np.arccos(min(1.0, abs(float(d1 @ d2))))))
        else:
            bend = 0.0
        lines.append((c, d, err, bend))
    return lines


def fit_polygon(pts: np.ndarray, corners: list[int], closed: bool):
    """Turn corner indices into clean vertices by intersecting fitted edge lines.

    Corners between nearly parallel edges are dropped. Returns
    (vertices, lines) or None.
    """
    corners = list(corners)
    while True:
        min_c = 3 if closed else 1
        if len(corners) < min_c:
            return None
        lines = _segment_lines(pts, corners, closed)
        m = len(lines)
        verts, drop = [], None
        pairs = range(m) if closed else range(1, m)
        for i in pairs:
            (c1, d1, *_), (c2, d2, *_) = lines[i - 1], lines[i]
            turn = np.degrees(np.arccos(min(1.0, abs(float(d1 @ d2)))))
            p = _line_intersection(c1, d1, c2, d2) if turn > 15 else None
            if p is None:
                drop = i if closed else i - 1
                break
            verts.append(p)
        if drop is None:
            break
        del corners[drop % len(corners)]

    verts = np.array(verts)
    if not closed:
        c0, d0, *_ = lines[0]
        cl, dl, *_ = lines[-1]
        start = c0 + ((pts[0] - c0) @ d0) * d0
        end = cl + ((pts[-1] - cl) @ dl) * dl
        verts = np.vstack([start, verts, end])
    # reject wild intersections far outside the drawing
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    margin = 0.25 * _diag(pts)
    if np.any(verts < lo - margin) or np.any(verts > hi + margin):
        return None
    return verts, lines


def _signed_area(v: np.ndarray) -> float:
    x, y = v[:, 0], v[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _regular_tol(n: int) -> float:
    # Triangles and quads are often drawn irregular on purpose, so demand
    # more evidence before forcing them to be regular. Irregular 5+-gons are
    # rarely intended, so be more forgiving there.
    return REGULAR_TOL_FEW if n <= 4 else REGULAR_TOL_MANY


def _regular_star_polygon(verts: np.ndarray):
    """Detect a regular polygon {n/1} or star polygon {n/k} (e.g. a pentagram).

    Works on the vertices in drawing order, so a pentagram drawn in one
    continuous stroke is recognized from its five tips. The test is holistic:
    build the best-matching ideal shape and accept it if the drawn vertices
    are, on average, close to the ideal ones.
    """
    n = len(verts)
    c = verts.mean(axis=0)
    rel = verts - c
    radii = np.linalg.norm(rel, axis=1)
    R = float(radii.mean())
    if R <= 0:
        return None
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    steps = np.angle(np.exp(1j * (np.roll(theta, -1) - theta)))
    step = steps.mean()
    k = int(round(abs(step) * n / (2 * np.pi)))
    if k < 1 or 2 * k >= n or np.gcd(n, k) != 1:
        return None
    exact = np.sign(step) * 2 * np.pi * k / n
    phase = float(np.angle(np.mean(np.exp(1j * (theta - exact * np.arange(n))))))
    t = phase + exact * np.arange(n)
    ideal = c + R * np.column_stack([np.cos(t), np.sin(t)])
    if np.linalg.norm(verts - ideal, axis=1).mean() / R > _regular_tol(n):
        return None
    # snap rotation so a vertex or an edge is axis-aligned when close
    phase = _snap_angle(phase, np.pi / n, offset=np.pi / 2)
    t = phase + exact * np.arange(n)
    return c + R * np.column_stack([np.cos(t), np.sin(t)]), k


def _outline_star(verts: np.ndarray):
    """Detect an outlined star: 2m vertices alternating outer/inner radius."""
    n = len(verts)
    if n < 8 or n % 2:
        return None
    c = verts.mean(axis=0)
    rel = verts - c
    radii = np.linalg.norm(rel, axis=1)
    r_even, r_odd = radii[0::2].mean(), radii[1::2].mean()
    outer_first = r_even >= r_odd
    r_out, r_in = max(r_even, r_odd), min(r_even, r_odd)
    if r_out <= 0 or r_in / r_out > 0.8:
        return None
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    steps = np.angle(np.exp(1j * (np.roll(theta, -1) - theta)))
    step = 2 * np.pi / n * np.sign(steps.mean())
    m = n // 2
    tips = theta[0::2] if outer_first else theta[1::2]
    phase = float(np.angle(np.mean(np.exp(1j * (tips - 2 * step * np.arange(m))))))
    start = phase if outer_first else phase - step

    def build(start_angle):
        t = start_angle + step * np.arange(n)
        r = np.where((np.arange(n) % 2 == 0) == outer_first, r_out, r_in)
        return c + r[:, None] * np.column_stack([np.cos(t), np.sin(t)])

    if np.linalg.norm(verts - build(start), axis=1).mean() / r_out > REGULAR_TOL_MANY:
        return None
    snapped = _snap_angle(phase, 2 * np.pi / m, offset=np.pi / 2)
    return build(start + (snapped - phase)), m


def _rectangle(verts: np.ndarray):
    """Regularize a quadrilateral with near-right angles into a rectangle."""
    edges = np.roll(verts, -1, axis=0) - verts
    ang = np.arctan2(edges[:, 1], edges[:, 0])
    interior = []
    for i in range(4):
        a, b = -edges[i - 1], edges[i]
        cosang = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        interior.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    if np.max(np.abs(np.array(interior) - 90)) > 15:
        return None
    theta = float(np.angle(np.mean(np.exp(4j * ang))) / 4)   # orientation mod 90
    theta = _snap_angle(theta, np.pi / 2)
    c = verts.mean(axis=0)
    local = _rotate(verts, -theta, c) - c
    xs, ys = np.sort(local[:, 0]), np.sort(local[:, 1])
    x0, x1 = xs[:2].mean(), xs[2:].mean()
    y0, y1 = ys[:2].mean(), ys[2:].mean()
    w, h = x1 - x0, y1 - y0
    square = abs(w - h) / max(w, h) < 0.12
    if square:
        mid_x, mid_y, half = (x0 + x1) / 2, (y0 + y1) / 2, (w + h) / 4
        x0, x1, y0, y1 = mid_x - half, mid_x + half, mid_y - half, mid_y + half
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    box = _rotate(box + c, theta, c)
    return box, square, theta


def _closed_polygon_shape(pts, verts, err, diag) -> Shape:
    n = len(verts)
    if _signed_area(verts) < 0:
        verts = verts[::-1]

    if n == 4:
        rect = _rectangle(verts)
        if rect is not None:
            box, square, theta = rect
            kind = "square" if square else "rectangle"
            return Shape(kind, _close(box), True,
                         {"angle_deg": float(np.degrees(theta))}, err)

    star = _outline_star(verts)
    if star is not None:
        out, m = star
        return Shape("star", _close(out), True, {"n": m}, err)

    reg = _regular_star_polygon(verts)
    if reg is not None:
        out, k = reg
        if k == 1:
            return Shape("regular_polygon", _close(out), True, {"n": n}, err)
        return Shape("star", _close(out), True, {"n": n, "k": k}, err)

    if n == 3:
        return Shape("triangle", _close(verts), True, {"n": 3}, err)
    return Shape("polygon", _close(verts), True, {"n": n}, err)


# ---------------------------------------------------------------------------
# Smooth-curve fallback
# ---------------------------------------------------------------------------
def smooth_curve(pts: np.ndarray, closed: bool, n_out: int = 300) -> np.ndarray:
    """Smoothing spline whose strength scales with the size of the stroke."""
    diag = _diag(pts)
    sigma = 0.006 * diag
    if closed:
        pts = _close(pts)
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=len(pts) * sigma ** 2,
                         per=int(closed), k=3)
        x, y = splev(np.linspace(0, 1, n_out), tck)
        return np.column_stack([x, y])
    except (ValueError, TypeError):
        return pts


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _line_shape(pts: np.ndarray, err: float) -> Shape:
    c, d, _ = fit_line(pts)
    t0, t1 = (pts[0] - c) @ d, (pts[-1] - c) @ d
    a, b = c + t0 * d, c + t1 * d
    mid = (a + b) / 2
    ang = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
    snapped = _snap_angle(ang, np.pi / 4)        # horizontal / vertical / 45 deg
    if snapped != ang:
        a, b = _rotate(np.array([a, b]), snapped - ang, mid)
    return Shape("line", np.array([a, b]), False,
                 {"angle_deg": float(np.degrees(snapped))}, err)


def _arc_shape(pts: np.ndarray, center, r, err: float) -> Shape:
    theta = np.unwrap(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
    t = np.linspace(theta[0], theta[-1], 200)
    out = center + r * np.column_stack([np.cos(t), np.sin(t)])
    return Shape("arc", out, False,
                 {"center": center.tolist(), "radius": r,
                  "sweep_deg": float(np.degrees(theta[-1] - theta[0]))}, err)


def _ellipse_shape(pts: np.ndarray, ell, err: float) -> Shape:
    center, (a, b), angle = ell
    if b / a >= CIRCLE_AXIS_RATIO:
        c, r, cerr = fit_circle(pts)
        return Shape("circle", ellipse_points(c, (r, r), 0.0), True,
                     {"center": c.tolist(), "radius": r}, cerr / _diag(pts))
    angle = _snap_angle(angle, np.pi / 2)
    return Shape("ellipse", ellipse_points(center, (a, b), angle), True,
                 {"center": center.tolist(), "axes": [a, b],
                  "angle_deg": float(np.degrees(angle))}, err)


def classify(points: np.ndarray, closed: bool | None = None) -> Shape:
    """Recognize and regularize a single stroke."""
    pts = _dedupe(points)
    if len(pts) < 2 or path_length(pts) < 1e-9:
        return _dot(pts)
    if closed is None:
        pts, closed = detect_closure(pts)
    diag = _diag(pts)
    L = path_length(pts)
    rs = resample(pts, RESAMPLE_N, closed=closed)

    if not closed:
        return _classify_open(rs, diag, L)
    return _classify_closed(rs, diag)


def _edges_straight(verts, lines, closed: bool) -> bool:
    """Every edge's points must lie close to a straight line.

    ``lines[i]`` is the fit for the edge that runs from ``verts[i]`` to the
    next vertex. This rejects curvy shapes that a polygon happens to
    approximate (e.g. a blob with a few lobes).
    """
    nxt = np.roll(verts, -1, axis=0) if closed else verts[1:]
    for (_, _, e, bend), a, b in zip(lines, verts, nxt):
        if e / max(np.linalg.norm(b - a), 1e-9) > 2 * LINE_TOL:
            return False
        if closed and bend > MAX_EDGE_BEND_DEG:
            return False
    return True


def _classify_open(rs, diag, L) -> Shape:
    _, _, lerr = fit_line(rs)
    chord = np.linalg.norm(rs[-1] - rs[0])
    if lerr / max(chord, 1e-9) < LINE_TOL:
        return _line_shape(rs, lerr / max(chord, 1e-9))

    corners = find_corners(rs, closed=False)
    if corners:
        poly = fit_polygon(rs, corners, closed=False)
        if poly is not None:
            verts, lines = poly
            err = _mean_dist_to(rs, verts) / diag
            if _edges_straight(verts, lines, closed=False) and err < POLY_TOL:
                return Shape("polyline", verts, False,
                             {"n_segments": len(verts) - 1}, err)

    c, r, cerr = fit_circle(rs)
    if cerr / r < ARC_TOL and r < 5 * diag:
        return _arc_shape(rs, c, r, cerr / r)

    return Shape("curve", smooth_curve(rs, closed=False), False)


def _classify_closed(rs, diag) -> Shape:
    ell = fit_ellipse(rs)
    e_err = np.inf
    if ell is not None and ell[1][1] > 1e-6:
        e_err = _mean_dist_to(rs, ellipse_points(*ell, n=360)) / diag

    corners = find_corners(rs, closed=True)
    p_err, poly = np.inf, None
    if len(corners) >= 3:
        poly = fit_polygon(rs, corners, closed=True)
        if poly is not None and _edges_straight(poly[0], poly[1], closed=True):
            p_err = _mean_dist_to(rs, _close(poly[0])) / diag
        else:
            poly = None

    # A polygon needs real corners *and* has to beat the ellipse clearly.
    if poly is not None and p_err < POLY_TOL and p_err < 0.8 * e_err:
        return _closed_polygon_shape(rs, poly[0], p_err, diag)
    if e_err < ELLIPSE_TOL:
        return _ellipse_shape(rs, ell, e_err)
    if poly is not None and p_err < 1.5 * POLY_TOL:
        return _closed_polygon_shape(rs, poly[0], p_err, diag)
    return Shape("closed_curve", smooth_curve(rs, closed=True), True)


def _dot(pts: np.ndarray) -> Shape:
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts) == 0:
        return Shape("dot", pts, False)
    c = pts.mean(axis=0)
    r = 0.5 * float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    return Shape("dot", c[None, :], False, {"center": c.tolist(), "radius": r})


def _extent(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) if len(pts) else 0.0


def correct_drawing(strokes: list[np.ndarray], merge: bool = True,
                    dot_size: float | None = None) -> list[Shape]:
    """Full pipeline: separate dots, merge/close strokes, recognize each one.

    Shape tolerances are relative to each stroke's own size, so a tiny
    scribble would otherwise be fitted like a tiny line. A stroke counts as a
    dot when its extent is below ``dot_size`` (in input units, e.g. a few
    pixels for a canvas) or below ``DOT_FRAC`` of the whole drawing.
    """
    strokes = [p for p in (_dedupe(s) for s in strokes) if len(p) >= 1]
    if not strokes:
        return []
    overall = _extent(np.vstack(strokes))
    threshold = max(dot_size or 0.0, DOT_FRAC * overall)

    dots, rest = [], []
    for p in strokes:
        if len(p) < 2 or _extent(p) <= threshold:
            dots.append(_dot(p))
        else:
            rest.append(p)

    if merge:
        items = merge_strokes(rest)
    else:
        items = [detect_closure(p) for p in rest]
    return [classify(p, closed=c) for p, c in items] + dots


# ---------------------------------------------------------------------------
# I/O helpers (CSV format used by the original hackathon data)
# ---------------------------------------------------------------------------
def read_csv(source) -> list[np.ndarray]:
    """Read ``path_id, segment_id, x, y`` rows (header optional).

    Each (path_id, segment_id) pair becomes one stroke.
    """
    import pandas as pd

    df = pd.read_csv(source, header=None)
    if not np.issubdtype(df.dtypes.iloc[0], np.number):
        df = df.iloc[1:].astype(float)      # first row was a header
    df = df.iloc[:, :4]
    df.columns = ["path", "seg", "x", "y"]
    strokes = []
    for _, g in df.groupby(["path", "seg"], sort=True):
        strokes.append(g[["x", "y"]].to_numpy(float))
    return strokes


def shapes_to_csv(shapes: list[Shape]) -> str:
    lines = []
    for i, sh in enumerate(shapes):
        for x, y in sh.points:
            lines.append(f"{i},0,{x:.6f},{y:.6f}")
    return "\n".join(lines) + "\n"


def shapes_to_svg(shapes: list[Shape], flip_y: bool = False, stroke="#1f6feb") -> str:
    allp = np.vstack([s.points for s in shapes]) if shapes else np.zeros((1, 2))
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    pad = 0.05 * float(np.max(hi - lo) or 1.0)
    w, h = (hi - lo) + 2 * pad
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.2f} {h:.2f}" '
           f'width="{w:.0f}" height="{h:.0f}">']
    sw = max(w, h) / 250
    for sh in shapes:
        p = sh.points - lo + pad
        if flip_y:
            p[:, 1] = h - p[:, 1]
        if sh.kind == "dot":
            out.append(f'<circle cx="{p[0, 0]:.2f}" cy="{p[0, 1]:.2f}" '
                       f'r="{2 * sw:.2f}" fill="{stroke}"/>')
            continue
        d = " ".join(f"{x:.2f},{y:.2f}" for x, y in p)
        tag = "polygon" if sh.closed else "polyline"
        pts = d if not sh.closed else " ".join(d.split()[:-1])
        out.append(f'<{tag} points="{pts}" fill="none" stroke="{stroke}" '
                   f'stroke-width="{sw:.2f}" stroke-linejoin="round"/>')
    out.append("</svg>")
    return "\n".join(out)
