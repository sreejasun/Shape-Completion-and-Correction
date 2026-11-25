"""Tests on synthetic hand-drawn strokes (wobble, jitter, gaps, overshoot)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import shape_correction as sc  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic "hand-drawn" stroke generator
# ---------------------------------------------------------------------------
def _ideal(kind, rng):
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    if kind == "circle":
        return np.column_stack([np.cos(t), np.sin(t)]) * 100, True
    if kind == "ellipse":
        return np.column_stack([150 * np.cos(t), 70 * np.sin(t)]), True
    if kind == "square":
        v = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], float)
        return v, True
    if kind == "rectangle":
        v = np.array([[0, 0], [300, 0], [300, 140], [0, 140]], float)
        return v, True
    if kind == "triangle":
        v = np.array([[0, 0], [260, 0], [80, 180]], float)
        return v, True
    if kind in ("pentagon", "hexagon"):
        n = 5 if kind == "pentagon" else 6
        a = np.pi / 2 + 2 * np.pi * np.arange(n) / n
        return 120 * np.column_stack([np.cos(a), np.sin(a)]), True
    if kind == "star":                       # outlined 5-point star
        a = np.pi / 2 + np.pi * np.arange(10) / 5
        r = np.where(np.arange(10) % 2 == 0, 150, 60)
        return r[:, None] * np.column_stack([np.cos(a), np.sin(a)]), True
    if kind == "pentagram":                  # one-stroke, self-crossing star
        a = np.pi / 2 + 4 * np.pi * np.arange(5) / 5
        return 150 * np.column_stack([np.cos(a), np.sin(a)]), True
    if kind == "line":
        return np.array([[0, 0], [300, 0]], float), False
    if kind == "arc":
        a = np.linspace(0, np.radians(150), 100)
        return 120 * np.column_stack([np.cos(a), np.sin(a)]), False
    if kind == "polyline":
        return np.array([[0, 0], [200, 0], [200, 150]], float), False
    raise ValueError(kind)


def hand_drawn(kind, seed, gap=0.0, overshoot=0.0, wobble=0.012, jitter=0.003,
               tilt_deg=4.0):
    rng = np.random.default_rng(seed)
    pts, closed = _ideal(kind, rng)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    pts = sc.resample(pts, 300)
    if closed and overshoot:
        extra = int(300 * overshoot)
        pts = np.vstack([pts, pts[1:extra + 1]])
    if closed and gap:
        pts = pts[: int(len(pts) * (1 - gap))]
    size = sc._diag(pts)
    n = len(pts)
    # low-frequency wobble along the normal + small jitter
    freq = rng.uniform(2, 4)
    phase = rng.uniform(0, 2 * np.pi)
    tang = np.gradient(pts, axis=0)
    norm = np.column_stack([-tang[:, 1], tang[:, 0]])
    norm /= np.linalg.norm(norm, axis=1, keepdims=True) + 1e-12
    w = wobble * size * np.sin(freq * 2 * np.pi * np.arange(n) / n + phase)
    pts = pts + norm * w[:, None] + rng.normal(0, jitter * size, pts.shape)
    tilt = np.radians(rng.uniform(-tilt_deg, tilt_deg))
    return sc._rotate(pts, tilt, pts.mean(axis=0))


EXPECTED = {
    "circle": "circle",
    "ellipse": "ellipse",
    "square": "square",
    "rectangle": "rectangle",
    "triangle": "triangle",
    "pentagon": "regular_polygon",
    "hexagon": "regular_polygon",
    "star": "star",
    "pentagram": "star",
    "line": "line",
    "arc": "arc",
    "polyline": "polyline",
}


@pytest.mark.parametrize("kind", list(EXPECTED))
@pytest.mark.parametrize("seed", range(10))
def test_recognizes_hand_drawn(kind, seed):
    shape = sc.classify(hand_drawn(kind, seed))
    assert shape.kind == EXPECTED[kind], shape.label


@pytest.mark.parametrize("kind", ["circle", "square", "hexagon", "star"])
@pytest.mark.parametrize("seed", range(5))
def test_completes_gaps_and_overshoot(kind, seed):
    gap = sc.classify(hand_drawn(kind, seed, gap=0.06))
    over = sc.classify(hand_drawn(kind, seed + 100, overshoot=0.08))
    assert gap.closed and gap.kind == EXPECTED[kind]
    assert over.closed and over.kind == EXPECTED[kind]


def test_regular_polygon_vertex_count():
    assert sc.classify(hand_drawn("hexagon", 1)).params["n"] == 6
    assert sc.classify(hand_drawn("pentagon", 1)).params["n"] == 5


def test_near_horizontal_line_is_snapped():
    s = sc.classify(hand_drawn("line", 0, tilt_deg=4))
    assert s.params["angle_deg"] % 180 == pytest.approx(0, abs=1e-6)


def test_rectangle_is_axis_aligned_and_sized():
    s = sc.classify(hand_drawn("rectangle", 3))
    edges = np.diff(s.points, axis=0)
    # every edge exactly horizontal or vertical
    assert np.all(np.min(np.abs(edges), axis=1) < 1e-6)
    w, h = np.ptp(s.points, axis=0)
    assert w / h == pytest.approx(300 / 140, rel=0.1)


def test_square_from_four_separate_strokes():
    rng = np.random.default_rng(0)
    corners = np.array([[0, 0], [200, 0], [200, 200], [0, 200], [0, 0]], float)
    strokes = []
    for a, b in zip(corners[:-1], corners[1:]):
        seg = np.linspace(a, b, 40) + rng.normal(0, 1.0, (40, 2))
        strokes.append(seg + rng.normal(0, 4, 2))      # ends don't quite meet
    shapes = sc.correct_drawing(strokes)
    assert len(shapes) == 1
    assert shapes[0].kind == "square"


def test_separate_shapes_are_not_merged():
    a = hand_drawn("circle", 0)
    b = hand_drawn("square", 0) + 500
    shapes = sc.correct_drawing([a, b])
    assert sorted(s.kind for s in shapes) == ["circle", "square"]


def test_csv_roundtrip(tmp_path):
    p = tmp_path / "in.csv"
    pts = hand_drawn("circle", 0)
    p.write_text("\n".join(f"0,0,{x},{y}" for x, y in pts))
    strokes = sc.read_csv(str(p))
    assert len(strokes) == 1 and len(strokes[0]) == len(pts)   # no row lost
    shapes = sc.correct_drawing(strokes)
    text = sc.shapes_to_csv(shapes)
    assert text.count("\n") == len(shapes[0].points)


def test_degenerate_input_does_not_crash():
    assert sc.classify(np.array([[1.0, 1.0]])).kind == "dot"
    assert sc.classify(np.array([[1.0, 1.0], [1.0, 1.0]])).kind == "dot"
    assert sc.correct_drawing([]) == []


# ---------------------------------------------------------------------------
# Things that must NOT be over-corrected
# ---------------------------------------------------------------------------
def _wobbly(pts, seed, closed=True):
    rng = np.random.default_rng(seed)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    pts = sc.resample(np.asarray(pts, float), 300)
    return pts + rng.normal(0, 0.003 * sc._diag(pts), pts.shape)


@pytest.mark.parametrize("seed", range(5))
def test_irregular_pentagon_stays_irregular(seed):
    v = [[0, 0], [300, 0], [340, 120], [120, 260], [-40, 150]]
    s = sc.classify(_wobbly(v, seed))
    assert s.kind == "polygon" and s.params["n"] == 5


@pytest.mark.parametrize("seed", range(5))
def test_scalene_triangle_not_made_equilateral(seed):
    s = sc.classify(_wobbly([[0, 0], [300, 0], [40, 120]], seed))
    assert s.kind == "triangle"


@pytest.mark.parametrize("seed", range(5))
def test_freeform_blob_stays_smooth_curve(seed):
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    r = 100 + 35 * np.sin(3 * t) + 15 * np.cos(5 * t)
    s = sc.classify(_wobbly(np.column_stack([r * np.cos(t), r * np.sin(t)]), seed))
    assert s.kind == "closed_curve"


def test_s_curve_stays_curve():
    t = np.linspace(0, 2 * np.pi, 200)
    s = sc.classify(_wobbly(np.column_stack([t * 50, 60 * np.sin(t)]), 0, closed=False))
    assert s.kind == "curve"


# ---------------------------------------------------------------------------
# Dots
# ---------------------------------------------------------------------------
def _scribble_dot(center, seed, size=5.0, n=12):
    """A short wobbly mouse tap, a few pixels across."""
    rng = np.random.default_rng(seed)
    return np.asarray(center, float) + rng.uniform(-size / 2, size / 2, (n, 2))


@pytest.mark.parametrize("seed", range(5))
def test_lone_dot_on_canvas_is_a_dot(seed):
    shapes = sc.correct_drawing([_scribble_dot([200, 150], seed)], dot_size=10)
    assert [s.kind for s in shapes] == ["dot"]
    assert shapes[0].points.shape == (1, 2)


def test_short_straight_dash_on_canvas_is_a_dot():
    dash = np.column_stack([np.linspace(100, 106, 8), np.full(8, 50.0)])
    assert sc.correct_drawing([dash], dot_size=10)[0].kind == "dot"


def test_dot_next_to_shapes_without_absolute_size():
    # CSV-style input: no pixel size known; dot is tiny relative to drawing
    circle = hand_drawn("circle", 0)
    dot = _scribble_dot(circle.mean(axis=0) + [300, 0], 0, size=3)
    kinds = sorted(s.kind for s in sc.correct_drawing([circle, dot]))
    assert kinds == ["circle", "dot"]


def test_dot_is_not_merged_into_a_nearby_line():
    line = np.column_stack([np.linspace(0, 300, 60), np.zeros(60)])
    dot = _scribble_dot([304, 0], 1, size=4)
    kinds = sorted(s.kind for s in sc.correct_drawing([line, dot], dot_size=10))
    assert kinds == ["dot", "line"]


def test_small_real_shapes_are_not_dots():
    # a 40px circle on a canvas is a shape, not a dot
    t = np.linspace(0, 2 * np.pi, 80)
    small = np.column_stack([20 * np.cos(t), 20 * np.sin(t)]) + 100
    assert sc.correct_drawing([small], dot_size=10)[0].kind == "circle"
