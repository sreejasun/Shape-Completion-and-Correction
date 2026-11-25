"""Streamlit app: draw (or upload) rough shapes and get them auto-corrected.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np
import streamlit as st

import shape_correction as sc

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:  # component missing or incompatible with this Streamlit
    st_canvas = None

CANVAS_W, CANVAS_H = 520, 380
EXAMPLE_CSV = Path(__file__).parent / "examples" / "sample_drawing.csv"
PALETTE = ["#5B6B94", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444",
           "#8B5CF6", "#EC4899", "#14B8A6"]

CSS = """
<style>
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stSidebarCollapsedControl"] {
  display: none !important;
}
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }

/* header */
.app-header { display:flex; align-items:center; gap:14px; }
.app-logo { width:44px; height:44px; flex:none; display:block; }
.app-title { font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em;
             color:#0F172A; line-height:1.15; margin:0; }
.app-sub { color:#64748B; font-size: .93rem; margin: 2px 0 0; }

/* cards: containers created with key="card_..." */
[class*="st-key-card_"] {
  background:#FFFFFF; border:1px solid #E5E7EB; border-radius:14px;
  padding: 18px 20px 16px !important;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 6px 18px rgba(15,23,42,.05);
}
.card-head { display:flex; justify-content:space-between; align-items:baseline; gap:12px; }
.card-title { font-weight:600; font-size:1rem; color:#0F172A; }
.card-hint  { color:#94A3B8; font-size:.8rem; text-align:right; }

/* result surface */
.surface { border:1px solid #EEF0F4; border-radius:10px; background:#FFFFFF;
           overflow:hidden; aspect-ratio: 520 / 380; }
.surface img { width:100%; height:100%; display:block; }
.surface.empty-state { display:flex; align-items:center; justify-content:center;
           border-style:dashed; border-color:#E2E8F0; background:#FBFCFE; }
.empty { color:#94A3B8; text-align:center; font-size:.9rem; padding: 10px; }
.empty b { color:#475569; display:block; font-size:1rem; margin-bottom:4px; }

/* stats */
.stats { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:14px; }
.stat { background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:14px 18px;
        box-shadow: 0 1px 2px rgba(15,23,42,.04); }
.stat .k { color:#64748B; font-size:.72rem; text-transform:uppercase;
           letter-spacing:.07em; font-weight:600; }
.stat .v { color:#0F172A; font-size:1.6rem; font-weight:700; margin-top:2px;
           font-variant-numeric: tabular-nums; }
@media (max-width: 800px) { .stats { grid-template-columns: repeat(2, 1fr); } }

/* detected shapes list */
.shape-row { display:flex; align-items:center; gap:10px; padding:10px 2px;
             border-bottom:1px solid #F1F5F9; }
.shape-row:last-child { border-bottom:none; }
.dot { width:10px; height:10px; border-radius:50%; flex:none; }
.shape-name { font-weight:600; color:#0F172A; flex:1; font-size:.95rem; }
.shape-name::first-letter { text-transform: uppercase; }
.pill { font-size:.72rem; font-weight:600; padding:3px 9px; border-radius:999px;
        background:#F1F5F9; color:#475569; white-space:nowrap; }
.pill.good { background:#ECFDF5; color:#047857; }
.pill.ok   { background:#FFFBEB; color:#B45309; }
</style>
"""

LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 44 44">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#5B6B94"/><stop offset="1" stop-color="#A3B1D1"/></linearGradient></defs>
<rect width="44" height="44" rx="12" fill="url(#g)"/>
<g fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<path d="M9 31c1.5-5 4-8.5 7-9.5" opacity=".55"/>
<rect x="22" y="10" width="12" height="12" rx="1.5"/><circle cx="16" cy="29" r="6"/></g></svg>"""


def _img(svg: str, cls: str = "") -> str:
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<img class="{cls}" src="data:image/svg+xml;base64,{b64}" alt="">'


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def canvas_strokes(json_data: dict | None) -> list[np.ndarray]:
    """Extract free-hand strokes from the Fabric.js JSON of the canvas."""
    strokes = []
    if not json_data:
        return strokes
    for obj in json_data.get("objects", []):
        if obj.get("type", "").lower() != "path":
            continue
        pts = []
        for cmd in obj.get("path", []):
            # M x y | L x y | Q cx cy x y  -> keep every coordinate pair
            coords = cmd[1:]
            pts.extend(zip(coords[0::2], coords[1::2]))
        if len(pts) >= 2:
            strokes.append(np.array(pts, dtype=float))
    return strokes


def fit_view(arrays, width=CANVAS_W, height=CANVAS_H, flip_y=False):
    """Map data coordinates into a width x height viewport (keeps aspect)."""
    allp = np.vstack(arrays)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scale = 0.9 * min(width / span[0], height / span[1])
    off = np.array([width, height]) / 2 - (lo + hi) / 2 * scale

    def tf(p):
        q = p * scale + off
        if flip_y:
            q[:, 1] = height - q[:, 1]
        return q
    return tf


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _pts(points) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def render_svg(strokes, shapes, tf=None, show_original=True, show_labels=True,
               draw_shapes=True) -> str:
    tf = tf or (lambda p: p)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
           f'font-family="Inter, Helvetica, Arial, sans-serif">'
           f'<rect width="100%" height="100%" fill="#FFFFFF"/>']
    if show_original or not draw_shapes:
        color, width = ("#CBD5E1", 1.5) if draw_shapes else ("#1E293B", 2.5)
        for s in strokes:
            out.append(f'<polyline points="{_pts(tf(s))}" fill="none" stroke="{color}" '
                       f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>')
    if draw_shapes:
        for i, sh in enumerate(shapes):
            c = PALETTE[i % len(PALETTE)]
            p = tf(sh.points)
            if sh.kind == "dot":
                out.append(f'<circle cx="{p[0, 0]:.1f}" cy="{p[0, 1]:.1f}" r="5" fill="{c}"/>')
                continue
            style = (f'stroke="{c}" stroke-width="3" stroke-linecap="round" '
                     f'stroke-linejoin="round"')
            if sh.closed:
                p = p[:-1] if len(p) > 2 else p
                out.append(f'<polygon points="{_pts(p)}" fill="{c}" fill-opacity="0.07" {style}/>')
            else:
                out.append(f'<polyline points="{_pts(p)}" fill="none" {style}/>')
        if show_labels:
            for i, sh in enumerate(shapes):
                c = PALETTE[i % len(PALETTE)]
                cx, cy = tf(sh.points).mean(axis=0)
                if sh.kind == "dot":
                    cy -= 20                      # label above the dot, not on it
                text = html.escape(sh.label)
                w = 6.6 * len(text) + 18
                out.append(
                    f'<g transform="translate({cx:.1f},{cy:.1f})">'
                    f'<rect x="{-w / 2:.1f}" y="-11" width="{w:.1f}" height="22" rx="11" '
                    f'fill="#FFFFFF" stroke="{c}" stroke-width="1.2"/>'
                    f'<text y="4" text-anchor="middle" font-size="12" font-weight="600" '
                    f'fill="{c}">{text}</text></g>')
    out.append("</svg>")
    return "".join(out)


def card_header(title, hint=""):
    st.html(f'<div class="card-head"><span class="card-title">{title}</span>'
            f'<span class="card-hint">{hint}</span></div>')


def surface(svg: str | None, empty_title="", empty_text=""):
    if svg is None:
        st.html(f'<div class="surface empty-state"><div class="empty">'
                f'<b>{empty_title}</b>{empty_text}</div></div>')
    else:
        st.html(f'<div class="surface">{_img(svg)}</div>')


def stats_html(strokes, shapes):
    closed = sum(s.closed for s in shapes)
    regular = sum(s.kind in ("circle", "square", "regular_polygon", "star")
                  for s in shapes)
    tiles = [("Strokes", len(strokes)), ("Shapes", len(shapes)),
             ("Closed", closed), ("Regularized", regular)]
    return '<div class="stats">' + "".join(
        f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in tiles) + "</div>"


def shapes_list_html(shapes):
    rows = []
    for i, sh in enumerate(shapes):
        c = PALETTE[i % len(PALETTE)]
        if sh.kind in ("curve", "closed_curve"):
            fit = '<span class="pill">smoothed</span>'
        elif sh.kind == "dot":
            fit = ""
        else:
            dev = 100 * sh.error
            cls = "good" if dev < 1.5 else "ok"
            fit = f'<span class="pill {cls}">{dev:.1f}% deviation</span>'
        rows.append(f'<div class="shape-row"><span class="dot" style="background:{c}">'
                    f'</span><span class="shape-name">{html.escape(sh.label)}</span>'
                    f'<span class="pill">{"point" if sh.kind == "dot" else "closed" if sh.closed else "open"}</span>'
                    f'{fit}</div>')
    return "".join(rows)


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------
def header():
    """Title on the left; mode switch, settings and info on the right."""
    title_col, ctrl_col = st.columns([4, 3], vertical_alignment="center")
    with title_col:
        st.html(f'<div class="app-header">{_img(LOGO_SVG, "app-logo")}<div>'
                f'<p class="app-title">Shape Correction</p>'
                f'<p class="app-sub">Sketch rough shapes. Get clean, regular geometry back.'
                f'</p></div></div>')
    with ctrl_col:
        mode_col, set_col, info_col = st.columns([4, 3, 3], vertical_alignment="center")
        modes = ["Draw", "Upload CSV"] if st_canvas else ["Upload CSV"]
        with mode_col:
            mode = st.segmented_control("Input", modes, default=modes[0], key="mode",
                                        label_visibility="collapsed") or modes[0]
        with set_col.popover("Settings", icon=":material/tune:", use_container_width=True):
            brush = st.slider("Brush width", 1, 8, 3)
            merge = st.toggle("Join strokes whose ends touch", value=True,
                              help="A square drawn as four separate lines becomes one square.")
            show_original = st.toggle("Show original underneath", value=True)
            show_labels = st.toggle("Show labels", value=True)
        with info_col.popover("About", icon=":material/info:", use_container_width=True):
            st.markdown(
                "**Recognizes** lines, arcs, polylines, circles, ellipses, triangles, "
                "rectangles, squares, regular polygons and stars. Anything else is "
                "smoothed.\n\n"
                "**How it works**\n"
                "1. Strokes are resampled; nearly closed ones are closed.\n"
                "2. Sharp corners are found from turning angles.\n"
                "3. Lines, circles, ellipses and polygons are fitted; the simplest "
                "model that fits wins.\n"
                "4. Results are regularized: right angles, equal sides, snapped "
                "orientations.")
    if st_canvas is None:
        st.warning("Install `streamlit-drawable-canvas` to enable drawing.")
    return mode, brush, merge, show_original, show_labels


def input_card(mode, brush):
    """Returns (strokes, flip_y, transform for display, dot size in pixels)."""
    strokes, flip_y, tf = [], False, None
    with st.container(key="card_input"):
        if mode == "Draw":
            card_header("Sketch", "Each pen stroke is a shape or part of one")
            result = st_canvas(stroke_width=brush, stroke_color="#1E293B",
                               background_color="#FFFFFF", drawing_mode="freedraw",
                               height=CANVAS_H, width=CANVAS_W, key="canvas")
            return canvas_strokes(result.json_data), False, None, max(10.0, 3.0 * brush)

        card_header("Input file", "CSV rows: path_id, segment_id, x, y")
        up = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
        source = up
        if up is None and st.session_state.get("use_example"):
            source = EXAMPLE_CSV
        if source is not None:
            try:
                strokes = sc.read_csv(source)
            except Exception as e:  # malformed file
                st.error(f"Could not read the CSV: {e}")
        if strokes:
            flip_y = True                              # CSV data is y-up
            tf = fit_view(strokes, flip_y=True)
            surface(render_svg(strokes, [], tf, draw_shapes=False))
        elif st.button("Load example drawing", icon=":material/draw:"):
            st.session_state.use_example = True
            st.rerun()
    return strokes, flip_y, tf, None


def result_card(strokes, shapes, tf, show_original, show_labels, mode):
    with st.container(key="card_result"):
        card_header("Corrected", "Updates as you draw" if mode == "Draw" else "")
        if shapes:
            surface(render_svg(strokes, shapes, tf, show_original, show_labels))
        else:
            surface(None, "Nothing to correct yet",
                    "Draw a circle, square, star or line on the left."
                    if mode == "Draw" else "Upload a CSV or load the example.")


def details_section(strokes, shapes, flip_y):
    st.html(stats_html(strokes, shapes))
    left, right = st.columns([2, 1], gap="medium")
    with left, st.container(key="card_shapes"):
        card_header("Detected shapes", f"{len(shapes)} found")
        if shapes:
            st.html(shapes_list_html(shapes))
        else:
            st.caption("Shapes you draw will be listed here.")
    with right, st.container(key="card_export"):
        card_header("Export", "Clean geometry")
        disabled = not shapes
        st.download_button("Download SVG",
                           sc.shapes_to_svg(shapes, flip_y=flip_y) if shapes else "",
                           file_name="corrected.svg", mime="image/svg+xml",
                           type="primary", disabled=disabled, icon=":material/download:",
                           width="stretch")
        st.download_button("Download CSV", sc.shapes_to_csv(shapes) if shapes else "",
                           file_name="corrected.csv", mime="text/csv",
                           disabled=disabled, icon=":material/table:", width="stretch")
        st.caption("CSV rows: `shape_id, 0, x, y`")


def main():
    st.set_page_config(page_title="Shape Correction", page_icon="◆", layout="wide",
                       initial_sidebar_state="collapsed")
    st.html(CSS)
    mode, brush, merge, show_original, show_labels = header()

    in_col, out_col = st.columns(2, gap="medium")
    with in_col:
        strokes, flip_y, tf, dot_size = input_card(mode, brush)
    shapes = sc.correct_drawing(strokes, merge=merge, dot_size=dot_size) if strokes else []
    with out_col:
        result_card(strokes, shapes, tf, show_original, show_labels, mode)

    details_section(strokes, shapes, flip_y)


if __name__ == "__main__":
    main()
