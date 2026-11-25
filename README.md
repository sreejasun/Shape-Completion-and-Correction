# Shape Completion and Correction

Draw rough shapes by hand and get clean, regularized geometry back. The app
recognizes what you meant (circle, square, star, ...), completes gaps,
and snaps the result to tidy proportions and orientations.

![Shape Correction app](examples/app_screenshot.png)

## What it recognizes

| Drawn stroke | Corrected to |
| --- | --- |
| tiny tap or scribble | **dot** |
| roughly straight | **line** (snapped to horizontal / vertical / 45° when close) |
| open, evenly curved | **arc** of a circle |
| open with sharp turns | **polyline** with exact corners |
| closed, round | **circle** or **ellipse** (axes snapped when close) |
| closed, 4 right-ish angles | **rectangle** or **square** |
| closed, 3 corners | **triangle** (equilateral only if it really looks equilateral) |
| closed, 5+ near-equal sides | **regular polygon** (pentagon, hexagon, ...) |
| outlined star or one-stroke pentagram | **regular star** |
| anything else | smoothed **curve** (left as-is in spirit, just de-jittered) |

**Completion:**
- Strokes whose ends nearly meet are closed. This works for both a small gap and an overshoot past the start.
- Separate strokes whose endpoints touch are joined, so a square drawn as four lines becomes one square.

## How it works

`shape_correction.py` holds the engine. It has no UI code, so you can use it on its own.

1. **Resample** each stroke to evenly spaced points.
2. **Closure detection** finds the best join between the start and end, so a
   pentagram's self-crossing isn't mistaken for the closing point.
3. **Corner detection** measures the turning angle over a sliding window on a
   lightly smoothed copy, then applies non-maximum suppression.
4. **Model fitting.** Candidates are a total-least-squares line, a geometric
   circle fit, a direct ellipse fit (Fitzgibbon / Halir–Flusser), and a polygon
   whose vertices are the *intersections of lines fitted to each edge*. Using
   intersections rather than the raw corner points gives sharp corners even
   when the drawn corners are rounded.
5. **Model selection.** The simplest model that explains the points within a
   size-relative tolerance wins. A polygon must have straight edges (low
   residual and low bending) and must beat the ellipse clearly.
6. **Regularization**: rectangles become axis-aligned and squares get equal
   sides, near-regular polygons and stars become exactly regular, and
   orientations snap to "nice" angles.

All tolerances are relative to the stroke's size and are listed at the top of
`shape_correction.py`.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

- **Draw** mode: sketch on the canvas and the corrected version appears beside it.
- **Upload CSV** mode: rows of `path_id, segment_id, x, y` (header optional).
  Try `examples/sample_drawing.csv`.

You can download the result as CSV or SVG.

### Use the engine directly

```python
import shape_correction as sc

shapes = sc.correct_drawing(strokes)   # strokes: list of (N, 2) arrays
for s in shapes:
    print(s.label, s.params)           # e.g. "hexagon", {"n": 6}
    s.points                           # clean outline as an (M, 2) array
```

## Tests

```bash
pytest -q
```

The tests generate synthetic "hand-drawn" shapes with wobble, jitter, tilt,
gaps and overshoot, then check recognition across many random seeds. They
also check that irregular shapes are **not** over-corrected, for example that
a scalene triangle stays scalene and a free-form blob stays a curve.
