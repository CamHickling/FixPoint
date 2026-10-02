"""One constant gaze shift for the whole session, estimated from both calibration blocks.

The shift is measured in raw scene-camera pixels - the same space as Neon's own
offset correction - so it can be (A) entered in Pupil Cloud / the Companion app, or
(B) subtracted from gaze.csv directly. It is checked on targets it was not fitted to,
and flagged when it should not be trusted.
"""
import numpy as np
import pandas as pd

from .analysis import _angles
from .recording import Recording

MAX_BLOCK_DISAGREEMENT_DEG = 0.5  # start vs end shift; more suggests the headset moved
MIN_DIRECTION_AGREEMENT = 2 / 3  # share of targets that must be shifted the same way
MIN_SHIFT_DEG = 0.1  # below this an axis is effectively unbiased


def _accuracy(samples: pd.DataFrame, camera, shift_px=(0.0, 0.0)) -> tuple[float, float, float]:
    """(accuracy, median h offset, median v offset) in degrees after subtracting shift_px."""
    gaze = samples[["gaze_x_px", "gaze_y_px"]].to_numpy() - np.asarray(shift_px)
    g = _angles(camera.rays(camera.undistort(gaze)))
    t = _angles(camera.rays(camera.undistort(samples[["target_scene_x_px", "target_scene_y_px"]].to_numpy())))
    h, v = np.median(g - t, axis=0)
    return float(np.hypot(h, v)), float(h), float(v)


def estimate(metrics: pd.DataFrame, samples: pd.DataFrame, rec: Recording):
    """Returns (summary dict, per-target table)."""
    usable = metrics[metrics.usable][["block", "target"]]
    if usable.empty or samples.empty:
        return {"recommended": False, "flags": ["No usable calibration targets."]}, pd.DataFrame()
    s = samples.merge(usable, on=["block", "target"])
    s = s.assign(dx=s.gaze_x_px - s.target_scene_x_px, dy=s.gaze_y_px - s.target_scene_y_px)
    per = s.groupby(["block", "target"], sort=False)[["dx", "dy"]].median().reset_index()
    shift = per[["dx", "dy"]].median().to_numpy()

    groups = dict(list(s.groupby(["block", "target"], sort=False)))
    rows = []
    for i, r in per.iterrows():
        g = groups[(r.block, r.target)]
        loo_shift = per.drop(index=i)[["dx", "dy"]].median().to_numpy() if len(per) > 1 else np.zeros(2)
        raw, raw_h, raw_v = _accuracy(g, rec.camera)
        rows.append(dict(block=r.block, target=r.target, shift_x_px=r.dx, shift_y_px=r.dy,
                         offset_h_deg=raw_h, offset_v_deg=raw_v, raw_accuracy_deg=raw,
                         corrected_accuracy_deg=_accuracy(g, rec.camera, shift)[0],
                         loo_corrected_accuracy_deg=_accuracy(g, rec.camera, loo_shift)[0]))
    targets = pd.DataFrame(rows)

    shift_h, shift_v = targets.offset_h_deg.median(), targets.offset_v_deg.median()
    by_block = targets.groupby("block", sort=False).agg(
        n_targets=("target", "size"), shift_h_deg=("offset_h_deg", "median"), shift_v_deg=("offset_v_deg", "median"),
        raw_mean_accuracy_deg=("raw_accuracy_deg", "mean"),
        corrected_mean_accuracy_deg=("corrected_accuracy_deg", "mean"),
        loo_corrected_mean_accuracy_deg=("loo_corrected_accuracy_deg", "mean")).reset_index()

    flags = []
    agree = {}
    for axis, value, col, words in (("h", shift_h, "offset_h_deg", ("right", "left")),
                                    ("v", shift_v, "offset_v_deg", ("up", "down"))):
        same = float((np.sign(targets[col]) == np.sign(value)).mean())
        agree[axis] = same
        if abs(value) >= MIN_SHIFT_DEG and same < MIN_DIRECTION_AGREEMENT:
            flags.append(f"Only {same:.0%} of targets are shifted {words[0] if value > 0 else words[1]}: "
                         "the bias is not consistent across the screen.")
    block_gap = None
    if {"start", "end"} <= set(by_block.block):
        b = by_block.set_index("block")
        block_gap = float(np.hypot(b.loc["end", "shift_h_deg"] - b.loc["start", "shift_h_deg"],
                                   b.loc["end", "shift_v_deg"] - b.loc["start", "shift_v_deg"]))
        if block_gap > MAX_BLOCK_DISAGREEMENT_DEG:
            flags.append(f"Start and end shifts differ by {block_gap:.2f}° (> {MAX_BLOCK_DISAGREEMENT_DEG}°): "
                         "the headset may have moved; one constant shift may not fit the whole session.")
    else:
        flags.append("Need both a start and an end block to check the shift persists over the session.")
    raw_mean, loo_mean = targets.raw_accuracy_deg.mean(), targets.loo_corrected_accuracy_deg.mean()
    if loo_mean >= raw_mean:
        flags.append(f"On held-out targets the shift does not help ({raw_mean:.2f}° -> {loo_mean:.2f}°).")

    device = rec.info.get("gaze_offset") or [0.0, 0.0]
    summary = dict(
        shift_x_px=float(shift[0]), shift_y_px=float(shift[1]),
        shift_h_deg=float(shift_h), shift_v_deg=float(shift_v),
        direction=f"{abs(shift_h):.2f}° {'right' if shift_h > 0 else 'left'}, "
                  f"{abs(shift_v):.2f}° {'up' if shift_v > 0 else 'down'}",
        n_targets=len(targets), agreement_h=agree["h"], agreement_v=agree["v"],
        start_end_shift_difference_deg=block_gap,
        raw_mean_accuracy_deg=float(raw_mean),
        corrected_mean_accuracy_deg=float(targets.corrected_accuracy_deg.mean()),
        loo_corrected_mean_accuracy_deg=float(loo_mean),
        device_gaze_offset_px=list(device),
        # Neon adds its offset to gaze, so cancelling our measured shift means subtracting it.
        new_device_gaze_offset_px=[float(device[0] - shift[0]), float(device[1] - shift[1])],
        flags=flags, recommended=not flags,
        by_block=by_block.to_dict(orient="records"),
    )
    return summary, targets


def _recompute_angles(df: pd.DataFrame, x: str, y: str, az: str, el: str, camera):
    ok = df[x].notna() & df[y].notna()
    angles = _angles(camera.rays(camera.undistort(df.loc[ok, [x, y]].to_numpy())))
    df.loc[ok, az], df.loc[ok, el] = angles[:, 0], angles[:, 1]


def corrected_gaze(rec: Recording, shift_px) -> pd.DataFrame:
    """gaze.csv with the shift subtracted (all gaze pixel columns) and angles recomputed."""
    g = rec.gaze.copy()
    for side in ("gaze", "gaze mono left", "gaze mono right"):
        if f"{side} x [px]" in g:
            g[f"{side} x [px]"] -= shift_px[0]
            g[f"{side} y [px]"] -= shift_px[1]
    _recompute_angles(g, "gaze x [px]", "gaze y [px]", "azimuth [deg]", "elevation [deg]", rec.camera)
    return g


def corrected_fixations(rec: Recording, shift_px) -> pd.DataFrame | None:
    path = rec.folder / "fixations.csv"
    if not path.exists():
        return None
    f = pd.read_csv(path)
    f["fixation x [px]"] -= shift_px[0]
    f["fixation y [px]"] -= shift_px[1]
    _recompute_angles(f, "fixation x [px]", "fixation y [px]", "azimuth [deg]", "elevation [deg]", rec.camera)
    return f
