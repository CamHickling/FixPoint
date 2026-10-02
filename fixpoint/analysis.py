"""Accuracy, precision, data loss, drift and correction models for each calibration target.

Angles are computed in the scene camera: the target's slide position is projected back
into the (undistorted) scene image through that frame's homography and turned into a
viewing direction, as is the gaze. Their difference is the error in degrees - no need
to measure the screen or viewing distance.
"""
import numpy as np
import pandas as pd

from .layout import SlideLayout
from .mapping import FrameMap
from .recording import Recording

GAZE_RATE_HZ = 200
MAX_FRAME_GAP_NS = 100_000_000  # gaze further than this from a mapped frame is unmapped
MIN_VALID_SAMPLES = 50  # 0.25 s of gaze; fewer and the target is excluded from fits


def _angles(rays: np.ndarray) -> np.ndarray:
    """Unit rays (camera coords: x right, y down, z forward) -> (azimuth, elevation) deg."""
    az = np.degrees(np.arctan2(rays[:, 0], rays[:, 2]))
    el = np.degrees(np.arctan2(-rays[:, 1], np.hypot(rays[:, 0], rays[:, 2])))
    return np.column_stack([az, el])


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((a * b).sum(axis=1), -1, 1)))


def _slide_to_rays(points, h_slide_to_scene, camera) -> np.ndarray:
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    hom = np.column_stack([pts, np.ones(len(pts))]) @ h_slide_to_scene.T
    return camera.rays(hom[:, :2] / hom[:, 2:3])


def map_gaze(rec: Recording, maps: list[FrameMap], t0: int, t1: int) -> pd.DataFrame:
    """All gaze samples in [t0, t1] with slide coordinates and camera rays."""
    g = rec.gaze[(rec.gaze["timestamp [ns]"] >= t0) & (rec.gaze["timestamp [ns]"] <= t1)].copy()
    out = pd.DataFrame({"timestamp_ns": g["timestamp [ns]"].to_numpy()})
    out["worn"] = g["worn"].to_numpy() == 1
    out["blink"] = g["blink id"].notna().to_numpy()
    out["fixation_id"] = g["fixation id"].to_numpy()
    if out.empty or not maps:
        return out.assign(mapped=False)
    frame_ts = np.array([m.timestamp_ns for m in maps], np.int64)
    idx = np.clip(np.searchsorted(frame_ts, out.timestamp_ns), 1, len(frame_ts) - 1)
    idx -= (out.timestamp_ns - frame_ts[idx - 1]) < (frame_ts[idx] - out.timestamp_ns)
    out["frame_index"] = idx
    out["mapped"] = np.abs(frame_ts[idx] - out.timestamp_ns) <= MAX_FRAME_GAP_NS

    raw = g[["gaze x [px]", "gaze y [px]"]].to_numpy()
    out[["gaze_x_px", "gaze_y_px"]] = raw
    undist = rec.camera.undistort(raw)
    out[["scene_ux", "scene_uy"]] = undist
    rays = rec.camera.rays(undist)
    out[["ray_x", "ray_y", "ray_z"]] = rays
    hs = np.stack([maps[i].homography for i in idx])
    hom = np.einsum("nij,nj->ni", hs, np.column_stack([undist, np.ones(len(undist))]))
    out["slide_x"] = hom[:, 0] / hom[:, 2]
    out["slide_y"] = hom[:, 1] / hom[:, 2]
    return out


def target_rays(samples, maps, target_xy, camera) -> np.ndarray:
    rays = np.empty((len(samples), 3))
    for fi, rows in samples.groupby("frame_index").indices.items():
        h_inv = np.linalg.inv(maps[fi].homography)
        rays[rows] = _slide_to_rays(target_xy, h_inv, camera)
    return rays


def analyse_presentations(rec, maps, layout: SlideLayout, pres: pd.DataFrame, skip_s: float = 0.5):
    """Returns (per-target metrics, per-sample table)."""
    targets = layout.targets.set_index("target")
    metric_rows, sample_tables = [], []
    for p in pres.itertuples():
        t_xy = targets.loc[p.target, ["x", "y"]].to_numpy(np.float64)
        w0, w1 = p.onset_ns + int(skip_s * 1e9), p.offset_ns
        expected = max(0.0, (w1 - w0) / 1e9 * GAZE_RATE_HZ)
        s = map_gaze(rec, maps, w0, w1)
        valid = s[s.mapped & s.worn & ~s.blink] if len(s) else s
        row = dict(block=p.block_name, target=p.target, target_name=targets.loc[p.target, "name"],
                   target_x=t_xy[0], target_y=t_xy[1], timing_source=p.timing_source,
                   onset_s=(p.onset_ns - rec.start_ns) / 1e9, offset_s=(p.offset_ns - rec.start_ns) / 1e9,
                   window_s=(w1 - w0) / 1e9, n_samples=len(valid),
                   data_loss_pct=100 * (1 - len(valid) / expected) if expected else np.nan,
                   blink_pct=100 * s.blink.mean() if len(s) else np.nan,
                   unmapped_pct=100 * (~s.mapped).mean() if len(s) else np.nan)
        if len(valid):
            valid = valid.copy()
            g_rays = valid[["ray_x", "ray_y", "ray_z"]].to_numpy()
            t_rays = target_rays(valid, maps, t_xy, rec.camera)
            offsets = _angles(g_rays) - _angles(t_rays)  # +h = right of target, +v = above
            err = _angle_between(g_rays, t_rays)
            valid[["offset_h_deg", "offset_v_deg"]] = offsets
            valid["error_deg"] = err
            # Target position in the raw (distorted) scene image, same space as gaze x/y [px].
            t_und = t_rays @ rec.camera.camera_matrix.T
            valid[["target_scene_x_px", "target_scene_y_px"]] = rec.camera.distort(t_und[:, :2] / t_und[:, 2:3])
            med = np.median(offsets, axis=0)
            gx, gy = valid.slide_x.median(), valid.slide_y.median()
            mid = maps[int(np.median(valid.frame_index))]
            h_inv = np.linalg.inv(mid.homography)
            # Visual angle of one slide pixel at this target (horizontal / vertical).
            r = _slide_to_rays(np.array([t_xy, t_xy + [50, 0], t_xy + [0, 50]]), h_inv, rec.camera)
            deg_px_h = _angle_between(r[:1], r[1:2])[0] / 50
            deg_px_v = _angle_between(r[:1], r[2:3])[0] / 50
            consecutive = np.diff(valid.timestamp_ns.to_numpy()) <= 1.5e9 / GAZE_RATE_HZ
            s2s = _angle_between(g_rays[1:], g_rays[:-1])[consecutive]
            row.update(
                gaze_x=gx, gaze_y=gy, error_x_px=gx - t_xy[0], error_y_px=gy - t_xy[1],
                error_px=float(np.hypot(gx - t_xy[0], gy - t_xy[1])),
                offset_h_deg=med[0], offset_v_deg=med[1], accuracy_deg=float(np.hypot(*med)),
                mean_sample_error_deg=float(err.mean()), median_sample_error_deg=float(np.median(err)),
                precision_rms_s2s_deg=float(np.sqrt(np.mean(s2s ** 2))) if len(s2s) else np.nan,
                precision_sd_deg=float(np.sqrt(offsets.var(axis=0).sum())),
                # Robust SD (1.4826 x MAD per axis): not inflated by a late arrival or a stray glance.
                precision_robust_sd_deg=float(np.hypot(*(1.4826 * np.median(np.abs(offsets - med), axis=0)))),
                pct_within_1deg=100 * float((err <= 1).mean()), pct_within_2deg=100 * float((err <= 2).mean()),
                pct_on_slide=100 * float(((valid.slide_x.between(0, layout.size[0]))
                                          & (valid.slide_y.between(0, layout.size[1]))).mean()),
                deg_per_slide_px_h=deg_px_h, deg_per_slide_px_v=deg_px_v,
                tag_reprojection_error_px=float(np.mean([maps[i].reprojection_error_px for i in valid.frame_index])),
            )
            sample_tables.append(valid.assign(block=p.block_name, target=p.target))
        row["usable"] = row["n_samples"] >= MIN_VALID_SAMPLES
        metric_rows.append(row)
    samples = pd.concat(sample_tables, ignore_index=True) if sample_tables else pd.DataFrame()
    return pd.DataFrame(metric_rows), samples


# --- correction models -------------------------------------------------------------

def _fit(kind: str, gaze: np.ndarray, target: np.ndarray) -> np.ndarray:
    """3x2 matrix M so that [x, y, 1] @ M = corrected slide position."""
    if kind == "offset":
        shift = (target - gaze).mean(axis=0)
        return np.vstack([np.eye(2), shift])
    design = np.column_stack([gaze, np.ones(len(gaze))])
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.column_stack([pts, np.ones(len(pts))]) @ m


MIN_POINTS = {"offset": 1, "affine": 4}


def _px_error_to_deg(df: pd.DataFrame, corrected: np.ndarray) -> np.ndarray:
    """Approximate a corrected slide-px error in degrees using the local deg/px."""
    dx = corrected[:, 0] - df.target_x.to_numpy()
    dy = corrected[:, 1] - df.target_y.to_numpy()
    return np.hypot(dx * df.deg_per_slide_px_h.to_numpy(), dy * df.deg_per_slide_px_v.to_numpy())


def correction_models(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-out within each block, plus fit-on-start / test-on-end.

    Returns (summary rows, per-target corrected errors).
    """
    ok = metrics[metrics.usable].copy()
    summary, per_target = [], []
    for block, b in ok.groupby("block", sort=False):
        gaze, target = b[["gaze_x", "gaze_y"]].to_numpy(), b[["target_x", "target_y"]].to_numpy()
        for kind, need in MIN_POINTS.items():
            if len(b) < need + 1:
                continue
            loo = np.empty_like(gaze)
            for i in range(len(b)):
                keep = np.arange(len(b)) != i
                loo[i] = _apply(_fit(kind, gaze[keep], target[keep]), gaze[i:i + 1])[0]
            err = _px_error_to_deg(b, loo)
            full = _fit(kind, gaze, target)
            summary.append(dict(model=kind, fit_on=block, test_on=block, method="leave-one-out",
                                n_targets=len(b), raw_mean_accuracy_deg=b.accuracy_deg.mean(),
                                corrected_mean_error_deg=err.mean(), corrected_max_error_deg=err.max(),
                                **_describe(full)))
            per_target += [dict(model=kind, fit_on=block, test_on=block, target=t, corrected_error_deg=e)
                           for t, e in zip(b.target, err)]
    blocks = list(ok.block.unique())
    if "start" in blocks and "end" in blocks:
        s, e = ok[ok.block == "start"], ok[ok.block == "end"]
        for kind, need in MIN_POINTS.items():
            if len(s) < need:
                continue
            m = _fit(kind, s[["gaze_x", "gaze_y"]].to_numpy(), s[["target_x", "target_y"]].to_numpy())
            err = _px_error_to_deg(e, _apply(m, e[["gaze_x", "gaze_y"]].to_numpy()))
            summary.append(dict(model=kind, fit_on="start", test_on="end", method="fit start, test end",
                                n_targets=len(e), raw_mean_accuracy_deg=e.accuracy_deg.mean(),
                                corrected_mean_error_deg=err.mean(), corrected_max_error_deg=err.max(),
                                **_describe(m)))
            per_target += [dict(model=kind, fit_on="start", test_on="end", target=t, corrected_error_deg=v)
                           for t, v in zip(e.target, err)]
    return pd.DataFrame(summary), pd.DataFrame(per_target)


def _describe(m: np.ndarray) -> dict:
    return dict(shift_x_px=m[2, 0], shift_y_px=m[2, 1], a11=m[0, 0], a12=m[1, 0], a21=m[0, 1], a22=m[1, 1])


# --- summaries ---------------------------------------------------------------------

def block_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block, b in metrics.groupby("block", sort=False):
        u = b[b.usable]
        rows.append(dict(
            block=block, n_targets=len(b), n_usable=len(u),
            n_inferred_timing=int((b.timing_source == "inferred").sum()),
            mean_accuracy_deg=u.accuracy_deg.mean(), median_accuracy_deg=u.accuracy_deg.median(),
            max_accuracy_deg=u.accuracy_deg.max(),
            mean_offset_h_deg=u.offset_h_deg.mean(), mean_offset_v_deg=u.offset_v_deg.mean(),
            mean_precision_rms_s2s_deg=u.precision_rms_s2s_deg.mean(),
            mean_precision_sd_deg=u.precision_sd_deg.mean(),
            mean_precision_robust_sd_deg=u.precision_robust_sd_deg.mean(),
            mean_pct_within_2deg=u.pct_within_2deg.mean(),
            mean_data_loss_pct=b.data_loss_pct.mean(),
            mean_error_px=u.error_px.mean(),
        ))
    return pd.DataFrame(rows)


def drift(metrics: pd.DataFrame) -> pd.DataFrame:
    """Change in each target's offset from the start block to the end block."""
    u = metrics[metrics.usable]
    s = u[u.block == "start"].set_index("target")
    e = u[u.block == "end"].set_index("target")
    both = s.index.intersection(e.index)
    if both.empty:
        return pd.DataFrame()
    d = pd.DataFrame(index=both)
    d["drift_h_deg"] = e.loc[both, "offset_h_deg"] - s.loc[both, "offset_h_deg"]
    d["drift_v_deg"] = e.loc[both, "offset_v_deg"] - s.loc[both, "offset_v_deg"]
    d["drift_deg"] = np.hypot(d.drift_h_deg, d.drift_v_deg)
    d["accuracy_change_deg"] = e.loc[both, "accuracy_deg"] - s.loc[both, "accuracy_deg"]
    return d.reset_index(names="target")


def screen_extent_deg(maps: list[FrameMap], layout: SlideLayout, camera) -> dict:
    """Visual angle of the whole projected slide, and the equivalent viewing distance."""
    w, h = layout.size
    i = len(maps) // 2
    h_inv = np.linalg.inv(maps[i].homography)
    r = _slide_to_rays(np.array([[0, h / 2], [w, h / 2], [w / 2, 0], [w / 2, h]]), h_inv, camera)
    width_deg = float(_angle_between(r[:1], r[1:2])[0])
    height_deg = float(_angle_between(r[2:3], r[3:4])[0])
    return dict(slide_width_deg=width_deg, slide_height_deg=height_deg,
                viewing_distance_in_screen_widths=1 / (2 * np.tan(np.radians(width_deg / 2))))
