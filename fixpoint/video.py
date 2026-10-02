"""Render the whole scene video with gaze before and after the session offset correction.

Layout (1920x1080, real time): scene video on the left with both gaze rings and short
trails; on the right a 3x magnifier (locked on the current cross during calibration,
following gaze during the task), a live error readout and a session timeline.
"""
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .mapping import FrameMap
from .recording import Recording

W, H = 1920, 1080
SCENE_W = 1440  # scene panel width (1600x1200 scaled by 0.9)
COL_X, PAD = SCENE_W, 28
ZOOM_SRC = 144  # scene px shown in the magnifier (~9 deg)
ZOOM = 432  # magnifier size on screen
TRAIL_S = 0.2

# Dark presentation theme (colours as RGB; converted to BGR for OpenCV).
BG, INK, INK_2, MUTED, LINE = (26, 26, 25), (255, 255, 255), (195, 194, 183), (137, 135, 129), (56, 56, 53)
BEFORE, AFTER, TARGET = (235, 104, 52), (57, 135, 229), (255, 255, 255)


def _bgr(rgb):
    return rgb[::-1]


def _font(size, bold=False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class _GazeTrack:
    """Gaze samples (raw scene px) with fast lookups by time."""

    def __init__(self, gaze: pd.DataFrame, shift):
        ok = (gaze["worn"] == 1) & gaze["blink id"].isna() & gaze["gaze x [px]"].notna()
        g = gaze[ok]
        self.ts = g["timestamp [ns]"].to_numpy(np.int64)
        self.before = g[["gaze x [px]", "gaze y [px]"]].to_numpy(np.float64)
        self.after = self.before - np.asarray(shift, np.float64)

    def window(self, t_ns, span_ns):
        i1 = np.searchsorted(self.ts, t_ns, side="right")
        i0 = np.searchsorted(self.ts, t_ns - span_ns, side="left")
        # No ring when the latest sample is stale (blink, not worn, tracking gap).
        if i1 == 0 or t_ns - self.ts[i1 - 1] > 50_000_000:
            return None
        return slice(i0, i1)


def _draw_ring(img, p, radius, color, scale=1.0, offset=(0, 0)):
    c = (int(round((p[0] - offset[0]) * scale)), int(round((p[1] - offset[1]) * scale)))
    cv2.circle(img, c, int(radius * scale) + 2, (20, 20, 20), 2, cv2.LINE_AA)  # halo for contrast
    cv2.circle(img, c, int(radius * scale), _bgr(color), max(2, int(1.5 * scale)), cv2.LINE_AA)
    cv2.circle(img, c, max(2, int(1.2 * scale)), _bgr(color), -1, cv2.LINE_AA)


def _draw_trail(img, pts, color, scale=1.0, offset=(0, 0)):
    if len(pts) < 2:
        return
    q = ((pts - np.asarray(offset)) * scale).astype(np.int32)
    n = len(q)
    for k in range(1, n):  # thicker and brighter towards the present
        w = k / n
        col = tuple(int(c * w + b * (1 - w)) for c, b in zip(_bgr(color), (60, 60, 60)))
        cv2.line(img, tuple(q[k - 1]), tuple(q[k]), col, max(1, int(round(w * 3 * max(1, scale / 2)))),
                 cv2.LINE_AA)


def _target_scene_px(fm: FrameMap, xy, camera):
    h_inv = np.linalg.inv(fm.homography)
    p = h_inv @ np.array([xy[0], xy[1], 1.0])
    return camera.distort(p[None, :2] / p[2])[0]


def _running_errors(pres, samples_before, samples_after):
    """Per presentation: sample times and h/v offsets, before and after."""
    out = {}
    for p in pres.itertuples():
        entry = {}
        for name, s in (("before", samples_before), ("after", samples_after)):
            ss = s[(s.block == p.block_name) & (s.target == p.target)] if len(s) else s
            entry[name] = (ss.timestamp_ns.to_numpy(np.int64), ss[["offset_h_deg", "offset_v_deg"]].to_numpy())
        out[(p.block_name, p.target)] = entry
    return out


def _error_so_far(entry, t_ns):
    vals = {}
    for name, (ts, off) in entry.items():
        n = np.searchsorted(ts, t_ns, side="right")
        vals[name] = float(np.hypot(*np.median(off[:n], axis=0))) if n >= 20 else None
    return vals


def render(rec: Recording, maps: list[FrameMap], layout, pres: pd.DataFrame, offset: dict,
           samples_before: pd.DataFrame, samples_after: pd.DataFrame, out_path: Path,
           skip_s: float = 0.5, stills: list[float] | None = None, progress=print):
    shift = (offset["shift_x_px"], offset["shift_y_px"])
    track = _GazeTrack(rec.gaze, shift)
    px_per_deg = rec.camera.camera_matrix[0, 0] * np.tan(np.radians(1.0))
    ring_r = px_per_deg  # 1 degree radius
    frame_maps = {m.frame: m for m in maps}
    targets = layout.targets.set_index("target")
    errors = _running_errors(pres, samples_before, samples_after)
    names = layout.targets.set_index("target")["name"].str.replace("_", " ")
    pres = pres.sort_values("onset_ns").assign(target_name=lambda p: p.target.map(names))
    blocks = pres.groupby("block_name").agg(on=("onset_ns", "min"), off=("offset_ns", "max"))
    t_start, t_end = int(rec.world_timestamps[0]), int(rec.world_timestamps[-1])
    world = rec.world_timestamps

    f_title, f_h, f_body, f_small, f_big = _font(26, True), _font(19, True), _font(18), _font(15), _font(30, True)
    static = _static_panel(rec, offset, blocks, t_start, t_end, f_title, f_body, f_small)

    zoom_c = None

    def compose(f, frame):
        nonlocal zoom_c
        t = int(world[f])
        canvas = static.copy()

        # Current calibration presentation, if any.
        cur = pres[(pres.onset_ns <= t) & (pres.offset_ns > t)]
        cur = cur.iloc[0] if len(cur) else None
        fm = frame_maps.get(f)
        target_px = None
        if cur is not None and fm is not None:
            target_px = _target_scene_px(fm, targets.loc[cur.target, ["x", "y"]].to_numpy(float), rec.camera)

        win = track.window(t, int(TRAIL_S * 1e9))
        scene = frame.copy()
        if target_px is not None:
            _draw_target(scene, target_px, ring_r)
        if win is not None:
            overlay = scene.copy()
            for pts, col in ((track.before[win], BEFORE), (track.after[win], AFTER)):
                _draw_trail(overlay, pts, col)
            scene = _blend_trails(scene, overlay)
            _draw_ring(scene, track.before[win][-1], ring_r, BEFORE)
            _draw_ring(scene, track.after[win][-1], ring_r, AFTER)
        canvas[:, :SCENE_W] = cv2.resize(scene, (SCENE_W, H), interpolation=cv2.INTER_AREA)

        # Magnifier: on the cross during calibration, otherwise smoothly following gaze.
        if target_px is not None:
            want = target_px
        elif win is not None:
            want = (track.before[win][-1] + track.after[win][-1]) / 2
        else:
            want = zoom_c
        if want is not None:
            zoom_c = want if zoom_c is None or target_px is not None else zoom_c + 0.15 * (want - zoom_c)
        zx, zy = COL_X + PAD, 196
        if zoom_c is not None:
            canvas[zy:zy + ZOOM, zx:zx + ZOOM] = _magnifier(frame, zoom_c, track, win, target_px, ring_r)
        cv2.rectangle(canvas, (zx - 1, zy - 1), (zx + ZOOM, zy + ZOOM), _bgr(LINE), 1)
        return _dynamic_text(canvas, t, t_start, t_end, rec.start_ns, cur, blocks, errors, skip_s, f_h, f_body,
                             f_small, f_big, fm is not None)

    cap = cv2.VideoCapture(str(rec.video_path))
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(world))
    if stills:
        # Single frames as PNG (design checks, slides): out_path is then a folder.
        out_path.mkdir(parents=True, exist_ok=True)
        for sec in stills:
            f = int(min(np.searchsorted(world, t_start + sec * 1e9), n_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if ok:
                zoom_c = None
                cv2.imwrite(str(out_path / f"still_{sec:06.1f}s.png"), compose(f, frame))
        cap.release()
        progress(f"  video: wrote {len(stills)} stills to {out_path}")
        return

    fps = n_frames / ((t_end - t_start) / 1e9) if t_end > t_start else 30.0
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            enc.stdin.write(compose(f, frame).tobytes())
            if f % 1500 == 0:
                progress(f"  video: {f}/{n_frames} frames")
    finally:
        cap.release()
        enc.stdin.close()
        enc.wait()
    progress(f"  video: wrote {out_path}")


def _blend_trails(scene, overlay):
    # Trails are drawn on a copy; blending keeps them from hiding the scene.
    return cv2.addWeighted(overlay, 0.75, scene, 0.25, 0)


def _draw_target(img, p, ring_r, scale=1.0, offset=(0, 0)):
    c = (int(round((p[0] - offset[0]) * scale)), int(round((p[1] - offset[1]) * scale)))
    arm = int(0.35 * ring_r * scale)
    for col, th in ((_bgr(TARGET), 5), ((20, 20, 20), 2)):
        cv2.line(img, (c[0] - arm, c[1]), (c[0] + arm, c[1]), col, th, cv2.LINE_AA)
        cv2.line(img, (c[0], c[1] - arm), (c[0], c[1] + arm), col, th, cv2.LINE_AA)


def _magnifier(frame, centre, track, win, target_px, ring_r):
    s = ZOOM / ZOOM_SRC
    x0, y0 = centre[0] - ZOOM_SRC / 2, centre[1] - ZOOM_SRC / 2
    # Sub-pixel crop + upscale in one affine warp.
    m = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float64)
    z = cv2.warpAffine(frame, m, (ZOOM, ZOOM), flags=cv2.INTER_CUBIC, borderValue=BG[::-1])
    if target_px is not None:
        _draw_target(z, target_px, ring_r, s, (x0, y0))
        for deg, th in ((1, 1), (2, 1)):  # 1 and 2 degree guides around the cross
            c = (int((target_px[0] - x0) * s), int((target_px[1] - y0) * s))
            cv2.circle(z, c, int(deg * ring_r * s), _bgr(MUTED), th, cv2.LINE_AA)
    if win is not None:
        overlay = z.copy()
        for pts, col in ((track.before[win], BEFORE), (track.after[win], AFTER)):
            _draw_trail(overlay, pts, col, s, (x0, y0))
        z = _blend_trails(z, overlay)
        _draw_ring(z, track.before[win][-1], ring_r, BEFORE, s, (x0, y0))
        _draw_ring(z, track.after[win][-1], ring_r, AFTER, s, (x0, y0))
    # Scale bar: 1 degree.
    bar = int(ring_r * s)
    cv2.rectangle(z, (8, ZOOM - 34), (bar + 78, ZOOM - 6), _bgr(BG), -1)
    cv2.line(z, (16, ZOOM - 20), (16 + bar, ZOOM - 20), _bgr(INK), 2, cv2.LINE_AA)
    cv2.putText(z, "1 deg", (24 + bar, ZOOM - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _bgr(INK), 1, cv2.LINE_AA)
    return z


def _timeline_geom():
    return COL_X + PAD, H - 64, W - PAD, H - 52  # x0, y0, x1, y1


def _static_panel(rec, offset, blocks, t_start, t_end, f_title, f_body, f_small):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    x = COL_X + PAD
    d.text((x, 24), "Gaze offset correction", font=f_title, fill=INK)
    d.text((x, 60), f"{rec.info.get('wearer_name', '')} · {rec.name}", font=f_small, fill=MUTED)
    y = 98
    for col, label in ((BEFORE, "before correction"), (AFTER, f"after: shift {offset['direction']}")):
        d.ellipse((x, y + 3, x + 18, y + 21), outline=col, width=3)
        d.text((x + 28, y), label, font=f_body, fill=INK_2)
        y += 30
    d.text((x, 168), "Magnified ×3", font=f_small, fill=MUTED)
    # Timeline: calibration blocks vs task.
    x0, y0, x1, y1 = _timeline_geom()
    d.rectangle((x0, y0, x1, y1), fill=LINE)
    span = t_end - t_start
    for name, b in blocks.iterrows():
        a = x0 + (b.on - t_start) / span * (x1 - x0)
        c = x0 + (b.off - t_start) / span * (x1 - x0)
        d.rectangle((a, y0, max(c, a + 2), y1), fill=MUTED)
    d.rectangle((x0, y1 + 12, x0 + 12, y1 + 24), fill=MUTED)
    d.text((x0 + 18, y1 + 8), "calibration", font=f_small, fill=INK_2)
    d.rectangle((x0 + 120, y1 + 12, x0 + 132, y1 + 24), fill=LINE, outline=MUTED)
    d.text((x0 + 138, y1 + 8), "task", font=f_small, fill=INK_2)
    # Session result, for presenting.
    y = 880
    d.text((x, y), "Mean error to crosses (held-out targets)", font=f_small, fill=MUTED)
    y += 24
    for b in offset.get("by_block", []):
        d.text((x, y), f"{b['block']} block", font=f_body, fill=INK_2)
        d.text((x + 130, y), f"{b['raw_mean_accuracy_deg']:.2f}°", font=f_body, fill=BEFORE)
        d.text((x + 200, y), "→", font=f_body, fill=INK_2)
        d.text((x + 230, y), f"{b['loo_corrected_mean_accuracy_deg']:.2f}°", font=f_body, fill=AFTER)
        y += 26
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _dynamic_text(canvas, t, t_start, t_end, rec_start, cur, blocks, errors, skip_s, f_h, f_body, f_small, f_big, on_cal):
    x, y = COL_X + PAD, 196 + ZOOM + 22
    panel = Image.fromarray(cv2.cvtColor(canvas[:, COL_X:], cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(panel)
    px = PAD
    if cur is not None:
        phase = f"Calibration – {cur.block_name} block"
        d.text((px, y), phase, font=f_h, fill=INK)
        src = " (timing inferred)" if cur.timing_source == "inferred" else ""
        d.text((px, y + 30), f"Cross {cur.target} of 9 · {cur.target_name}{src}", font=f_body, fill=INK_2)
        if t < cur.onset_ns + skip_s * 1e9:
            d.text((px, y + 72), "eye moving to the cross…", font=f_body, fill=MUTED)
        else:
            e = _error_so_far(errors[(cur.block_name, cur.target)], t)
            d.text((px, y + 66), "Error to cross so far", font=f_small, fill=MUTED)
            yy = y + 90
            for name, col in (("before", BEFORE), ("after", AFTER)):
                v = e[name]
                d.ellipse((px, yy + 10, px + 16, yy + 26), outline=col, width=3)
                d.text((px + 28, yy), "–" if v is None else f"{v:.2f}°", font=f_big, fill=INK)
                d.text((px + 150, yy + 10), name, font=f_body, fill=INK_2)
                yy += 44
    else:
        label = "Calibration – overview slide" if on_cal else "Task"
        d.text((px, y), label, font=f_h, fill=INK)
        d.text((px, y + 30), "Magnifier follows gaze", font=f_body, fill=INK_2)
    # Clock and playhead.
    secs = (t - rec_start) / 1e9
    d.text((px, H - 98), f"{int(secs // 60)}:{secs % 60:04.1f}", font=f_small, fill=INK_2)
    x0, y0, x1, y1 = _timeline_geom()
    p = (x0 - COL_X) + (t - t_start) / max(1, t_end - t_start) * (x1 - x0)
    d.rectangle((p - 1.5, y0 - 6, p + 1.5, y1 + 6), fill=INK)
    canvas[:, COL_X:] = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    return canvas
