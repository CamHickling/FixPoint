"""Find the calibration slides in the scene video and map scene pixels onto the slide.

Each video frame gets a homography from *undistorted* scene pixels to slide pixels,
built from the AprilTag corners, plus a redness score for every cross position so we
can tell which calibration slide was on screen without needing PsychoPy events.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from pupil_apriltags import Detector

from .layout import TAG_FAMILY, SlideLayout, detect_tags
from .recording import Recording

PATCH_RADIUS = 40  # slide px around each cross that we inspect for red


@dataclass
class FrameMap:
    frame: int
    timestamp_ns: int
    tag_ids: tuple[int, ...]
    homography: np.ndarray  # undistorted scene px -> slide px
    reprojection_error_px: float  # mean tag-corner error, in slide px
    redness: np.ndarray  # one score per target


def _is_calibration_frame(tag_ids, layout: SlideLayout) -> bool:
    # Task slides reuse some tag ids at other positions, so every visible tag must
    # belong to the calibration layout, and at least two must be visible.
    return len(tag_ids) >= 2 and set(tag_ids) <= set(layout.tags)


def _spans(frames: list[int], step: int, gap_frames: int, pad: int, n_frames: int) -> list[tuple[int, int]]:
    spans = []
    for f in frames:
        if spans and f - spans[-1][1] <= max(gap_frames, step):
            spans[-1][1] = f
        else:
            spans.append([f, f])
    return [(max(0, a - pad), min(n_frames - 1, b + pad)) for a, b in spans]


def _patch_grid(layout: SlideLayout) -> np.ndarray:
    offsets = np.arange(-PATCH_RADIUS, PATCH_RADIUS + 1, 2, dtype=np.float64)
    gx, gy = np.meshgrid(offsets, offsets)
    centres = layout.targets[["x", "y"]].to_numpy()
    # (n_targets, h, w, 2) slide coordinates
    return centres[:, None, None, :] + np.stack([gx, gy], axis=-1)[None]


def _redness(frame_bgr, h_slide_to_scene, grid, camera) -> np.ndarray:
    """Sample each cross patch from the distorted frame and score how red it is."""
    n, h, w, _ = grid.shape
    slide_pts = grid.reshape(-1, 1, 2)
    undist = cv2.perspectiveTransform(slide_pts, h_slide_to_scene).reshape(-1, 2)
    norm = np.column_stack([undist, np.ones(len(undist))]) @ np.linalg.inv(camera.camera_matrix).T
    dist, _ = cv2.projectPoints(norm, np.zeros(3), np.zeros(3), camera.camera_matrix, camera.distortion)
    maps = dist.reshape(n * h, w, 2).astype(np.float32)
    patch = cv2.remap(frame_bgr, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    patch = patch.reshape(n, h, w, 3).astype(np.float32)
    b, g, r = patch[..., 0], patch[..., 1], patch[..., 2]
    red = r - np.maximum(g, b)
    flat = red.reshape(n, -1)
    # Peak redness above the patch background; robust to the slide's colour cast.
    return np.percentile(flat, 99, axis=1) - np.median(flat, axis=1)


def scan_video(rec: Recording, layout: SlideLayout, coarse_step: int = 10, progress=print) -> list[FrameMap]:
    detector = Detector(families=TAG_FAMILY, quad_decimate=2.0, refine_edges=True)
    cap = cv2.VideoCapture(str(rec.video_path))
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(rec.world_timestamps))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Pass 1: sample sparsely to find when the calibration layout is on screen.
    hits = []
    for f in range(n_frames):
        if f % coarse_step:
            cap.grab()
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if _is_calibration_frame(tuple(detect_tags(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), detector)), layout):
            hits.append(f)
    spans = _spans(hits, coarse_step, gap_frames=int(3 * fps), pad=int(fps), n_frames=n_frames)
    progress(f"  calibration layout visible in {len(spans)} span(s): "
             + ", ".join(f"{a / fps:.1f}-{b / fps:.1f}s" for a, b in spans))

    # Pass 2: every frame inside those spans.
    grid = _patch_grid(layout)
    maps: list[FrameMap] = []
    for start, end in spans:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for f in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            tags = detect_tags(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), detector)
            if not _is_calibration_frame(tuple(tags), layout):
                continue
            ids = tuple(sorted(tags))
            scene = rec.camera.undistort(np.concatenate([tags[i] for i in ids]))
            slide = np.concatenate([layout.tags[i] for i in ids])
            H, _ = cv2.findHomography(scene, slide, 0)
            if H is None:
                continue
            reproj = cv2.perspectiveTransform(scene.reshape(-1, 1, 2), H).reshape(-1, 2)
            err = float(np.linalg.norm(reproj - slide, axis=1).mean())
            red = _redness(frame, np.linalg.inv(H), grid, rec.camera)
            maps.append(FrameMap(f, int(rec.world_timestamps[f]), ids, H, err, red))
    cap.release()
    return maps


def frames_table(maps: list[FrameMap], layout: SlideLayout) -> pd.DataFrame:
    rows = []
    for m in maps:
        row = dict(frame=m.frame, timestamp_ns=m.timestamp_ns, tag_ids=";".join(map(str, m.tag_ids)),
                   tag_reprojection_error_px=m.reprojection_error_px)
        row.update({f"redness_{t}": v for t, v in zip(layout.targets.target, m.redness)})
        rows.append(row)
    return pd.DataFrame(rows)
