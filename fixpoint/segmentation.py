"""Work out when each calibration cross was on screen.

Preferred source: PsychoPy events sent to Neon (e.g. ``cal_target_3`` at onset).
Fallback: which crosses are visibly red in the scene video. Crosses the camera
cannot see (e.g. washed out by the projector hotspot) are inferred from the known
presentation order 1..N and flagged ``timing_source = "inferred"``; crosses seen for only
part of a slide get their onset extended back (``"video_extended"``).
"""
import re

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from .layout import SlideLayout
from .mapping import FrameMap

RED_ON_THRESHOLD = 10.0  # redness above the target's own background level
ALL_TARGETS_MIN = 5  # this many visible crosses = the "all 9" illustration slide
MIN_SEGMENT_S = 0.5
BLOCK_GAP_S = 20.0
EVENT_PATTERN = re.compile(r"^cal(?:ibration)?_target_(\d+)(?:_on)?$")


def frame_states(maps: list[FrameMap], layout: SlideLayout, smooth_frames: int = 7) -> pd.DataFrame:
    """Per mapped frame: which single target is visible (0 = none/ambiguous, -1 = illustration)."""
    ts = np.array([m.timestamp_ns for m in maps], np.int64)
    red = np.array([m.redness for m in maps])
    on = np.zeros_like(red, bool)
    block = _block_ids(ts)
    for b in np.unique(block):
        rows = block == b
        baseline = np.percentile(red[rows], 25, axis=0)
        on[rows] = (red[rows] - baseline) > RED_ON_THRESHOLD
    on = median_filter(on.astype(np.uint8), size=(smooth_frames, 1), mode="nearest").astype(bool)
    n_on = on.sum(axis=1)
    targets = layout.targets.target.to_numpy()
    state = np.where(n_on == 1, targets[on.argmax(axis=1)], 0)
    state[n_on >= ALL_TARGETS_MIN] = -1
    return pd.DataFrame(dict(timestamp_ns=ts, block=block, state=state, n_visible=n_on))


def _block_ids(ts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.diff(ts) > BLOCK_GAP_S * 1e9)])


def _runs(states: pd.DataFrame) -> pd.DataFrame:
    s = states.state.to_numpy()
    change = np.concatenate([[True], s[1:] != s[:-1]])
    run_id = np.cumsum(change)
    runs = states.assign(run=run_id).groupby("run").agg(
        state=("state", "first"), block=("block", "first"),
        start_ns=("timestamp_ns", "first"), end_ns=("timestamp_ns", "last"))
    runs["duration_s"] = (runs.end_ns - runs.start_ns) / 1e9
    return runs.reset_index(drop=True)


def presentations_from_video(states: pd.DataFrame, layout: SlideLayout) -> pd.DataFrame:
    """One row per (block, target): onset/offset in ns and how we know it."""
    targets = layout.targets.target.tolist()
    rows = []
    for block, bstates in states.groupby("block"):
        runs = _runs(bstates)
        known = {}
        for t, truns in runs[runs.state > 0].groupby("state"):
            # Anchor on the longest run, then absorb brief flickers of the same
            # target next to it (camera exposure / blur dropouts).
            main = truns.loc[truns.duration_s.idxmax()]
            if main.duration_s < MIN_SEGMENT_S:
                continue
            near = truns[(truns.end_ns >= main.start_ns - 2e9) & (truns.start_ns <= main.end_ns + 2e9)]
            known[t] = (int(near.start_ns.min()), int(near.end_ns.max()), "video")
        illus = runs[(runs.state == -1) & (runs.duration_s >= MIN_SEGMENT_S)]
        block_start, block_end = bstates.timestamp_ns.iloc[0], bstates.timestamp_ns.iloc[-1]
        last_seen = max((v[1] for v in known.values()), default=block_start)
        seq_end = illus.start_ns.min() if len(illus) and illus.start_ns.min() > last_seen else block_end
        durations = [v[1] - v[0] for v in known.values()]
        typical = float(np.median(durations)) if durations else 3e9

        # A faint cross can be detected only part of the time it was shown. If a
        # window is clearly shorter than a typical slide, start it one typical
        # duration before its end (never overlapping the previous target).
        for i, t in enumerate(targets):
            if t not in known or known[t][1] - known[t][0] >= 0.8 * typical:
                continue
            on, off, _ = known[t]
            prev_off = max((known[p][1] for p in targets[:i] if p in known), default=block_start)
            known[t] = (int(max(min(on, off - typical), prev_off)), off, "video_extended")

        # Fill each run of consecutive undetected targets: between two detected
        # targets split the gap evenly; at the start or end of the sequence step
        # outwards by the typical slide duration (slides advance at a fixed pace).
        i = 0
        while i < len(targets):
            if targets[i] in known:
                i += 1
                continue
            j = i
            while j < len(targets) and targets[j] not in known:
                j += 1
            missing = targets[i:j]
            n = len(missing)
            if i > 0 and j < len(targets):
                lo, hi = known[targets[i - 1]][1], known[targets[j]][0]
            elif j < len(targets):
                hi = known[targets[j]][0]
                lo = hi - typical * n
            elif i > 0:
                lo = known[targets[i - 1]][1]
                hi = min(lo + typical * n, seq_end)
            else:
                break  # nothing detected in this block
            step = (hi - lo) / n
            for k, t in enumerate(missing):
                # May start before the video caught the slides; the analysis then
                # reports the missing data rather than us silently dropping the target.
                known[t] = (int(lo + step * k), int(lo + step * (k + 1)), "inferred")
            i = j
        for t in targets:
            if t in known:
                on, off, src = known[t]
                rows.append(dict(block=int(block), target=t, onset_ns=on, offset_ns=off, timing_source=src))
    return pd.DataFrame(rows)


def presentations_from_events(events: pd.DataFrame) -> pd.DataFrame | None:
    """Use PsychoPy events if the recording has them; None otherwise."""
    ev = events.copy()
    ev["target"] = ev.name.str.extract(EVENT_PATTERN, expand=False)
    ev = ev.sort_values("timestamp [ns]")
    marked = ev.dropna(subset=["target"])
    if marked.empty:
        return None
    rows = []
    ts_all = ev["timestamp [ns]"].to_numpy()
    onsets = marked["timestamp [ns]"].to_numpy()
    block = _block_ids(onsets)
    gaps = np.diff(onsets)[np.diff(block) == 0]
    typical = int(np.median(gaps)) if len(gaps) else int(3e9)
    for (idx, r), b in zip(marked.iterrows(), block):
        on = int(r["timestamp [ns]"])
        # The next event of any kind (e.g. cal_end) ends this target, but never
        # later than a typical cross duration.
        later = ts_all[ts_all > on]
        off = min(int(later[0]), on + typical) if len(later) else on + typical
        rows.append(dict(block=int(b), target=int(r.target), onset_ns=on, offset_ns=off, timing_source="event"))
    return pd.DataFrame(rows)


def label_blocks(pres: pd.DataFrame) -> pd.DataFrame:
    blocks = sorted(pres.block.unique())
    names = {b: f"block_{i + 1}" for i, b in enumerate(blocks)}
    if len(blocks) >= 1:
        names[blocks[0]] = "start"
    if len(blocks) >= 2:
        names[blocks[-1]] = "end"
    return pres.assign(block_name=pres.block.map(names))
