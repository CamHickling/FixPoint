"""Timing of calibration crosses from per-frame visibility (no video needed)."""
import numpy as np
import pandas as pd

from fixpoint.segmentation import presentations_from_events, presentations_from_video


class _Layout:
    targets = pd.DataFrame({"target": range(1, 10)})


def _states(visible_from: dict[int, float], slide_s: float = 3.0, fps: float = 30.0) -> pd.DataFrame:
    """Frames for targets 1..9 shown back to back; visible_from[t] = fraction of the
    slide after which target t becomes visible to the camera (1.0 = never)."""
    ts, state = [], []
    for k, t in enumerate(range(1, 10)):
        for i in range(int(slide_s * fps)):
            ts.append(int((k * slide_s + i / fps) * 1e9))
            state.append(t if i / fps >= visible_from.get(t, 0.0) * slide_s else 0)
    return pd.DataFrame(dict(timestamp_ns=ts, block=0, state=state, n_visible=np.sign(state)))


def _window(pres, target):
    r = pres.set_index("target").loc[target]
    return r.onset_ns / 1e9, r.offset_ns / 1e9, r.timing_source


def test_all_visible():
    pres = presentations_from_video(_states({}), _Layout)
    assert len(pres) == 9 and set(pres.timing_source) == {"video"}
    on, off, _ = _window(pres, 5)
    assert abs(on - 12) < 0.05 and abs(off - 15) < 0.05


def test_invisible_cross_is_inferred_between_neighbours():
    pres = presentations_from_video(_states({2: 1.0}), _Layout)
    on, off, src = _window(pres, 2)
    assert src == "inferred" and abs(on - 3) < 0.1 and abs(off - 6) < 0.1


def test_invisible_first_cross_steps_back_one_slide():
    pres = presentations_from_video(_states({1: 1.0}), _Layout)
    on, off, src = _window(pres, 1)
    assert src == "inferred" and abs(off - 3) < 0.1 and abs(on - 0) < 0.15


def test_partly_visible_cross_is_extended():
    pres = presentations_from_video(_states({4: 0.6}), _Layout)
    on, off, src = _window(pres, 4)
    assert src == "video_extended" and abs(on - 9) < 0.15 and abs(off - 12) < 0.1


def test_events_take_priority_and_end_at_next_event():
    names = [f"cal_target_{i}" for i in range(1, 10)] + ["cal_end"]
    ts = [i * 3e9 for i in range(10)]
    pres = presentations_from_events(pd.DataFrame({"name": names, "timestamp [ns]": ts}))
    assert list(pres.target) == list(range(1, 10))
    assert (pres.offset_ns - pres.onset_ns == 3e9).all()
    assert presentations_from_events(pd.DataFrame({"name": ["trial_start_0"], "timestamp [ns]": [0]})) is None
