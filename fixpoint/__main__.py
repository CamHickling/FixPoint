"""Command line: python -m fixpoint <recording folder(s) or a parent folder> [options]"""
import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

from .layout import DEFAULT_SLIDES, load_calibration_slides
from .pipeline import run_recording
from .recording import find_recordings


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fixpoint", description="Check Neon gaze accuracy against the 3x3 calibration crosses.")
    ap.add_argument("paths", nargs="+", help="Neon timeseries export folder(s), or folders containing them")
    ap.add_argument("--slides", default=str(DEFAULT_SLIDES),
                    help="folder with your calibration_target_*.png slides (default: the bundled 1920x1080 set)")
    ap.add_argument("--out", default="output", help="where to write reports")
    ap.add_argument("--skip", type=float, default=0.5,
                    help="seconds after each cross appears to ignore while the eye moves to it (default 0.5)")
    ap.add_argument("--video", action="store_true",
                    help="also render the whole scene video with gaze before/after correction (~real time)")
    args = ap.parse_args(argv)

    layout = load_calibration_slides(args.slides)
    print(f"Slides: {len(layout.targets)} targets, tags {sorted(layout.tags)}, {layout.size[0]}x{layout.size[1]} px")
    folders = sorted({r for p in args.paths for r in find_recordings(p)})
    if not folders:
        sys.exit("No Neon recordings (folders with gaze.csv + world_timestamps.csv) found.")

    out = Path(args.out)
    summaries, failed = [], []
    for folder in folders:
        try:
            summaries.append(run_recording(folder, layout, out, skip_s=args.skip, make_video=args.video))
        except Exception as exc:  # keep going with the other recordings
            traceback.print_exc()
            failed.append(f"{folder}: {exc}")
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        cols = ["recording", "wearer", "block"] + [c for c in combined if c not in ("recording", "wearer", "block")]
        combined[cols].to_csv(out / "all_recordings_blocks.csv", index=False)
        print(f"Combined summary: {out / 'all_recordings_blocks.csv'}")
    if failed:
        print("\nFailed:\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
