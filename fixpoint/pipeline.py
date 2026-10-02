"""Run the full calibration check on one recording and write every output."""
import base64
import dataclasses
import json
from pathlib import Path

import pandas as pd

from . import analysis, report, segmentation, session_offset, video
from .layout import SlideLayout
from .mapping import frames_table, scan_video
from .recording import load_recording


def run_recording(folder, layout: SlideLayout, out_root: Path, skip_s: float = 0.5, make_video: bool = False,
                  progress=print) -> pd.DataFrame:
    rec = load_recording(folder)
    out = Path(out_root) / rec.name
    out.mkdir(parents=True, exist_ok=True)
    progress(f"{rec.name}: scanning scene video for the calibration slides")
    maps = scan_video(rec, layout, progress=progress)
    if not maps:
        raise RuntimeError("Calibration slides (tags " + ", ".join(map(str, sorted(layout.tags)))
                           + ") were never visible in the scene video")

    notes = []
    pres = segmentation.presentations_from_events(rec.events)
    if pres is None:
        states = segmentation.frame_states(maps, layout)
        pres = segmentation.presentations_from_video(states, layout)
        notes.append("Target timing taken from the scene video (no cal_target_N events in this recording).")
    else:
        notes.append("Target timing taken from cal_target_N events.")
    pres = segmentation.label_blocks(pres)
    inferred = pres[pres.timing_source == "inferred"]
    if len(inferred):
        notes.append("Timing inferred from slide order (cross not visible to the scene camera) for: "
                     + ", ".join(f"{r.block_name} target {r.target}" for r in inferred.itertuples()) + ".")
    extended = pres[pres.timing_source == "video_extended"]
    if len(extended):
        notes.append("Cross only partly visible to the scene camera; onset set one typical slide duration before "
                     "its end for: " + ", ".join(f"{r.block_name} target {r.target}" for r in extended.itertuples())
                     + ".")
    n_blocks = pres.block_name.nunique()
    if n_blocks != 2:
        notes.append(f"Expected a start and an end block; found {n_blocks}.")

    progress(f"{rec.name}: analysing {len(pres)} target presentations")
    metrics, samples = analysis.analyse_presentations(rec, maps, layout, pres, skip_s=skip_s)
    blocks = analysis.block_summary(metrics)
    drift = analysis.drift(metrics)
    corrections, corrections_per_target = analysis.correction_models(metrics)
    extent = analysis.screen_extent_deg(maps, layout, rec.camera)
    offset, offset_targets = session_offset.estimate(metrics, samples, rec)
    for r in metrics[~metrics.usable].itertuples():
        notes.append(f"{r.block} target {r.target} ({r.target_name}): too little gaze data "
                     f"({r.n_samples} samples, {r.data_loss_pct:.0f}% lost) - excluded from summaries and fits.")

    # Gaze over the whole of each block, for the segmentation check plot.
    block_gaze = []
    for _, b in pres.groupby("block"):
        g = analysis.map_gaze(rec, maps, int(b.onset_ns.min() - 1e9), int(b.offset_ns.max() + 1e9))
        if len(g):
            block_gaze.append(g[g.mapped & g.worn & ~g.blink])
    gaze = pd.concat(block_gaze, ignore_index=True) if block_gaze else pd.DataFrame(columns=["timestamp_ns"])
    gaze["t"] = (gaze.timestamp_ns - rec.start_ns) / 1e9

    info = {
        "Recording": rec.name,
        "Wearer": rec.info.get("wearer_name", ""),
        "Device gaze offset already applied [scene px]": rec.info.get("gaze_offset"),
        "Analysis window": f"target onset + {skip_s:.2f} s to offset",
        "Projected slide size": f"{extent['slide_width_deg']:.1f}° × {extent['slide_height_deg']:.1f}° "
                                f"(viewing distance ≈ {extent['viewing_distance_in_screen_widths']:.2f} screen widths)",
        "Tag fit (mean corner error)": f"{metrics.tag_reprojection_error_px.mean():.2f} slide px",
    }

    metrics.to_csv(out / "targets.csv", index=False)
    blocks.to_csv(out / "blocks.csv", index=False)
    drift.to_csv(out / "drift.csv", index=False)
    corrections.to_csv(out / "corrections.csv", index=False)
    corrections_per_target.to_csv(out / "corrections_per_target.csv", index=False)
    samples.to_csv(out / "samples.csv", index=False)
    pres.to_csv(out / "presentations.csv", index=False)
    frames_table(maps, layout).to_csv(out / "frames.csv", index=False)
    offset_targets.to_csv(out / "session_offset_targets.csv", index=False)
    (out / "session_offset.json").write_text(json.dumps(offset, indent=2))
    before_after_png = None
    if "shift_x_px" in offset:
        # Option B: drop-in replacements for the export's gaze.csv / fixations.csv.
        corrected = out / "corrected"
        corrected.mkdir(exist_ok=True)
        shift = (offset["shift_x_px"], offset["shift_y_px"])
        corrected_gaze = session_offset.corrected_gaze(rec, shift)
        corrected_gaze.to_csv(corrected / "gaze.csv", index=False)
        # Same targets and windows, corrected gaze: where it lands on the slide now.
        _, samples_after = analysis.analyse_presentations(
            dataclasses.replace(rec, gaze=corrected_gaze), maps, layout, pres, skip_s=skip_s)
        before_after_png = report.before_after(layout, samples, samples_after, metrics, offset)
        (out / "before_after_correction.png").write_bytes(base64.b64decode(before_after_png))
        fixations = session_offset.corrected_fixations(rec, shift)
        if fixations is not None:
            fixations.to_csv(corrected / "fixations.csv", index=False)
    (out / "summary.json").write_text(json.dumps(
        {"info": info, "extent": extent, "notes": notes,
         "blocks": blocks.to_dict(orient="records")}, indent=2, default=str))

    target_cols = ["block", "target", "target_name", "timing_source", "n_samples", "data_loss_pct",
                   "accuracy_deg", "offset_h_deg", "offset_v_deg", "error_x_px", "error_y_px",
                   "precision_robust_sd_deg", "precision_sd_deg", "precision_rms_s2s_deg", "pct_within_1deg", "pct_within_2deg"]
    report.write_html(
        out / "report.html", rec, info, notes,
        offset=offset,
        tables=dict(blocks=blocks, offset_blocks=pd.DataFrame(offset.get("by_block", [])),
                    offset_targets=offset_targets, drift=drift, corrections=corrections,
                    targets=metrics[[c for c in target_cols if c in metrics]]),
        figures=dict(before_after=before_after_png, slide=report.gaze_on_slide(layout, metrics, samples),
                     bars=report.accuracy_bars(metrics),
                     timeline=report.timeline(metrics, gaze)),
    )
    progress(f"{rec.name}: wrote {out / 'report.html'}")
    if make_video:
        if before_after_png is None:
            progress(f"{rec.name}: no session offset could be estimated - skipping the video")
        else:
            progress(f"{rec.name}: rendering before/after gaze video")
            video.render(rec, maps, layout, pres, offset, samples, samples_after,
                         out / "gaze_before_after.mp4", skip_s=skip_s, progress=progress)
    return blocks.assign(recording=rec.name, wearer=rec.info.get("wearer_name", ""),
                         slide_width_deg=extent["slide_width_deg"],
                         session_shift_h_deg=offset.get("shift_h_deg"), session_shift_v_deg=offset.get("shift_v_deg"),
                         session_shift_x_px=offset.get("shift_x_px"), session_shift_y_px=offset.get("shift_y_px"),
                         session_shift_recommended=offset.get("recommended"),
                         session_shift_flags=" | ".join(offset.get("flags", [])))
