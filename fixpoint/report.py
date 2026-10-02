"""Figures and a self-contained HTML report for one recording."""
import base64
import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLOCK_COLORS = {"start": "#2a78d6", "end": "#eb6834"}
FALLBACK_COLORS = ["#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlesize": 10, "axes.titlecolor": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "legend.frameon": False,
})


def _color(block: str, order: list[str]) -> str:
    return BLOCK_COLORS.get(block, FALLBACK_COLORS[order.index(block) % len(FALLBACK_COLORS)])


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _short(name: str) -> str:
    """top_left -> TL, center -> C"""
    return "".join(w[0].upper() for w in name.split("_"))


def gaze_on_slide(layout, metrics: pd.DataFrame, samples: pd.DataFrame) -> str:
    blocks = list(metrics.block.unique())
    w, h = layout.size
    fig, axes = plt.subplots(1, len(blocks), figsize=(6.2 * len(blocks), 6.2 * h / w + 0.6), squeeze=False)
    for ax, block in zip(axes[0], blocks):
        color = _color(block, blocks)
        ax.imshow(layout.reference_image, extent=(0, w, h, 0), alpha=0.35)
        s = samples[samples.block == block] if len(samples) else samples
        if len(s):
            ax.scatter(s.slide_x, s.slide_y, s=3, color=color, alpha=0.25, linewidths=0, rasterized=True)
        for r in metrics[metrics.block == block].itertuples():
            if np.isfinite(getattr(r, "deg_per_slide_px_h", np.nan)):
                # 1 degree and 2 degree rings around each target for scale.
                for deg, ls in ((1, "-"), (2, ":")):
                    ax.add_patch(matplotlib.patches.Ellipse(
                        (r.target_x, r.target_y), 2 * deg / r.deg_per_slide_px_h, 2 * deg / r.deg_per_slide_px_v,
                        fill=False, color=MUTED, lw=0.8, ls=ls))
            if r.usable:
                ax.plot([r.target_x, r.gaze_x], [r.target_y, r.gaze_y], color=INK, lw=1)
                ax.scatter([r.gaze_x], [r.gaze_y], s=40, color=color, edgecolors="#fcfcfb", linewidths=2, zorder=3)
                ax.annotate(f"{r.accuracy_deg:.2f}°", (r.target_x, r.target_y), xytext=(8, 10),
                            textcoords="offset points", color=INK, fontsize=8)
            else:
                ax.annotate("no data", (r.target_x, r.target_y), xytext=(8, 10),
                            textcoords="offset points", color=MUTED, fontsize=8)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(f"{block} block: gaze samples, median gaze (dot) and error to target", loc="left")
        ax.set_xlabel("slide x [px]")
        ax.set_ylabel("slide y [px]")
    fig.text(0.01, -0.01, "Rings: 1° (solid) and 2° (dotted) of visual angle around each target.",
             color=INK_2, fontsize=8)
    return _png(fig)


BEFORE, AFTER = "#eb6834", "#2a78d6"


def before_after(layout, before: pd.DataFrame, after: pd.DataFrame, metrics: pd.DataFrame,
                 offset: dict, half_width_deg: float = 2.5) -> str:
    """Per block, a 3x3 grid of panels zoomed on each cross: gaze samples and median
    before and after the session offset correction."""
    blocks = list(metrics.block.unique())
    targets = layout.targets.set_index("target")
    deg_px = np.nanmedian(metrics[["deg_per_slide_px_h", "deg_per_slide_px_v"]].to_numpy())
    half = half_width_deg / deg_px  # panel half-width in slide px
    cols = sorted(targets.x.unique())
    rows = sorted(targets.y.unique())
    fig = plt.figure(figsize=(5.4 * len(blocks), 6.4))
    outer = fig.add_gridspec(1, len(blocks), wspace=0.12, top=0.86, bottom=0.07)
    for bi, block in enumerate(blocks):
        grid = outer[bi].subgridspec(len(rows), len(cols), wspace=0.06, hspace=0.06)
        m = metrics[metrics.block == block].set_index("target")
        for t, tr in targets.iterrows():
            ax = fig.add_subplot(grid[rows.index(tr.y), cols.index(tr.x)])
            ax.set_xlim(tr.x - half, tr.x + half)
            ax.set_ylim(tr.y + half, tr.y - half)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(GRID)
            for deg, ls in ((1, "-"), (2, ":")):
                ax.add_patch(matplotlib.patches.Circle((tr.x, tr.y), deg / deg_px, fill=False, color=MUTED,
                                                       lw=0.7, ls=ls))
            ax.plot([tr.x - 0.3 / deg_px, tr.x + 0.3 / deg_px], [tr.y, tr.y], color=INK, lw=1.2)
            ax.plot([tr.x, tr.x], [tr.y - 0.3 / deg_px, tr.y + 0.3 / deg_px], color=INK, lw=1.2)
            if t not in m.index or not m.loc[t, "usable"]:
                ax.text(0.5, 0.15, "no data", transform=ax.transAxes, ha="center", color=MUTED, fontsize=8)
                continue
            meds = []
            for df, color in ((before, BEFORE), (after, AFTER)):
                s = df[(df.block == block) & (df.target == t)]
                ax.scatter(s.slide_x, s.slide_y, s=2, color=color, alpha=0.18, linewidths=0, rasterized=True)
                meds.append((s.slide_x.median(), s.slide_y.median()))
            ax.annotate("", xy=meds[1], xytext=meds[0],
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1, shrinkA=4, shrinkB=4))
            for (mx, my), color in zip(meds, (BEFORE, AFTER)):
                ax.scatter([mx], [my], s=46, color=color, edgecolors="#fcfcfb", linewidths=2, zorder=3)
            a = after[(after.block == block) & (after.target == t)]
            acc_after = float(np.hypot(*np.median(a[["offset_h_deg", "offset_v_deg"]], axis=0)))
            ax.text(0.03, 0.97, f"{m.loc[t, 'accuracy_deg']:.2f}° → {acc_after:.2f}°", transform=ax.transAxes,
                    va="top", fontsize=7.5, color=INK)
        fig.text(outer[bi].get_position(fig).x0, 0.875, f"{block} block", fontsize=11, color=INK, ha="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markeredgecolor="#fcfcfb", markersize=8, label=l)
               for c, l in ((BEFORE, "before correction"), (AFTER, "after session correction"))]
    fig.legend(handles=handles, loc="upper left", ncol=2, bbox_to_anchor=(0.115, 0.955))
    fig.text(0.125, 0.99, f"Gaze around each cross, before and after a {offset.get('direction', '')} "
             "whole-session shift", fontsize=11, color=INK, va="top")
    fig.text(0.125, 0.03, f"Each panel is ±{half_width_deg}° around its cross (layout matches the screen). "
             "Big dots = median gaze, small dots = samples, rings = 1° (solid) and 2° (dotted). "
             "Numbers = accuracy before → after.", fontsize=8, color=INK_2)
    return _png(fig)


def accuracy_bars(metrics: pd.DataFrame) -> str:
    blocks = list(metrics.block.unique())
    targets = sorted(metrics.target.unique())
    names = metrics.drop_duplicates("target").set_index("target").target_name
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    width = 0.8 / len(blocks)
    for metric, ax, title in (("accuracy_deg", axes[0], "Accuracy (error of median gaze) [°]"),
                              ("precision_robust_sd_deg", axes[1], "Precision (robust SD of samples) [°]")):
        for i, block in enumerate(blocks):
            b = metrics[metrics.block == block].set_index("target").reindex(targets)
            x = np.arange(len(targets)) + (i - (len(blocks) - 1) / 2) * width
            ax.bar(x, b[metric], width=width - 0.04, color=_color(block, blocks), label=block)
        ax.set_xticks(np.arange(len(targets)), [f"{t}\n{_short(names[t])}" for t in targets], fontsize=8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("target (T/M/B = top/mid/bottom, L/C/R = left/centre/right)", color=MUTED, fontsize=8)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right")
    return _png(fig)


def timeline(metrics: pd.DataFrame, gaze: pd.DataFrame) -> str:
    """Gaze position over time with each target window - for checking the segmentation."""
    blocks = list(metrics.block.unique())
    fig, axes = plt.subplots(2, len(blocks), figsize=(6.2 * len(blocks), 4.6), squeeze=False, sharey="row")
    for col, block in enumerate(blocks):
        b = metrics[metrics.block == block]
        t0, t1 = b.onset_s.min() - 1, b.offset_s.max() + 1
        g = gaze[(gaze.t >= t0) & (gaze.t <= t1)]
        for row, (axis, label) in enumerate((("x", "slide x [px]"), ("y", "slide y [px]"))):
            ax = axes[row, col]
            for r in b.itertuples():
                ax.axvspan(r.onset_s, r.offset_s, color=GRID if r.timing_source != "inferred" else "#f0e6cf",
                           lw=0, alpha=0.7)
                ax.hlines(getattr(r, f"target_{axis}"), r.onset_s, r.offset_s, color=INK, lw=2)
                if row == 0:
                    ax.text((r.onset_s + r.offset_s) / 2, 1.02, str(r.target), transform=ax.get_xaxis_transform(),
                            ha="center", color=INK_2, fontsize=8)
            ax.plot(g.t, g[f"slide_{axis}"], color=_color(block, blocks), lw=1)
            ax.set_xlim(t0, t1)
            ax.set_ylabel(label)
            if axis == "y":
                ax.invert_yaxis()
                ax.set_xlabel("time since recording start [s]")
            else:
                ax.set_title(f"{block} block", loc="left", pad=14)
    fig.text(0.01, -0.02, "Black lines: target position while shown. Grey bands: timing seen in the video; "
             "beige bands: timing inferred from slide order.", color=INK_2, fontsize=8)
    fig.tight_layout()
    return _png(fig)


def _table(df: pd.DataFrame, digits: int = 2) -> str:
    if df is None or df.empty:
        return "<p class='muted'>Nothing to show.</p>"
    return df.round(digits).to_html(index=False, border=0, classes="t", na_rep="–")


def _offset_html(offset: dict, tables: dict) -> str:
    if "shift_x_px" not in offset:
        return "<p>" + html.escape(" ".join(offset.get("flags", []))) + "</p>"
    verdict = ("<p><b>Recommended</b>: the shift is consistent and helps on held-out targets.</p>"
               if offset["recommended"] else
               "<p><b>Use with care</b>:</p><ul>"
               + "".join(f"<li>{html.escape(f)}</li>" for f in offset["flags"]) + "</ul>")
    gap = offset["start_end_shift_difference_deg"]
    rows = {
        "Shift (gaze relative to targets)": offset["direction"],
        "Shift in scene-camera pixels (x, y)": f"{offset['shift_x_px']:.1f}, {offset['shift_y_px']:.1f}",
        "Targets agreeing in direction (horizontal / vertical)":
            f"{offset['agreement_h']:.0%} / {offset['agreement_v']:.0%} of {offset['n_targets']}",
        "Start vs end shift difference": "–" if gap is None else f"{gap:.2f}°",
        "Mean accuracy: raw → corrected (held-out targets)":
            f"{offset['raw_mean_accuracy_deg']:.2f}° → {offset['loo_corrected_mean_accuracy_deg']:.2f}°",
        "A · Pupil Cloud / Companion: current gaze offset [px]":
            ", ".join(f"{v:.1f}" for v in offset["device_gaze_offset_px"]),
        "A · enter this new gaze offset [px]": ", ".join(f"{v:.1f}" for v in offset["new_device_gaze_offset_px"]),
        "B · corrected files": "corrected/gaze.csv and corrected/fixations.csv (same columns as the export)",
    }
    dl = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in rows.items())
    return (f"{verdict}<dl>{dl}</dl><h3>By block</h3>{_table(tables['offset_blocks'])}"
            f"<h3>By target</h3>{_table(tables['offset_targets'])}"
            "<p class='muted'>Corrected = whole-session shift applied (fitted on all targets). Held-out = shift "
            "fitted without that target, an honest estimate for unseen gaze. After entering option A in Cloud, "
            "re-export and re-run: the remaining shift should be close to zero.</p>")


def write_html(path, rec, info: dict, notes: list[str], tables: dict, figures: dict, offset: dict):
    css = """
    body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#f9f9f7;color:#0b0b0b}
    main{max-width:1200px;margin:auto;padding:24px 16px}
    h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:32px 0 8px} h3{font-size:14px;margin:16px 0 4px}
    .muted{color:#52514e} .card{background:#fcfcfb;border:1px solid #e1e0d9;border-radius:8px;padding:12px;overflow-x:auto}
    table.t{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
    table.t th,table.t td{padding:4px 8px;border-bottom:1px solid #e1e0d9;text-align:right;white-space:nowrap}
    table.t th{color:#52514e;font-weight:600} img{max-width:100%}
    dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px;margin:0} dt{color:#52514e}
    ul{margin:0;padding-left:20px}
    """
    meta = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in info.items())
    note_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    sections = [
        ("Summary per block", _table(tables["blocks"])),
        ("Session offset correction", _offset_html(offset, tables)
         + (f"<img src='data:image/png;base64,{figures['before_after']}'>" if figures.get("before_after") else "")),
        ("Gaze on the calibration slide", f"<img src='data:image/png;base64,{figures['slide']}'>"),
        ("Accuracy and precision per target", f"<img src='data:image/png;base64,{figures['bars']}'>"),
        ("Drift: end block minus start block", _table(tables["drift"])),
        ("Correction models (errors in °)", _table(tables["corrections"])),
        ("Segmentation check", f"<img src='data:image/png;base64,{figures['timeline']}'>"),
        ("All per-target metrics", _table(tables["targets"])),
    ]
    body = "".join(f"<h2>{t}</h2><div class='card'>{c}</div>" for t, c in sections)
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FixPoint calibration – {html.escape(rec.name)}</title><style>{css}</style></head>
<body><main><h1>Calibration check: {html.escape(rec.name)}</h1>
<p class="muted">Accuracy = angle between the median gaze and the target. Precision = spread of samples.
Positive horizontal offset = gaze right of target; positive vertical = gaze above target.</p>
<div class="card"><dl>{meta}</dl></div>
<h2>Notes</h2><div class="card"><ul>{note_html}</ul></div>{body}</main></body></html>""", encoding="utf-8")
