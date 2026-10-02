"""Read the calibration slides: AprilTag corners and cross centres in slide pixels."""
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pupil_apriltags import Detector

TAG_FAMILY = "tag36h11"
DEFAULT_SLIDES = Path(__file__).parent / "slides"
SLIDE_PATTERN = re.compile(r"calibration_target_(\d+)_\d+_(\w+)\.png$")


@dataclass
class SlideLayout:
    """Where everything sits on the projected slide, in slide pixels."""

    size: tuple[int, int]  # (width, height)
    tags: dict[int, np.ndarray]  # tag id -> (4, 2) corners in detector order
    targets: pd.DataFrame  # target, name, x, y
    reference_image: np.ndarray  # RGB slide with all targets, for plotting


def detect_tags(gray: np.ndarray, detector: Detector) -> dict[int, np.ndarray]:
    return {d.tag_id: d.corners.astype(np.float64) for d in detector.detect(gray)}


def red_mask(bgr: np.ndarray) -> np.ndarray:
    b, g, r = (bgr[..., i].astype(np.int16) for i in range(3))
    return (r > 150) & (r - g > 80) & (r - b > 80)


def load_calibration_slides(folder: str | Path) -> SlideLayout:
    folder = Path(folder)
    detector = Detector(families=TAG_FAMILY)
    rows, tags, size = [], None, None
    for path in sorted(folder.glob("calibration_target_*.png")):
        match = SLIDE_PATTERN.search(path.name)
        if not match:
            continue
        bgr = cv2.imread(str(path))
        slide_tags = detect_tags(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), detector)
        if tags is None:
            tags, size = slide_tags, (bgr.shape[1], bgr.shape[0])
        elif set(slide_tags) != set(tags) or any(
            not np.allclose(slide_tags[i], tags[i], atol=1) for i in tags
        ):
            raise ValueError(f"{path.name} has a different tag layout from the other slides")
        ys, xs = np.nonzero(red_mask(bgr))
        if len(xs) == 0:
            raise ValueError(f"No red cross found in {path.name}")
        rows.append(
            dict(
                target=int(match.group(1)),
                name=match.group(2),
                x=(xs.min() + xs.max()) / 2,
                y=(ys.min() + ys.max()) / 2,
            )
        )
    if not rows:
        raise FileNotFoundError(f"No calibration_target_*.png slides in {folder}")
    if len(tags) < 2:
        raise ValueError("Need at least two AprilTags on the calibration slides")

    reference = folder / "reference_all_9_targets.png"
    if reference.exists():
        image = cv2.cvtColor(cv2.imread(str(reference)), cv2.COLOR_BGR2RGB)
    else:
        image = np.full((size[1], size[0], 3), 255, np.uint8)
    targets = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)
    return SlideLayout(size=size, tags=tags, targets=targets, reference_image=image)
