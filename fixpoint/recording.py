"""Load a Neon "Timeseries Data + Scene Video" export folder from Pupil Cloud."""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class SceneCamera:
    camera_matrix: np.ndarray
    distortion: np.ndarray

    def undistort(self, points: np.ndarray) -> np.ndarray:
        """Distorted scene-image pixels -> undistorted pixels (same camera matrix)."""
        pts = np.asarray(points, np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.camera_matrix, self.distortion, P=self.camera_matrix).reshape(-1, 2)

    def distort(self, undistorted: np.ndarray) -> np.ndarray:
        """Undistorted pixels -> distorted scene-image pixels (inverse of undistort)."""
        pts = np.asarray(undistorted, np.float64).reshape(-1, 2)
        norm = np.column_stack([pts, np.ones(len(pts))]) @ np.linalg.inv(self.camera_matrix).T
        out, _ = cv2.projectPoints(norm, np.zeros(3), np.zeros(3), self.camera_matrix, self.distortion)
        return out.reshape(-1, 2)

    def rays(self, undistorted: np.ndarray) -> np.ndarray:
        """Undistorted pixels -> unit viewing directions in camera coordinates."""
        pts = np.asarray(undistorted, np.float64).reshape(-1, 2)
        homogeneous = np.column_stack([pts, np.ones(len(pts))])
        rays = homogeneous @ np.linalg.inv(self.camera_matrix).T
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)


@dataclass
class Recording:
    folder: Path
    info: dict
    camera: SceneCamera
    gaze: pd.DataFrame
    events: pd.DataFrame
    world_timestamps: np.ndarray  # ns, one per video frame
    video_path: Path

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def start_ns(self) -> int:
        return int(self.info["start_time"])


def load_recording(folder: str | Path) -> Recording:
    folder = Path(folder)
    videos = sorted(folder.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No scene video (.mp4) in {folder}")
    info = json.loads((folder / "info.json").read_text())
    cam = json.loads((folder / "scene_camera.json").read_text())
    camera = SceneCamera(
        np.array(cam["camera_matrix"], np.float64),
        np.array(cam["distortion_coefficients"], np.float64).ravel(),
    )
    gaze = pd.read_csv(folder / "gaze.csv")
    events = pd.read_csv(folder / "events.csv")
    world = pd.read_csv(folder / "world_timestamps.csv")["timestamp [ns]"].to_numpy(np.int64)
    return Recording(folder, info, camera, gaze, events, world, videos[0])


def find_recordings(root: str | Path) -> list[Path]:
    """Every folder under root that looks like a Neon timeseries export."""
    root = Path(root)
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    return sorted(p for p in candidates if (p / "gaze.csv").exists() and (p / "world_timestamps.csv").exists())
