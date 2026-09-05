"""Convert one CARLA mesh-tracking clip to the WorldTrack NPZ format.

The converter keeps actor mesh points and background points in one stable point
axis.  Actor points are marked by ``point_is_dynamic`` even when an actor is
stationary; ``static_world_points`` means background geometry only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _first(pack: np.lib.npyio.NpzFile, names: Iterable[str], required: bool = True):
    for name in names:
        if name in pack.files:
            return pack[name]
    if required:
        raise KeyError(f"None of {tuple(names)!r} found; available keys: {pack.files}")
    return None


def _time_points(value: np.ndarray, frames: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[-1] == 3:
        array = np.repeat(array[None], frames, axis=0)
    if array.ndim != 3 or array.shape[0] != frames or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (T,N,3) or (N,3), got {array.shape}")
    return array.astype(np.float32, copy=False)


def _time_mask(value: np.ndarray | None, frames: int, points: int, name: str) -> np.ndarray:
    if value is None:
        return np.ones((frames, points), dtype=bool)
    array = np.asarray(value)
    if array.ndim == 1:
        array = np.repeat(array[None], frames, axis=0)
    if array.shape != (frames, points):
        raise ValueError(f"{name} must have shape (T,N) or (N,), got {array.shape}")
    return array.astype(bool, copy=False)


def _world_to_camera(points: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((*points.shape[:2], 1), dtype=np.float64)],
        axis=-1,
    )
    result = np.einsum("tij,tnj->tni", matrices.astype(np.float64), homogeneous)
    w = result[..., 3:4]
    finite_w = np.isfinite(w) & (np.abs(w) > 1e-12)
    result[..., :3] = np.divide(
        result[..., :3], w, out=np.full_like(result[..., :3], np.nan), where=finite_w
    )
    return result[..., :3].astype(np.float32)


def _read_jpeg_bytes(path: Path, quality: int) -> bytes:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read RGB image: {path}")
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Failed to encode RGB image: {path}")
    return encoded.tobytes()


def _intrinsics(raw: np.ndarray) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float32)
    if value.ndim == 1 and value.size >= 4:
        return value[:4]
    if value.ndim == 2 and value.shape[-1] >= 4:
        return value[0, :4]
    if value.shape == (3, 3):
        return np.asarray(
            [value[0, 0], value[1, 1], value[0, 2], value[1, 2]], dtype=np.float32
        )
    raise ValueError(f"Unsupported intrinsics shape: {value.shape}")


def convert_clip(clip: Path, output: Path, jpeg_quality: int = 95) -> dict[str, object]:
    tracks_path = clip / "tracks.npz"
    rgb_paths = sorted((clip / "rgb").glob("*.png"))
    if not tracks_path.is_file():
        raise FileNotFoundError(tracks_path)
    if not rgb_paths:
        raise FileNotFoundError(f"No PNG frames under {clip / 'rgb'}")

    with np.load(tracks_path, allow_pickle=True) as pack:
        dynamic = _first(pack, ("dynamic_world_points",))
        static = _first(pack, ("static_world_points",))
        w2c = np.asarray(_first(pack, ("world_to_camera", "extrinsics_w2c")), dtype=np.float32)
        if w2c.ndim == 2:
            w2c = np.repeat(w2c[None], len(rgb_paths), axis=0)
        if w2c.ndim != 3 or w2c.shape[1:] != (4, 4):
            raise ValueError(f"world_to_camera must have shape (T,4,4), got {w2c.shape}")
        frames = min(len(rgb_paths), w2c.shape[0])
        dynamic = _time_points(dynamic, frames, "dynamic_world_points")
        static = _time_points(static, frames, "static_world_points")
        dynamic_visible = _first(pack, ("dynamic_visible",), required=False)
        static_visible = _first(pack, ("static_visible",), required=False)
        intrinsics = _intrinsics(_first(pack, ("intrinsics", "fx_fy_cx_cy")))

    if dynamic.shape[0] != frames or static.shape[0] != frames:
        raise ValueError("Point arrays and camera matrices do not have the same frame count")
    dynamic_mask = _time_mask(dynamic_visible, frames, dynamic.shape[1], "dynamic_visible")
    static_mask = _time_mask(static_visible, frames, static.shape[1], "static_visible")
    world_points = np.concatenate([dynamic, static], axis=1)
    visibility = np.concatenate([dynamic_mask, static_mask], axis=1)
    tracks_cam = _world_to_camera(world_points, w2c[:frames])
    visibility &= np.isfinite(tracks_cam).all(axis=-1)
    point_is_dynamic = np.concatenate(
        [np.ones(dynamic.shape[1], dtype=bool), np.zeros(static.shape[1], dtype=bool)]
    )
    jpeg = np.asarray([_read_jpeg_bytes(path, jpeg_quality) for path in rgb_paths[:frames]], dtype=object)

    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        images_jpeg_bytes=jpeg,
        tracks_XYZ=tracks_cam,
        visibility=visibility,
        fx_fy_cx_cy=intrinsics,
        extrinsics_w2c=w2c[:frames],
        point_is_dynamic=point_is_dynamic,
    )
    summary = {
        "input_clip": str(clip),
        "output": str(output),
        "frames": frames,
        "points": int(world_points.shape[1]),
        "dynamic_points": int(dynamic.shape[1]),
        "static_points": int(static.shape[1]),
        "visible_observations": int(visibility.sum()),
        "dynamic_visible_observations": int(visibility[:, point_is_dynamic].sum()),
        "intrinsics": intrinsics.tolist(),
        "optical_flow_included": False,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, help="Directory containing tracks.npz and rgb/*.png")
    parser.add_argument("output", type=Path, help="Output WorldTrack NPZ path")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    summary = convert_clip(args.clip.resolve(), args.output.resolve(), args.jpeg_quality)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
