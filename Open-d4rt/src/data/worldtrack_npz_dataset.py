"""Training dataset adapter for WorldTrack-format NPZ clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .depth_query_builder import build_queries_from_trajectories
from .raw_augment import build_augment_info


@dataclass
class WorldTrackNPZConfig:
    root: Path | None
    manifest_paths: list[str] | None
    split: str
    clip_frames: int
    image_size: tuple[int, int]
    queries_per_clip: int
    hard_query_ratio: float
    prob_t_tgt_equals_t_cam: float
    t_src_tgt_delta_choices: tuple[int | None, ...] | None = None
    t_src_tgt_delta_probs: tuple[float, ...] | None = None
    training: bool = False
    samples_per_file: int = 40
    dynamic_query_ratio: float | None = None
    seed: int = 42


def _decode_jpeg(frame: Any) -> np.ndarray:
    if isinstance(frame, np.ndarray) and frame.dtype != object:
        encoded = np.asarray(frame, dtype=np.uint8).reshape(-1)
    else:
        if isinstance(frame, np.ndarray) and frame.shape == ():
            frame = frame.item()
        if not isinstance(frame, (bytes, bytearray, memoryview)):
            raise TypeError(f"Unsupported JPEG payload type: {type(frame)!r}")
        encoded = np.frombuffer(frame, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("Failed to decode JPEG payload")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _intrinsics_to_k(raw: np.ndarray, frames: int) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float32)
    if value.ndim == 1:
        value = np.repeat(value[None], frames, axis=0)
    if value.ndim == 2 and value.shape[-1] >= 4:
        if value.shape[0] == 1:
            value = np.repeat(value, frames, axis=0)
        fx, fy, cx, cy = (value[:frames, i] for i in range(4))
        k = np.zeros((frames, 3, 3), dtype=np.float32)
        k[:, 0, 0] = fx
        k[:, 1, 1] = fy
        k[:, 0, 2] = cx
        k[:, 1, 2] = cy
        k[:, 2, 2] = 1.0
        return k
    if value.ndim == 3 and value.shape[1:] == (3, 3):
        if value.shape[0] == 1:
            value = np.repeat(value, frames, axis=0)
        return value[:frames].astype(np.float32)
    raise ValueError(f"Unsupported fx_fy_cx_cy shape: {value.shape}")


def _discover_npz(root: Path | None, manifest_paths: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    for raw in manifest_paths or []:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.npz")))
        elif path.suffix.lower() == ".npz":
            paths.append(path)
        elif path.is_file():
            for line in path.read_text().splitlines():
                item = line.strip()
                if item and not item.startswith("#"):
                    paths.append(Path(item))
        else:
            raise FileNotFoundError(path)
    if not paths and root is not None:
        if root.is_file() and root.suffix.lower() == ".npz":
            paths = [root]
        elif root.is_dir():
            paths = sorted(root.glob("*.npz"))
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing WorldTrack NPZ: {missing[0]}")
    if not unique:
        raise ValueError("No WorldTrack NPZ files were found")
    return unique


class WorldTrackNPZDataset(Dataset):
    def __init__(self, cfg: WorldTrackNPZConfig):
        self.cfg = cfg
        self.paths = _discover_npz(cfg.root, cfg.manifest_paths)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        multiplier = max(1, int(self.cfg.samples_per_file)) if self.cfg.training else 1
        return len(self.paths) * multiplier

    def _rng(self, index: int) -> np.random.Generator:
        seed = int(self.cfg.seed) + self.epoch * 1_000_003 + int(index) * 97
        return np.random.default_rng(seed)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[int(index) % len(self.paths)]
        with np.load(path, allow_pickle=True) as pack:
            images = np.asarray(pack["images_jpeg_bytes"])
            tracks_cam = np.asarray(pack["tracks_XYZ"], dtype=np.float32)
            visibility = np.asarray(pack["visibility"], dtype=bool)
            intrinsics_raw = np.asarray(pack["fx_fy_cx_cy"])
            point_is_dynamic = (
                np.asarray(pack["point_is_dynamic"], dtype=bool).reshape(-1)
                if "point_is_dynamic" in pack.files
                else None
            )
            w2c_raw = (
                np.asarray(pack["extrinsics_w2c"], dtype=np.float32)
                if "extrinsics_w2c" in pack.files
                else None
            )

        frames = min(
            int(self.cfg.clip_frames), len(images), tracks_cam.shape[0], visibility.shape[0]
        )
        if frames != int(self.cfg.clip_frames):
            raise ValueError(f"{path} has {frames} usable frames, expected {self.cfg.clip_frames}")
        images = images[:frames]
        tracks_cam = tracks_cam[:frames]
        visibility = visibility[:frames]

        decoded = [_decode_jpeg(frame) for frame in images]
        src_h, src_w = decoded[0].shape[:2]
        out_h, out_w = self.cfg.image_size
        if any(image.shape[:2] != (src_h, src_w) for image in decoded):
            raise ValueError(f"Inconsistent image dimensions in {path}")
        video = np.stack(
            [cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA) for image in decoded]
        ).astype(np.float32) / 255.0
        video = np.transpose(video, (0, 3, 1, 2))

        k_seq = _intrinsics_to_k(intrinsics_raw, frames)
        sx = float(out_w) / float(src_w)
        sy = float(out_h) / float(src_h)
        k_seq[:, 0, 0] *= sx
        k_seq[:, 0, 2] *= sx
        k_seq[:, 1, 1] *= sy
        k_seq[:, 1, 2] *= sy

        if w2c_raw is None:
            w2c = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
        else:
            w2c = w2c_raw[:frames].copy()
            w2c = np.asarray([item @ np.linalg.inv(w2c[0]) for item in w2c], dtype=np.float32)
        t_wc = np.asarray([np.linalg.inv(item) for item in w2c], dtype=np.float32)
        camera_valid = np.isfinite(k_seq).all(axis=(1, 2)) & np.isfinite(t_wc).all(axis=(1, 2))

        tracks_world = np.empty_like(tracks_cam, dtype=np.float32)
        for frame in range(frames):
            tracks_world[frame] = (
                tracks_cam[frame] @ t_wc[frame, :3, :3].T + t_wc[frame, :3, 3]
            )
        track_valid = np.isfinite(tracks_world).all(axis=-1)

        rng = self._rng(index)
        requested_ratio = (
            self.cfg.dynamic_query_ratio if self.cfg.training else None
        )
        query, target, mask, query_stats = build_queries_from_trajectories(
            rng=rng,
            traj_3d_world=tracks_world,
            traj_visible=visibility,
            traj_valid=track_valid,
            traj_is_dynamic=point_is_dynamic,
            dynamic_query_ratio=requested_ratio,
            k_seq=k_seq,
            t_wc_seq=t_wc,
            camera_valid=camera_valid,
            depth=None,
            depth_valid=None,
            queries_per_clip=int(self.cfg.queries_per_clip),
            hard_query_ratio=0.0,
            prob_t_tgt_equals_t_cam=float(self.cfg.prob_t_tgt_equals_t_cam),
            t_src_tgt_delta_choices=self.cfg.t_src_tgt_delta_choices,
            t_src_tgt_delta_probs=self.cfg.t_src_tgt_delta_probs,
        )

        supervised = mask["xyz_3d"].astype(bool)
        dynamic_supervised = query_stats["is_dynamic_query"].astype(bool) & supervised
        actual_ratio = (
            float(dynamic_supervised.sum() / supervised.sum())
            if self.cfg.training and supervised.any()
            else -1.0
        )

        depth = np.zeros((frames, out_h, out_w), dtype=np.float32)
        depth_valid = np.zeros((frames, out_h, out_w), dtype=bool)
        aspect_ratio = np.asarray([src_w / max(float(src_h), 1.0)], dtype=np.float32)
        return {
            "video": torch.from_numpy(video).float(),
            "aspect_ratio": torch.from_numpy(aspect_ratio),
            "depth_m": torch.from_numpy(depth),
            "depth_valid": torch.from_numpy(depth_valid),
            "query": {k: torch.from_numpy(v).to(torch.long if k.startswith("t_") else torch.float32) for k, v in query.items()},
            "query_stats": {k: torch.from_numpy(v).bool() for k, v in query_stats.items()},
            "target": {k: torch.from_numpy(v).float() for k, v in target.items()},
            "mask": {k: torch.from_numpy(v).bool() for k, v in mask.items()},
            "camera": {
                "K": torch.from_numpy(k_seq).float(),
                "T_wc": torch.from_numpy(t_wc).float(),
                "camera_valid": torch.from_numpy(camera_valid).bool(),
            },
            "augment_info": {k: torch.from_numpy(v) for k, v in build_augment_info({}, image_hw=(out_h, out_w)).items()},
            "meta": {
                "dataset": "worldtrack_npz",
                "scene_id": path.stem,
                "clip_start": 0,
                "source_mode": "worldtrack_tracks_world",
                "native_hw": (src_h, src_w),
                "sample_key": str(path),
                "dynamic_pool_ratio": actual_ratio,
                "dynamic_query_ratio": actual_ratio,
            },
        }
