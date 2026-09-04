"""Preprocess Kubric MOVi-F TFDS records into scene-local numpy files.

The training dataset can read TFDS directly, but TFDS startup and per-sample
decode overhead are expensive in large mixtures. This script materializes only
the fields used by ``KubricFullRobustPreprocessDataset``:

    data/kubric_full/kubric_full_process_v1/{split}/{scene}/
        rgb.npy
        depth_uint16.npy
        segmentation.npy
        normal_uint16.npy
        object_coordinates_uint16.npy
        camera_positions.npy
        camera_quaternions.npy
        instances_bboxes_3d.npy
        instances_positions.npy
        instances_quaternions.npy
        meta.json

Run a small validation conversion first:

    python -m src.data_preprocess.prepare_kubric_full_preprocessed \
      --input-root data/kubric_full/movi-f_full/512x512/1.0.0 \
      --output-root data/kubric_full/kubric_full_process_v1_smoke \
      --splits train validation \
      --max-scenes-per-split 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_INPUT_ROOT = Path("data/kubric_full/movi-f_full/512x512/1.0.0")
DEFAULT_OUTPUT_ROOT = Path("data/kubric_full/kubric_full_process_v1")
DEFAULT_SPLITS = ("train", "validation")


ARRAY_FILES = {
    "rgb.npy": ("video",),
    "depth_uint16.npy": ("depth",),
    "segmentation.npy": ("segmentations",),
    "normal_uint16.npy": ("normal",),
    "object_coordinates_uint16.npy": ("object_coordinates",),
    "camera_positions.npy": ("camera", "positions"),
    "camera_quaternions.npy": ("camera", "quaternions"),
    "instances_bboxes_3d.npy": ("instances", "bboxes_3d"),
    "instances_positions.npy": ("instances", "positions"),
    "instances_quaternions.npy": ("instances", "quaternions"),
}


@dataclass(frozen=True)
class SavedScene:
    split: str
    scene_index: int
    relative_dir: str
    scene_id: str
    num_frames: int
    num_instances: int
    height: int
    width: int
    source_shard: str | None = None
    source_shard_index: int | None = None
    source_record_index: int | None = None


@dataclass(frozen=True)
class SourceRecord:
    shard: str
    shard_index: int
    record_index: int


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    return value


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_text(value.item())
        if value.dtype.kind in {"S", "U", "O"} and value.size == 1:
            return _decode_text(value.reshape(-1)[0])
    return str(value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _json_scalar(value.item())
        return [_json_scalar(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_json_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_scalar(v) for k, v in value.items()}
    return value


def _as_int(value: Any) -> int:
    return int(np.asarray(value).item())


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "scene"


def _scene_name(sample: dict[str, Any], index: int) -> tuple[str, str]:
    meta = sample.get("metadata", {})
    video_name = _decode_text(meta.get("video_name", f"{index:08d}"))
    return f"scene_{index:06d}", video_name


def _validate_sample(sample: dict[str, Any]) -> None:
    missing: list[str] = []
    for path in ARRAY_FILES.values():
        try:
            _nested_get(sample, path)
        except KeyError:
            missing.append(".".join(path))
    for path in (
        ("camera", "field_of_view"),
        ("metadata", "depth_range"),
        ("metadata", "num_frames"),
        ("metadata", "num_instances"),
        ("metadata", "height"),
        ("metadata", "width"),
    ):
        try:
            _nested_get(sample, path)
        except KeyError:
            missing.append(".".join(path))
    if missing:
        raise ValueError(f"Kubric sample missing required fields: {missing}")

    video = np.asarray(sample["video"])
    depth = np.asarray(sample["depth"])
    seg = np.asarray(sample["segmentations"])
    normal = np.asarray(sample["normal"])
    obj = np.asarray(sample["object_coordinates"])
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Invalid video shape: {video.shape}")
    if depth.ndim != 4 or depth.shape[-1] != 1:
        raise ValueError(f"Invalid depth shape: {depth.shape}")
    if seg.ndim != 4 or seg.shape[-1] != 1:
        raise ValueError(f"Invalid segmentations shape: {seg.shape}")
    if normal.ndim != 4 or normal.shape[-1] != 3:
        raise ValueError(f"Invalid normal shape: {normal.shape}")
    if obj.ndim != 4 or obj.shape[-1] != 3:
        raise ValueError(f"Invalid object_coordinates shape: {obj.shape}")


def _save_array(path: Path, array: Any) -> None:
    np.save(path, np.asarray(array), allow_pickle=False)


def _build_meta(sample: dict[str, Any], *, split: str, index: int, video_name: str) -> dict[str, Any]:
    metadata = sample["metadata"]
    camera = sample["camera"]
    return {
        "format": "kubric_full_process_v1",
        "source_dataset": "movi_f",
        "source_format": "tfds",
        "split": str(split),
        "scene_index": int(index),
        "video_name": video_name,
        "num_frames": _as_int(metadata["num_frames"]),
        "num_instances": _as_int(metadata["num_instances"]),
        "height": _as_int(metadata["height"]),
        "width": _as_int(metadata["width"]),
        "depth_range": _json_scalar(np.asarray(metadata["depth_range"], dtype=np.float32)),
        "camera_field_of_view": float(np.asarray(camera["field_of_view"]).item()),
    }


def save_sample(
    *,
    sample: dict[str, Any],
    output_root: Path | str,
    split: str,
    index: int,
    overwrite: bool = False,
) -> Path:
    """Save one TFDS numpy sample and return its final scene directory."""
    _validate_sample(sample)
    output_root = Path(output_root)
    scene_name, video_name = _scene_name(sample, index)
    split_dir = output_root / str(split)
    scene_dir = split_dir / scene_name

    if scene_dir.exists():
        if not overwrite:
            return scene_dir
        shutil.rmtree(scene_dir)

    split_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{scene_name}.", dir=str(split_dir)))
    try:
        for filename, field_path in ARRAY_FILES.items():
            _save_array(tmp_dir / filename, _nested_get(sample, field_path))
        meta = _build_meta(sample, split=split, index=index, video_name=video_name)
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_dir, scene_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return scene_dir


def _resolve_tfds_dir(input_root: Path) -> Path:
    if (input_root / "dataset_info.json").is_file():
        return input_root
    candidates = sorted(p for p in input_root.iterdir() if p.is_dir() and (p / "dataset_info.json").is_file())
    if not candidates:
        raise FileNotFoundError(f"No TFDS version directory with dataset_info.json under: {input_root}")
    return candidates[-1]


def _read_dataset_info(tfds_dir: Path) -> dict[str, Any]:
    path = tfds_dir / "dataset_info.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset_info.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _split_shard_lengths(tfds_dir: Path, split: str) -> list[int]:
    info = _read_dataset_info(tfds_dir)
    for item in info.get("splits", []):
        if isinstance(item, dict) and str(item.get("name")) == split:
            return [int(v) for v in item.get("shardLengths", [])]
    raise ValueError(f"Split '{split}' not found in dataset_info.json")


def _source_record_plan(
    *,
    tfds_dir: Path,
    split: str,
    shard_start: int | None = None,
    shard_end: int | None = None,
) -> list[SourceRecord]:
    lengths = _split_shard_lengths(tfds_dir, split)
    start = 0 if shard_start is None else int(shard_start)
    end = len(lengths) if shard_end is None else int(shard_end)
    if start < 0 or end < start or end > len(lengths):
        raise ValueError(f"Invalid shard range for split {split}: start={start}, end={end}, shards={len(lengths)}")
    total = len(lengths)
    plan: list[SourceRecord] = []
    for shard_index in range(start, end):
        shard = f"movi_f-{split}.tfrecord-{shard_index:05d}-of-{total:05d}"
        for record_index in range(int(lengths[shard_index])):
            plan.append(SourceRecord(shard=shard, shard_index=shard_index, record_index=record_index))
    return plan


def _load_tfds_modules() -> tuple[Any, Any]:
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except Exception as exc:
        raise ImportError(
            "This preprocessing script requires tensorflow and tensorflow_datasets to read MOVi-F TFRecords. "
            "Install them in the environment used for preprocessing, then rerun the command."
        ) from exc
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    return tf, tfds


def iter_tfds_samples(input_root: Path, split: str) -> Iterable[dict[str, Any]]:
    """Yield numpy samples from a TFDS split."""
    _, tfds = _load_tfds_modules()
    tfds_dir = _resolve_tfds_dir(input_root)
    builder = tfds.builder_from_directory(str(tfds_dir))
    if split not in builder.info.splits:
        raise ValueError(f"Split '{split}' not found in {tfds_dir}; available={list(builder.info.splits.keys())}")
    read_config = tfds.ReadConfig(try_autocache=False)
    ds = builder.as_dataset(split=split, shuffle_files=False, read_config=read_config)
    try:
        ds = ds.ignore_errors()
    except Exception:
        pass
    yield from tfds.as_numpy(ds)


def _count_split_examples(input_root: Path, split: str) -> int | None:
    try:
        _, tfds = _load_tfds_modules()
        builder = tfds.builder_from_directory(str(_resolve_tfds_dir(input_root)))
        if split not in builder.info.splits:
            return None
        return int(builder.info.splits[split].num_examples)
    except Exception:
        return None


def _split_manifest_options(
    scenes: list[SavedScene],
    *,
    max_scenes_per_split: int | None,
    shard_start: int | None,
    shard_end: int | None,
) -> dict[str, dict[str, int | None]]:
    out: dict[str, dict[str, int | None]] = {}
    for split in sorted({item.split for item in scenes}):
        split_scenes = [item for item in scenes if item.split == split]
        out[split] = {
            "max_scenes_per_split": max_scenes_per_split,
            "scene_count": len(split_scenes),
            "scene_index_base": min((item.scene_index for item in split_scenes), default=0),
            "shard_start": shard_start,
            "shard_end": shard_end,
        }
    return out


def write_manifest(
    output_root: Path,
    scenes: list[SavedScene],
    *,
    input_root: Path,
    split_options: dict[str, dict[str, int | None]] | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_root": str(input_root),
        "splits": split_options or _split_manifest_options(
            scenes,
            max_scenes_per_split=None,
            shard_start=None,
            shard_end=None,
        ),
        "target_root": str(output_root),
        "version": 1,
        "scenes": [
            {
                "split": item.split,
                "scene_index": item.scene_index,
                "relative_dir": item.relative_dir,
                "scene_id": item.scene_id,
                "num_frames": item.num_frames,
                "num_instances": item.num_instances,
                "height": item.height,
                "width": item.width,
                "source_shard": item.source_shard,
                "source_shard_index": item.source_shard_index,
                "source_record_index": item.source_record_index,
            }
            for item in scenes
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preprocess_split(
    *,
    input_root: Path,
    output_root: Path,
    split: str,
    max_scenes: int | None,
    overwrite: bool,
    shard_start: int | None = None,
    shard_end: int | None = None,
) -> list[SavedScene]:
    saved: list[SavedScene] = []
    total = _count_split_examples(input_root, split)
    limit = total if max_scenes is None else min(int(max_scenes), int(total or max_scenes))
    source_plan = _source_record_plan(tfds_dir=input_root, split=split, shard_start=shard_start, shard_end=shard_end)
    for index, sample in enumerate(iter_tfds_samples(input_root, split)):
        if max_scenes is not None and index >= int(max_scenes):
            break
        source = source_plan[index] if index < len(source_plan) else SourceRecord(shard="", shard_index=-1, record_index=-1)
        scene_dir = save_sample(sample=sample, output_root=output_root, split=split, index=index, overwrite=overwrite)
        meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
        saved.append(
            SavedScene(
                split=split,
                scene_index=index,
                relative_dir=str(scene_dir.relative_to(output_root)),
                scene_id=str(meta["video_name"]),
                num_frames=int(meta["num_frames"]),
                num_instances=int(meta["num_instances"]),
                height=int(meta["height"]),
                width=int(meta["width"]),
                source_shard=source.shard,
                source_shard_index=int(source.shard_index),
                source_record_index=int(source.record_index),
            )
        )
        print(f"[{split}] saved {index + 1}/{limit if limit is not None else '?'}: {scene_dir}", flush=True)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--max-scenes-per-split", type=int, default=None)
    parser.add_argument("--shard-start", type=int, default=None)
    parser.add_argument("--shard-end", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write manifest.json. The dataset can still discover split directories without it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = _resolve_tfds_dir(Path(args.input_root))
    output_root = Path(args.output_root)
    all_scenes: list[SavedScene] = []
    split_options: dict[str, dict[str, int | None]] = {}
    for split in args.splits:
        split_name = str(split)
        split_scenes = preprocess_split(
            input_root=input_root,
            output_root=output_root,
            split=split_name,
            max_scenes=args.max_scenes_per_split,
            overwrite=bool(args.overwrite),
            shard_start=args.shard_start,
            shard_end=args.shard_end,
        )
        all_scenes.extend(split_scenes)
        split_options[split_name] = {
            "max_scenes_per_split": args.max_scenes_per_split,
            "scene_count": len(split_scenes),
            "scene_index_base": min((item.scene_index for item in split_scenes), default=0),
            "shard_start": args.shard_start,
            "shard_end": args.shard_end,
        }
    if not args.no_manifest:
        write_manifest(output_root, all_scenes, input_root=input_root, split_options=split_options)
        print(f"Wrote manifest: {output_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
