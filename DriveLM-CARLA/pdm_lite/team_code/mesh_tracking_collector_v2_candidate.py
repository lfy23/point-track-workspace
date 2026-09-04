"""Route-integrated dynamic and static point-track collection for CARLA."""

import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np


def _matrix(transform):
    return np.asarray(transform.get_matrix(), dtype=np.float64)


def _points(values):
    return np.asarray([[p.x, p.y, p.z] for p in values], dtype=np.float64)


class MeshTrackingCollector:
    """Collect non-overlapping source-anchored clips during a route."""

    def __init__(self, root, world, ego, camera_sensor, width, height, fov):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.world = world
        self.ego = ego
        self.camera_sensor = camera_sensor
        self.width = int(width)
        self.height = int(height)
        self.fov = float(fov)
        self.validate_flow = os.environ.get(
            "MESH_TRACKING_VALIDATE_FLOW", "0"
        ).lower() in ("1", "true", "yes", "on")
        self.focal = self.width / (2.0 * math.tan(math.radians(self.fov) / 2.0))

        self.frame_count = int(os.environ.get("MESH_TRACKING_FRAMES", 48))
        self.clip_stride = int(os.environ.get("MESH_TRACKING_STRIDE", self.frame_count))
        self.points_per_actor = int(os.environ.get("MESH_TRACKING_ACTOR_POINTS", 5000))
        self.static_count = int(os.environ.get("MESH_TRACKING_STATIC_POINTS", 20000))
        self.static_grid_columns = int(os.environ.get("MESH_TRACKING_STATIC_GRID_COLUMNS", 32))
        self.static_grid_rows = int(os.environ.get("MESH_TRACKING_STATIC_GRID_ROWS", 16))
        self.static_grid_fraction = float(os.environ.get("MESH_TRACKING_STATIC_GRID_FRACTION", 0.70))
        self.static_source_margin = int(os.environ.get("MESH_TRACKING_STATIC_SOURCE_MARGIN", 1))
        self.max_depth = float(os.environ.get("MESH_TRACKING_MAX_DEPTH", 100.0))
        self.seed = int(os.environ.get("MESH_TRACKING_SEED", 128))
        self.min_actor_points = int(os.environ.get("MESH_TRACKING_MIN_ACTOR_POINTS", 32))
        self.lod = int(os.environ.get("MESH_TRACKING_LOD", 0))
        self.motion_threshold = float(os.environ.get(
            "MESH_TRACKING_MOTION_THRESHOLD_METERS", "0.01"
        ))
        self.max_uncovered_movable_fraction = float(os.environ.get(
            "MESH_TRACKING_MAX_UNCOVERED_MOVABLE_FRACTION", "1.0"
        ))
        if self.motion_threshold < 0.0:
            raise ValueError("motion threshold cannot be negative")
        if not 0.0 <= self.max_uncovered_movable_fraction <= 1.0:
            raise ValueError(
                "maximum uncovered movable fraction must be between zero and one"
            )
        if self.clip_stride < self.frame_count:
            raise ValueError("overlapping clips are not supported")
        if self.static_grid_columns < 1 or self.static_grid_rows < 1:
            raise ValueError("static sampling grid dimensions must be positive")
        if not 0.0 <= self.static_grid_fraction <= 1.0:
            raise ValueError("static grid fraction must be between zero and one")
        if self.static_source_margin < 1:
            raise ValueError("static source margin must be at least one pixel")
        if (
            2 * self.static_source_margin >= self.width
            or 2 * self.static_source_margin >= self.height
        ):
            raise ValueError("static source margin is too large")

        self.active = None
        self.frames_until_start = 0
        self.completed = 0
        self.max_clips = int(os.environ.get("MESH_TRACKING_MAX_CLIPS", 0))
        self.require_walker = os.environ.get(
            "MESH_TRACKING_REQUIRE_WALKER", "0"
        ).lower() in ("1", "true", "yes", "on")
        self.min_walker_points = int(os.environ.get(
            "MESH_TRACKING_MIN_WALKER_POINTS",
            self.min_actor_points,
        ))
        if self.min_walker_points < self.min_actor_points:
            raise ValueError(
                "minimum walker points cannot be smaller "
                "than minimum actor points"
            )
        if self.min_walker_points > self.points_per_actor:
            raise ValueError(
                "minimum walker points cannot exceed "
                "points per actor"
            )

    @staticmethod
    def _decode_depth(raw):
        raw = raw.astype(np.float64)
        return (raw[:, :, 2] + 256.0 * raw[:, :, 1] + 65536.0 * raw[:, :, 0]) / 16777215.0 * 1000.0

    @staticmethod
    def _decode_instance(raw):
        red = raw[:, :, 2].astype(np.uint32)
        green = raw[:, :, 1].astype(np.uint32)
        blue = raw[:, :, 0].astype(np.uint32)
        return red, (red << 16) | (green << 8) | blue

    def _evaluate(self, local, actor_transform, camera_transform, depth, packed, color):
        count = len(local)
        local_h = np.column_stack((local, np.ones(count, dtype=np.float64)))
        world_h = _matrix(actor_transform).dot(local_h.T).T
        camera = np.asarray(camera_transform.get_inverse_matrix(), dtype=np.float64).dot(world_h.T).T
        forward = camera[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.focal * camera[:, 1] / forward + self.width / 2.0
            v = self.focal * -camera[:, 2] / forward + self.height / 2.0
        inside = ((forward > 0.0) & np.isfinite(u) & np.isfinite(v)
                  & (u >= -0.01) & (u < self.width + 0.01)
                  & (v >= -0.01) & (v < self.height + 0.01))
        depth_ok = np.zeros(count, dtype=np.bool_)
        instance_ok = np.zeros(count, dtype=np.bool_)
        sampled = np.zeros(count, dtype=np.uint32)
        ii = np.flatnonzero(inside)
        if len(ii):
            x = np.rint(u[ii]).astype(np.int64).clip(0, self.width - 1)
            y = np.rint(v[ii]).astype(np.int64).clip(0, self.height - 1)
            tolerance = np.maximum(0.10, 0.01 * forward[ii])
            depth_ok[ii] = depth[y, x] + tolerance >= forward[ii]
            sampled[ii] = packed[y, x]
            if color is not None:
                instance_ok[ii] = sampled[ii] == color
        return {
            "world": world_h[:, :3], "uv": np.column_stack((u, v)),
            "camera_depth": forward, "inside": inside, "depth_ok": depth_ok,
            "instance_ok": instance_ok, "visible": depth_ok & instance_ok,
            "sampled_instance": sampled,
        }

    def _evaluate_static(self, world_points, camera_transform, depth, semantic, dynamic_tags):
        count = len(world_points)
        world_h = np.column_stack((world_points, np.ones(count, dtype=np.float64)))
        camera = np.asarray(camera_transform.get_inverse_matrix(), dtype=np.float64).dot(world_h.T).T
        forward = camera[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.focal * camera[:, 1] / forward + self.width / 2.0
            v = self.focal * -camera[:, 2] / forward + self.height / 2.0
        inside = ((forward > 0.0) & np.isfinite(u) & np.isfinite(v)
                  & (u >= -0.01) & (u < self.width + 0.01)
                  & (v >= -0.01) & (v < self.height + 0.01))
        depth_ok = np.zeros(count, dtype=np.bool_)
        semantic_ok = np.zeros(count, dtype=np.bool_)
        rendered = np.full(count, np.nan, dtype=np.float32)
        ii = np.flatnonzero(inside)
        if len(ii):
            x = np.rint(u[ii]).astype(np.int64).clip(0, self.width - 1)
            y = np.rint(v[ii]).astype(np.int64).clip(0, self.height - 1)
            rendered[ii] = depth[y, x]
            tolerance = np.maximum(0.10, 0.01 * forward[ii])
            depth_ok[ii] = rendered[ii] + tolerance >= forward[ii]
            semantic_ok[ii] = ~np.isin(semantic[y, x], dynamic_tags)
        return {
            "uv": np.column_stack((u, v)), "camera_depth": forward,
            "rendered_depth": rendered, "inside": inside,
            "depth_ok": depth_ok, "semantic_ok": semantic_ok,
            "visible": inside & depth_ok & semantic_ok,
        }

    def _candidate_actors(self, camera_transform):
        origin = camera_transform.location
        result = []
        for actor in self.world.get_actors():
            if actor.id == self.ego.id:
                continue
            if not (actor.type_id.startswith("walker.pedestrian.") or actor.type_id.startswith("vehicle.")):
                continue
            if actor.get_location().distance(origin) <= self.max_depth + 20.0:
                result.append(actor)
        return result

    def _initialize_actors(
        self,
        camera_transform,
        depth,
        packed,
        frame_id,
        actor_candidates=None,
    ):
        records = []
        used_colors = set()
        if actor_candidates is None:
            actor_candidates = self._candidate_actors(
                camera_transform
            )
        for actor in actor_candidates:
            try:
                mesh = actor.get_mesh(self.lod)
                vertices = _points(mesh["vertices"])
                if not len(vertices):
                    continue
                tags = [int(value) for value in actor.semantic_tags]
                if not tags:
                    continue
                state = self._evaluate(vertices, actor.get_transform(), camera_transform, depth, packed, None)
                candidates = state["depth_ok"] & ((state["sampled_instance"] >> 16) == tags[0])
                colors, counts = np.unique(state["sampled_instance"][candidates], return_counts=True)
                nonzero = colors != 0
                colors, counts = colors[nonzero], counts[nonzero]
                if not len(colors):
                    continue
                order = np.argsort(counts)[::-1]
                color = next((int(colors[i]) for i in order if int(colors[i]) not in used_colors), None)
                if color is None:
                    continue
                visible_indices = np.flatnonzero(state["depth_ok"] & (state["sampled_instance"] == color))
                if len(visible_indices) < self.min_actor_points:
                    continue
                count = min(self.points_per_actor, len(visible_indices))
                rng = np.random.RandomState((self.seed + int(frame_id) + int(actor.id)) & 0x7fffffff)
                indices = rng.choice(visible_indices, count, replace=False).astype(np.int64)
                used_colors.add(color)
                records.append({
                    "actor": actor, "actor_id": int(actor.id), "type_id": actor.type_id,
                    "actor_class": "walker" if actor.type_id.startswith("walker.") else "vehicle",
                    "semantic_id": tags[0], "instance_color": color,
                    "indices": indices, "source_local": vertices[indices],
                    "mesh_vertices": int(len(vertices)), "tracks": [],
                })
            except RuntimeError as error:
                print("mesh tracking: skip actor {}: {}".format(actor.id, error))
        return records

    @staticmethod
    def _hybrid_static_indices(
        x, y, width, height, count, columns, rows, grid_fraction, rng
    ):
        """Mix image-grid-balanced and global random samples."""
        candidate_count = len(x)
        count = min(int(count), candidate_count)
        grid_target = min(int(round(count * grid_fraction)), count)

        grid_x = np.minimum(x * columns // width, columns - 1)
        grid_y = np.minimum(y * rows // height, rows - 1)
        grid_ids = (grid_y * columns + grid_x).astype(np.int64)

        permutation = rng.permutation(candidate_count)
        sorted_candidates = permutation[
            np.argsort(grid_ids[permutation], kind="stable")
        ]
        occupied, starts, cell_counts = np.unique(
            grid_ids[sorted_candidates],
            return_index=True,
            return_counts=True,
        )

        allocations = np.zeros(len(occupied), dtype=np.int64)
        remaining = grid_target
        while remaining:
            eligible = np.flatnonzero(allocations < cell_counts)
            if not len(eligible):
                break
            eligible = eligible[rng.permutation(len(eligible))]
            batch = eligible[:remaining]
            allocations[batch] += 1
            remaining -= len(batch)

        balanced_parts = [
            sorted_candidates[start:start + allocation]
            for start, allocation in zip(starts, allocations)
            if allocation
        ]
        balanced = (
            np.concatenate(balanced_parts)
            if balanced_parts else np.empty(0, dtype=np.int64)
        )

        selected_mask = np.zeros(candidate_count, dtype=np.bool_)
        selected_mask[balanced] = True
        global_target = count - len(balanced)
        remaining_candidates = np.flatnonzero(~selected_mask)
        global_indices = rng.choice(
            remaining_candidates,
            global_target,
            replace=False,
        ) if global_target else np.empty(0, dtype=np.int64)

        chosen = np.concatenate((balanced, global_indices))
        groups = np.concatenate((
            np.zeros(len(balanced), dtype=np.uint8),
            np.ones(len(global_indices), dtype=np.uint8),
        ))
        shuffle = rng.permutation(len(chosen))
        return chosen[shuffle], groups[shuffle], grid_ids

    def _sample_static(self, depth, semantic, camera_transform, dynamic_tags, frame_id):
        mask = np.isfinite(depth) & (depth > 0.0) & (depth <= self.max_depth)
        mask &= ~np.isin(semantic, dynamic_tags)
        margin = self.static_source_margin
        if margin:
            mask[:margin] = False
            mask[-margin:] = False
            mask[:, :margin] = False
            mask[:, -margin:] = False
        y, x = np.nonzero(mask)
        if not len(x):
            raise RuntimeError("no static source candidates")
        count = min(self.static_count, len(x))
        rng = np.random.RandomState((self.seed + int(frame_id) + 104729) & 0x7fffffff)
        chosen, sampling_group, candidate_grid_ids = self._hybrid_static_indices(
            x, y, self.width, self.height, count,
            self.static_grid_columns, self.static_grid_rows,
            self.static_grid_fraction, rng,
        )
        source_candidates = len(x)
        x, y = x[chosen], y[chosen]
        z = depth[y, x]
        source_semantic = semantic[y, x]
        source_grid_ids = candidate_grid_ids[chosen]
        camera_points = np.column_stack((z, (x - self.width / 2.0) * z / self.focal,
                                         -(y - self.height / 2.0) * z / self.focal))
        camera_h = np.column_stack((camera_points, np.ones(count, dtype=np.float64)))
        world = _matrix(camera_transform).dot(camera_h.T).T[:, :3]
        details = {
            "source_depth": z.astype(np.float32),
            "source_semantic": source_semantic.astype(np.int64),
            "source_grid_id": source_grid_ids.astype(np.int64),
            "sampling_group": sampling_group,
            "source_candidates": int(source_candidates),
            "occupied_grid_cells": int(len(np.unique(candidate_grid_ids))),
            "selected_grid_cells": int(len(np.unique(source_grid_ids))),
        }
        return world, np.column_stack((x, y)).astype(np.int64), details

    def _source_coverage(self, depth, semantic, packed, records, candidates):
        valid = (
            np.isfinite(depth)
            & (depth > 0.0)
            & (depth <= self.max_depth)
        )
        movable_semantic_ids = sorted(set(
            [10, 12]
            + [
                int(tag)
                for actor in candidates
                for tag in actor.semantic_tags
            ]
        ))
        movable = (
            valid
            & np.isin(semantic, movable_semantic_ids)
            & (packed != 0)
        )
        colors, counts = np.unique(packed[movable], return_counts=True)
        colors = colors.astype(np.uint32, copy=False)
        counts = counts.astype(np.int64, copy=False)

        slots_by_color = {
            int(record["instance_color"]): slot
            for slot, record in enumerate(records)
        }
        slots = np.asarray([
            slots_by_color.get(int(color), -1)
            for color in colors
        ], dtype=np.int64)
        covered = slots >= 0
        covered_pixels = int(counts[covered].sum())
        total_pixels = int(counts.sum())
        uncovered_pixels = total_pixels - covered_pixels
        uncovered_fraction = (
            float(uncovered_pixels) / float(total_pixels)
            if total_pixels else 0.0
        )

        semantic_ids = np.empty(len(colors), dtype=np.int64)
        for index, color in enumerate(colors):
            values, value_counts = np.unique(
                semantic[movable & (packed == color)],
                return_counts=True,
            )
            semantic_ids[index] = int(values[np.argmax(value_counts)])

        return {
            "instance_colors": colors,
            "instance_pixel_counts": counts,
            "instance_semantic_ids": semantic_ids,
            "instance_actor_slots": slots,
            "visible_instances": int(len(colors)),
            "covered_instances": int(covered.sum()),
            "uncovered_instances": int((~covered).sum()),
            "movable_pixels": total_pixels,
            "covered_movable_pixels": covered_pixels,
            "uncovered_movable_pixels": uncovered_pixels,
            "uncovered_movable_fraction": uncovered_fraction,
            "movable_semantic_ids": np.asarray(
                movable_semantic_ids, dtype=np.int64
            ),
            "candidate_actor_ids": np.asarray(
                [int(actor.id) for actor in candidates], dtype=np.int64
            ),
            "sampled_actor_ids": np.asarray(
                [record["actor_id"] for record in records], dtype=np.int64
            ),
        }

    def _motion_labels(self, records, dynamic_world):
        point_count = dynamic_world.shape[1]
        if point_count == 0:
            return (
                np.empty(0, dtype=np.bool_),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.bool_),
            )

        source = dynamic_world[0]
        finite = np.isfinite(dynamic_world).all(axis=2)
        displacement = np.linalg.norm(
            dynamic_world - source[None, :, :], axis=2
        )
        displacement[~finite] = np.nan
        with np.errstate(all="ignore"):
            max_displacement = np.nanmax(displacement, axis=0)
        max_displacement[~np.isfinite(max_displacement)] = 0.0
        point_is_moving = max_displacement > self.motion_threshold

        actor_is_moving = np.asarray([
            bool(point_is_moving[
                sum(len(item["indices"]) for item in records[:slot]):
                sum(len(item["indices"]) for item in records[:slot + 1])
            ].any())
            for slot in range(len(records))
        ], dtype=np.bool_)
        return (
            point_is_moving,
            max_displacement.astype(np.float32),
            actor_is_moving,
        )

    def _make_directory(self, frame_id):
        path = self.root / (".clip_inprogress_{}_{}".format(frame_id, uuid.uuid4().hex[:8]))
        for name in ("rgb", "depth_raw", "instance_raw"):
            (path / name).mkdir(parents=True, exist_ok=False)
        return path

    def _start(self, frame_id, timestamp, tick, camera_transform, depth, packed, semantic):
        candidates = self._candidate_actors(camera_transform)

        if self.require_walker:
            walker_candidates = [
                actor for actor in candidates
                if actor.type_id.startswith(
                    "walker.pedestrian."
                )
            ]
            if not walker_candidates:
                return False

            visible_walkers = self._initialize_actors(
                camera_transform,
                depth,
                packed,
                frame_id,
                actor_candidates=walker_candidates,
            )
            if not any(
                len(record["indices"])
                >= self.min_walker_points
                for record in visible_walkers
            ):
                return False

        actors = self._initialize_actors(
            camera_transform,
            depth,
            packed,
            frame_id,
            actor_candidates=candidates,
        )
        coverage = self._source_coverage(
            depth, semantic, packed, actors, candidates
        )
        if (
            coverage["uncovered_movable_fraction"]
            > self.max_uncovered_movable_fraction
        ):
            print(
                "mesh tracking: reject source frame={} uncovered movable "
                "pixels={:.3f}".format(
                    frame_id,
                    coverage["uncovered_movable_fraction"],
                )
            )
            return False
        dynamic_tags = sorted(set([10, 12] + [record["semantic_id"] for record in actors]))
        static_world, static_pixels, static_details = self._sample_static(
            depth, semantic, camera_transform, dynamic_tags, frame_id
        )
        self.active = {
            "directory": self._make_directory(frame_id), "frame_ids": [], "timestamps": [],
            "camera_transforms": [], "world_to_camera": [], "actors": actors,
            "dynamic_tags": np.asarray(dynamic_tags, dtype=np.int64),
            "static_world": static_world, "static_pixels": static_pixels,
            "static_details": static_details, "static_tracks": [],
            "coverage": coverage,
        }
        print("mesh tracking: started frame={} actors={} dynamic_points={} static_points={}".format(
            frame_id, len(actors), sum(len(a["indices"]) for a in actors), len(static_world)))
        return True

    def _write_images(self, directory, frame_id, tick):
        name = "{:010d}.png".format(frame_id)
        values = (("rgb", tick["rgb"]), ("depth_raw", tick["depth_raw"]),
                  ("instance_raw", tick["instance_raw"]))
        for stream, image in values:
            if image is None or not cv2.imwrite(str(directory / stream / name), image):
                raise RuntimeError("failed to save {} frame {}".format(stream, frame_id))
        if self.validate_flow:
            flow = tick.get("optical_flow")
            if flow is None or flow.shape != (
                self.height, self.width, 2
            ):
                raise RuntimeError(
                    "invalid optical flow frame {}".format(
                        frame_id
                    )
                )
            flow_dir = directory / "optical_flow"
            flow_dir.mkdir(exist_ok=True)
            np.save(
                str(flow_dir / "{:010d}.npy".format(frame_id)),
                flow.astype(np.float32, copy=False),
                allow_pickle=False,
            )

    def _append(self, frame_id, timestamp, tick, camera_transform, depth, packed, semantic):
        active = self.active
        self._write_images(active["directory"], frame_id, tick)
        active["frame_ids"].append(frame_id)
        active["timestamps"].append(float(timestamp))
        active["camera_transforms"].append(_matrix(camera_transform))
        active["world_to_camera"].append(np.asarray(camera_transform.get_inverse_matrix(), dtype=np.float64))
        frame_index = len(active["frame_ids"]) - 1
        for record in active["actors"]:
            try:
                if frame_index == 0:
                    local = record["source_local"]
                else:
                    local = _points(record["actor"].get_mesh_vertices(record["indices"].tolist(), self.lod))
                transform = record["actor"].get_transform()
                state = self._evaluate(local, transform, camera_transform, depth, packed, record["instance_color"])
                state["local"] = local
                state["actor_transform"] = _matrix(transform)
                state["exists"] = True
            except RuntimeError:
                count = len(record["indices"])
                state = {
                    "local": np.full((count, 3), np.nan), "world": np.full((count, 3), np.nan),
                    "uv": np.full((count, 2), np.nan), "camera_depth": np.full(count, np.nan),
                    "inside": np.zeros(count, bool), "depth_ok": np.zeros(count, bool),
                    "instance_ok": np.zeros(count, bool), "visible": np.zeros(count, bool),
                    "actor_transform": np.full((4, 4), np.nan), "exists": False,
                }
            record["tracks"].append(state)
        active["static_tracks"].append(self._evaluate_static(
            active["static_world"], camera_transform, depth, semantic, active["dynamic_tags"]))

    def _stack_actor(self, records, key, tail, dtype):
        if not records:
            return np.empty((self.frame_count, 0) + tail, dtype=dtype)
        frames = len(records[0]["tracks"])
        return np.concatenate([np.stack([t[key] for t in r["tracks"]], axis=0) for r in records], axis=1).astype(dtype)

    def _finish(self):
        active = self.active
        records = active["actors"]
        frame_ids = np.asarray(active["frame_ids"], dtype=np.int64)
        offsets = np.zeros(len(records) + 1, dtype=np.int64)
        if records:
            offsets[1:] = np.cumsum([len(r["indices"]) for r in records])
        point_actor_slot = np.concatenate([np.full(len(r["indices"]), i, np.int64) for i, r in enumerate(records)]) if records else np.empty(0, np.int64)
        dynamic_world = self._stack_actor(
            records, "world", (3,), np.float32
        )
        point_is_moving_actor, point_max_displacement, actor_is_moving = (
            self._motion_labels(records, dynamic_world)
        )
        actor_point_count = int(offsets[-1])
        static_point_count = int(len(active["static_world"]))
        point_is_actor = np.concatenate((
            np.ones(actor_point_count, dtype=np.bool_),
            np.zeros(static_point_count, dtype=np.bool_),
        ))
        point_is_movable_class = point_is_actor.copy()
        point_is_moving = np.concatenate((
            point_is_moving_actor,
            np.zeros(static_point_count, dtype=np.bool_),
        ))
        static = active["static_tracks"]
        if not np.all(static[0]["visible"]):
            invisible = int((~static[0]["visible"]).sum())
            raise RuntimeError(
                "static source frame contains {} invisible points".format(
                    invisible
                )
            )
        intrinsics = np.asarray([[self.focal, 0.0, self.width / 2.0],
                                 [0.0, self.focal, self.height / 2.0], [0.0, 0.0, 1.0]])
        output = active["directory"] / "tracks.npz"
        np.savez_compressed(
            str(output), frame_ids=frame_ids, timestamps=np.asarray(active["timestamps"]),
            intrinsics=intrinsics, camera_transforms=np.stack(active["camera_transforms"]),
            world_to_camera=np.stack(active["world_to_camera"]),
            actor_ids=np.asarray([r["actor_id"] for r in records], np.int64),
            actor_type_ids=np.asarray([r["type_id"] for r in records]),
            actor_classes=np.asarray([r["actor_class"] for r in records]),
            actor_semantic_ids=np.asarray([r["semantic_id"] for r in records], np.int64),
            instance_rgb=np.asarray([[(r["instance_color"] >> 16) & 255,
                                      (r["instance_color"] >> 8) & 255,
                                      r["instance_color"] & 255] for r in records], np.uint8).reshape(-1, 3),
            actor_point_offsets=offsets, point_actor_slot=point_actor_slot,
            point_actor_id=np.asarray([records[i]["actor_id"] for i in point_actor_slot], np.int64),
            point_vertex_index=np.concatenate([r["indices"] for r in records]) if records else np.empty(0, np.int64),
            dynamic_local_points=self._stack_actor(records, "local", (3,), np.float32),
            dynamic_world_points=dynamic_world,
            dynamic_image_points=self._stack_actor(records, "uv", (2,), np.float32),
            dynamic_camera_depth=self._stack_actor(records, "camera_depth", (), np.float32),
            dynamic_inside_image=self._stack_actor(records, "inside", (), np.bool_),
            dynamic_depth_visible=self._stack_actor(records, "depth_ok", (), np.bool_),
            dynamic_instance_visible=self._stack_actor(records, "instance_ok", (), np.bool_),
            dynamic_visible=self._stack_actor(records, "visible", (), np.bool_),
            dynamic_point_is_actor=np.ones(actor_point_count, dtype=np.bool_),
            dynamic_point_is_movable_class=np.ones(actor_point_count, dtype=np.bool_),
            dynamic_point_is_moving=point_is_moving_actor,
            dynamic_point_max_displacement_m=point_max_displacement,
            actor_is_moving=actor_is_moving,
            point_is_actor=point_is_actor,
            point_is_movable_class=point_is_movable_class,
            point_is_moving=point_is_moving,
            actor_transforms=np.stack([[t["actor_transform"] for t in r["tracks"]] for r in records]) if records else np.empty((0, self.frame_count, 4, 4)),
            actor_exists=(
                np.asarray([
                    [t["exists"] for t in r["tracks"]]
                    for r in records
                ], np.bool_)
                if records
                else np.empty((0, self.frame_count), dtype=np.bool_)
            ),
            static_world_points=active["static_world"], static_source_pixels=active["static_pixels"],
            static_source_depth=active["static_details"]["source_depth"],
            static_source_semantic=active["static_details"]["source_semantic"],
            static_source_grid_id=active["static_details"]["source_grid_id"],
            static_sampling_group=active["static_details"]["sampling_group"],
            static_image_points=np.stack([s["uv"] for s in static]).astype(np.float32),
            static_camera_depth=np.stack([s["camera_depth"] for s in static]).astype(np.float32),
            static_rendered_depth=np.stack([s["rendered_depth"] for s in static]).astype(np.float32),
            static_inside_image=np.stack([s["inside"] for s in static]),
            static_depth_visible=np.stack([s["depth_ok"] for s in static]),
            static_semantic_visible=np.stack([s["semantic_ok"] for s in static]),
            static_visible=np.stack([s["visible"] for s in static]),
            static_point_is_actor=np.zeros(static_point_count, dtype=np.bool_),
            static_point_is_movable_class=np.zeros(static_point_count, dtype=np.bool_),
            static_point_is_moving=np.zeros(static_point_count, dtype=np.bool_),
            coverage_instance_colors=active["coverage"]["instance_colors"],
            coverage_instance_pixel_counts=active["coverage"]["instance_pixel_counts"],
            coverage_instance_semantic_ids=active["coverage"]["instance_semantic_ids"],
            coverage_instance_actor_slots=active["coverage"]["instance_actor_slots"],
            coverage_candidate_actor_ids=active["coverage"]["candidate_actor_ids"],
            coverage_sampled_actor_ids=active["coverage"]["sampled_actor_ids"],
            coverage_movable_semantic_ids=active["coverage"]["movable_semantic_ids"],
        )
        metadata = {
            "format": "carla_mesh_tracking_route_clip_v1", "frames": int(len(frame_ids)),
            "frame_ids": frame_ids.tolist(), "first_frame": int(frame_ids[0]),
            "last_frame": int(frame_ids[-1]), "actor_count": len(records),
            "dynamic_points": int(offsets[-1]), "static_points": int(len(active["static_world"])),
            "actor_points": actor_point_count,
            "moving_points": int(point_is_moving_actor.sum()),
            "stationary_actor_points": int((~point_is_moving_actor).sum()),
            "moving_actors": int(actor_is_moving.sum()),
            "stationary_actors": int((~actor_is_moving).sum()),
            "motion_threshold_meters": self.motion_threshold,
            "point_label_order": "actor_points_then_static_points",
            "sampling": "source_visible_lod0_random_without_replacement",
            "static_sampling": "source_visible_hybrid_grid_global_without_replacement",
            "static_grid_columns": self.static_grid_columns,
            "static_grid_rows": self.static_grid_rows,
            "static_grid_fraction": self.static_grid_fraction,
            "static_global_fraction": 1.0 - self.static_grid_fraction,
            "static_source_margin_pixels": self.static_source_margin,
            "static_source_candidates": active["static_details"]["source_candidates"],
            "static_occupied_grid_cells": active["static_details"]["occupied_grid_cells"],
            "static_selected_grid_cells": active["static_details"]["selected_grid_cells"],
            "visible_movable_instances": active["coverage"]["visible_instances"],
            "covered_movable_instances": active["coverage"]["covered_instances"],
            "uncovered_movable_instances": active["coverage"]["uncovered_instances"],
            "visible_movable_pixels": active["coverage"]["movable_pixels"],
            "covered_movable_pixels": active["coverage"]["covered_movable_pixels"],
            "uncovered_movable_pixels": active["coverage"]["uncovered_movable_pixels"],
            "uncovered_movable_fraction": active["coverage"]["uncovered_movable_fraction"],
            "max_uncovered_movable_fraction": self.max_uncovered_movable_fraction,
            "mesh_lod": self.lod, "random_seed": self.seed,
            "require_walker": self.require_walker,
            "min_walker_points": self.min_walker_points,
            "width": self.width, "height": self.height, "fov": self.fov,
            "optical_flow_saved": self.validate_flow,
        }
        (active["directory"] / "metadata.json").write_text(json.dumps(metadata, indent=2))
        coverage_report = {
            "visible_instances": active["coverage"]["visible_instances"],
            "covered_instances": active["coverage"]["covered_instances"],
            "uncovered_instances": active["coverage"]["uncovered_instances"],
            "movable_pixels": active["coverage"]["movable_pixels"],
            "covered_movable_pixels": active["coverage"]["covered_movable_pixels"],
            "uncovered_movable_pixels": active["coverage"]["uncovered_movable_pixels"],
            "uncovered_movable_fraction": active["coverage"]["uncovered_movable_fraction"],
            "instances": [
                {
                    "instance_color": int(color),
                    "semantic_id": int(semantic_id),
                    "source_pixels": int(pixel_count),
                    "actor_slot": int(actor_slot),
                    "covered": bool(actor_slot >= 0),
                }
                for color, semantic_id, pixel_count, actor_slot in zip(
                    active["coverage"]["instance_colors"],
                    active["coverage"]["instance_semantic_ids"],
                    active["coverage"]["instance_pixel_counts"],
                    active["coverage"]["instance_actor_slots"],
                )
            ],
        }
        (active["directory"] / "coverage_report.json").write_text(
            json.dumps(coverage_report, indent=2)
        )
        files = sorted(p for p in active["directory"].rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        with (active["directory"] / "SHA256SUMS").open("w") as target:
            for path in files:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                target.write("{}  {}\n".format(digest, path.relative_to(active["directory"])))
        final = self.root / "clip_{}_{}".format(frame_ids[0], frame_ids[-1])
        if final.exists():
            final = self.root / (final.name + "_" + uuid.uuid4().hex[:8])
        active["directory"].rename(final)
        self.completed += 1
        print("mesh tracking: completed {} actors={} dynamic_points={} static_points={}".format(
            final, len(records), int(offsets[-1]), len(active["static_world"])))
        self.active = None

    def process(self, frame_id, timestamp, tick):
        if self.max_clips > 0 and self.completed >= self.max_clips:
            return
        frame_id = int(frame_id)
        for key in ("rgb", "depth_raw", "instance_raw", "semantics"):
            if tick.get(key) is None:
                raise RuntimeError("mesh tracking requires tick_data[{}]".format(key))
        if self.active is None:
            if self.frames_until_start > 0:
                self.frames_until_start -= 1
                return
            camera_transform = self.camera_sensor.get_transform()
            depth = self._decode_depth(tick["depth_raw"])
            _, packed = self._decode_instance(tick["instance_raw"])
            if not self._start(
                frame_id,
                timestamp,
                tick,
                camera_transform,
                depth,
                packed,
                tick["semantics"],
            ):
                return
        elif frame_id != self.active["frame_ids"][-1] + 1:
            print("mesh tracking: discarding incomplete clip after frame gap")
            self.close(discard_incomplete=True)
            return self.process(frame_id, timestamp, tick)
        camera_transform = self.camera_sensor.get_transform()
        depth = self._decode_depth(tick["depth_raw"])
        _, packed = self._decode_instance(tick["instance_raw"])
        self._append(frame_id, timestamp, tick, camera_transform, depth, packed, tick["semantics"])
        if len(self.active["frame_ids"]) == self.frame_count:
            self._finish()
            self.frames_until_start = self.clip_stride - self.frame_count

    def close(self, discard_incomplete=True):
        if self.active is not None:
            directory = self.active["directory"]
            count = len(self.active["frame_ids"])
            self.active = None
            if discard_incomplete and directory.exists():
                shutil.rmtree(str(directory))
            print("mesh tracking: closed incomplete clip frames={}".format(count))
