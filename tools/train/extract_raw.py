"""Cache the raw per-frame landmarks, so feature ideas can be tried without re-reading 3.5 GB.

`extract.py` bakes one feature layout straight into a matrix, which is what the shipped model
needs. This writes the layer underneath it — hands *and* pose, frame by frame — so an
experiment can change the features and retrain in seconds instead of minutes.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path

import numpy as np

from extract import DATA, concept_id, install_protobuf_stub, load_labels, load_split
from face import FACE_COUNT, parse_landmark_list, select

POSE_LANDMARKS = 33


def raw_frames(
    pickled: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per frame: right hand, left hand, pose, face. Missing parts are all-zero."""
    rights: list[np.ndarray] = []
    lefts: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    faces: list[np.ndarray] = []

    for frame in pickled:
        hands = frame.get("hands")
        if hands is None or not hands.hand_landmarks:
            continue

        right = np.zeros((21, 3), dtype=np.float32)
        left = np.zeros((21, 3), dtype=np.float32)
        for index, landmarks in enumerate(hands.hand_landmarks):
            categories = hands.handedness[index] if index < len(hands.handedness) else None
            name = categories[0].category_name.lower() if categories else "left"
            points = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
            # No inversion. The belief that MediaPipe labels a mirrored selfie view was written
            # here, in `extract.py`, in `check_calse.py` and in the app, and it is wrong: this
            # corpus is filmed in the third person and the label is already anatomical. Measured
            # against Pose's own wrists, inverting put the anatomical *left* hand in the right
            # block 97.6% of the time — while LSE-Health, extracted without the inversion, sits
            # at 98.4% correct. Training on both at once taught the model two opposite
            # conventions, which is what `check_convention.py` now exists to prevent.
            if name == "right":
                right = points
            else:
                left = points

        pose = np.zeros((POSE_LANDMARKS, 3), dtype=np.float32)
        result = frame.get("pose")
        if result is not None and result.pose_landmarks:
            pose = np.array([[p.x, p.y, p.z] for p in result.pose_landmarks[0]], dtype=np.float32)

        # The legacy Holistic face mesh survives only as raw protobuf; `face.py` decodes it.
        face = np.zeros((FACE_COUNT, 3), dtype=np.float32)
        legacy = frame.get("holistic_legacy") or {}
        mesh = legacy.get("face_landmarks")
        payload = getattr(mesh, "serialized", None) if mesh is not None else None
        if payload:
            face = select(parse_landmark_list(payload))

        rights.append(right)
        lefts.append(left)
        poses.append(pose)
        faces.append(face)

    empty = (np.zeros((0, 21, 3), np.float32), np.zeros((0, 21, 3), np.float32),
             np.zeros((0, POSE_LANDMARKS, 3), np.float32),
             np.zeros((0, FACE_COUNT, 3), np.float32))
    if not rights:
        return empty
    return np.stack(rights), np.stack(lefts), np.stack(poses), np.stack(faces)


def main() -> None:
    install_protobuf_stub()
    labels = load_labels()

    with zipfile.ZipFile(DATA / "MEDIAPIPE.zip") as archive:
        available = {Path(n).stem: n for n in archive.namelist() if n.endswith(".pkl")}

        for split in ("train", "val", "test"):
            bundle: dict[str, np.ndarray] = {}
            concepts: list[str] = []
            kept = 0

            for sample, class_id in load_split(split):
                entry = available.get(sample)
                label = labels.get(class_id, "")
                if entry is None or not label or label == "#N/A":
                    continue

                right, left, pose, face = raw_frames(
                    pickle.load(io.BytesIO(archive.read(entry)))
                )
                if len(right) == 0:
                    continue

                bundle[f"r{kept}"] = right
                bundle[f"l{kept}"] = left
                bundle[f"p{kept}"] = pose
                bundle[f"f{kept}"] = face
                concepts.append(concept_id(label))
                kept += 1

            bundle["y"] = np.array(concepts)
            bundle["n"] = np.array([kept])
            np.savez_compressed(DATA / f"{split}_raw.npz", **bundle)
            print(f"{split:5} {kept:5} samples cached")


if __name__ == "__main__":
    main()
