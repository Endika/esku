"""Turn the SWL-LSE MediaPipe pickles into signature matrices, once.

Reads `data/MEDIAPIPE.zip` (8,000 pickles) and writes `data/<split>.npz`. Extraction takes
minutes and training is run many times, so it is deliberately a separate step.

The pickles were written by an older MediaPipe whose `mediapipe.framework.formats` package
no longer exists in 1.x. Only that one module is stubbed; every other class unpickles
against the real library.
"""

from __future__ import annotations

import csv
import io
import pickle
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np

from features import SIGNATURE_LENGTH, window_signature

DATA = Path(__file__).parent / "data"


def install_protobuf_stub() -> None:
    """Satisfy the pickles' reference to a module MediaPipe 1.x dropped.

    These entries carry no landmark data we use — the hand and pose results come through
    the `tasks` dataclasses, which are real — so a placeholder is enough.
    """
    module = types.ModuleType("mediapipe.framework.formats.landmark_pb2")

    class _Placeholder:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {"raw": state})

    module.LandmarkList = type("LandmarkList", (_Placeholder,), {})
    module.NormalizedLandmarkList = type("NormalizedLandmarkList", (_Placeholder,), {})

    for name in ("mediapipe.framework", "mediapipe.framework.formats"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["mediapipe.framework.formats.landmark_pb2"] = module


def concept_id(label: str) -> str:
    """Strip SWL-LSE variant markers.

    Mirrors `conceptIdOf` in `src/domain/recognition/value-objects/Gloss.ts`: `AZUCAR`,
    `AZUCAR2`, `AZUCAR(2M)`, `AZUCAR(M-ES)` and `AZUCAR(M-ES)(2M)` are all sugar. Merging
    them trains better — fewer classes, more examples each — and reads better in a
    transcript.
    """
    return re.sub(r"\d+$", "", re.sub(r"\([^)]*\)", "", label)).strip()


def load_labels() -> dict[int, str]:
    """class_id -> gloss, from the reference-video annotations."""
    labels: dict[int, str] = {}
    with open(DATA / "videos_ref_annotations.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[int(row["CLASS_ID"])] = row["LABEL"]
    return labels


def load_split(split: str) -> list[tuple[str, int]]:
    """The official splits, headerless `sample_id,class_id`."""
    with zipfile.ZipFile(DATA / "ANNOTATIONS.zip") as archive:
        raw = archive.read(f"ANNOTATIONS/{split}_labels.csv").decode("utf-8", "replace")
    rows = [line.split(",") for line in raw.strip().splitlines()]
    return [(sample.strip(), int(klass)) for sample, klass in rows if klass.strip().isdigit()]


def frames_from(pickled: list[dict]) -> list[dict[str, np.ndarray]]:
    """Pull just the hands out of each frame, keyed by side.

    Frames where MediaPipe found no hand are dropped rather than kept as zeros: about half
    of every recording is hands-down lead-in and lead-out, and resampling across those would
    spend signature slots on nothing.
    """
    frames: list[dict[str, np.ndarray]] = []
    for frame in pickled:
        hands = frame.get("hands")
        if hands is None or not hands.hand_landmarks:
            continue

        entry: dict[str, np.ndarray] = {}
        for index, landmarks in enumerate(hands.hand_landmarks):
            categories = hands.handedness[index] if index < len(hands.handedness) else None
            name = categories[0].category_name.lower() if categories else "right"
            # MediaPipe labels the mirrored selfie view, so its "left" is the user's right.
            side = "right" if name == "left" else "left"
            entry[side] = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)

        if entry:
            frames.append(entry)
    return frames


def main() -> None:
    install_protobuf_stub()
    labels = load_labels()

    with zipfile.ZipFile(DATA / "MEDIAPIPE.zip") as archive:
        available = {
            Path(name).stem: name for name in archive.namelist() if name.endswith(".pkl")
        }

        for split in ("train", "val", "test"):
            rows = load_split(split)
            signatures: list[np.ndarray] = []
            concepts: list[str] = []
            missing = 0
            empty = 0

            for sample, class_id in rows:
                entry = available.get(sample)
                if entry is None:
                    missing += 1
                    continue

                label = labels.get(class_id, "")
                # One row in the dataset is labelled `#N/A`; it is dirt, not a class.
                if not label or label == "#N/A":
                    continue

                frames = frames_from(pickle.load(io.BytesIO(archive.read(entry))))
                if not frames:
                    empty += 1
                    continue

                signatures.append(window_signature(frames))
                concepts.append(concept_id(label))

            matrix = np.stack(signatures) if signatures else np.zeros((0, SIGNATURE_LENGTH))
            np.savez_compressed(
                DATA / f"{split}.npz", x=matrix.astype(np.float32), y=np.array(concepts)
            )
            print(
                f"{split:5} kept={len(signatures):5} "
                f"concepts={len(set(concepts)):4} missing={missing} no-hands={empty}"
            )


if __name__ == "__main__":
    main()
