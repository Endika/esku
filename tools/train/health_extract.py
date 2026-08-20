"""Cache LSE-Health-UVigo landmarks once, so the segmenter can be measured on real signing.

Every segmenter number so far comes from dictionary recordings: one isolated sign, a signer
who starts and ends at rest, a studio clip a few seconds long. LSE-Health-UVigo is the
opposite — 10.8 hours of continuous discourse from ten signers — and it is the only material
here that can say whether `SignSegmenter` cuts *signing* or cuts *pauses*. Reading it from
the .mp4s takes hours, so it is read exactly once into one `.npz` per video and never again.

Two facts about this corpus decide the file's shape:

- It was recorded in the **third person**, not in a selfie mirror, and *the inversion that
  `check_calse.py` and `extract_raw.py` apply unconditionally is wrong here*. That was
  measured, not reasoned, three independent ways, and `--report` reproduces all three:
  MediaPipe's raw label agrees with the anatomical wrist that Pose reports in 98.6% of
  detections; Pose puts the anatomical left shoulder at the larger image x, which is what a
  third-person camera does; and with the inversion applied all eight right-handed signers put
  most of their motion in the left slot and both left-handed ones (6 and 7) in the right slot.
  So `--no-mirror` is the default and `--mirror` is the flag you have to ask for. Getting this
  backwards is not cosmetic: every left-handed signer lands in the wrong slot and the
  segmenter's `dominantHand` reads the passive hand.

- Its rate is whatever YouTube served, not the 25 fps the paper quotes. `simulate_app` used
  to count frames instead of milliseconds and agreed with the app only at 20 fps, which is
  why the real container rate is cached per video rather than assumed.

The cache contract, because other scripts read these files: `right` and `left` are
(frames, 21, 3) float32 in **anatomical** terms — the signer's own hands, not MediaPipe's —
`pose` is (frames, 33, 3) and `face` is (frames, FACE_COUNT, 3) on the full pass only, `fps`
and `frames` are (1,) and `inverted` is (1,) recording whether the label was flipped.

    python health_extract.py --hands-only --workers 8
    python health_extract.py --report
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import multiprocessing
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import simulate_app as sim
from face import FACE_COUNT, select

MODELS = Path("../../public/models")
DATA = Path("data/uvigo")
CACHE = DATA / "cache"
ARCHIVE = DATA / "Videos-LSE-Health-UVigo.zip"
MEMBER_DIR = "Videos-LSE-Health-UVigo"
CATALOGUE = DATA / "videos.csv"

HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
#: Pose names its landmarks anatomically and carries no mirror caveat, which makes it the
#: independent referee for the whole laterality question. Shoulders say which way the camera
#: looks; wrists say which anatomical hand a hand-landmarker detection is sitting on.
POSE_SHOULDER_LEFT, POSE_SHOULDER_RIGHT = 11, 12
POSE_WRIST_LEFT, POSE_WRIST_RIGHT = 15, 16
#: Signers declared left-handed in `videos.csv`; the report's whole point is to confirm they
#: come out of the cache in the left slot.
LEFT_HANDED = {"6", "7"}


def catalogue() -> dict[str, dict[str, str]]:
    """Declared signer, duration and laterality per video id."""
    with open(CATALOGUE, newline="") as handle:
        return {row["video"]: row for row in csv.DictReader(handle)}


def _base(name: str) -> mp_python.BaseOptions:
    return mp_python.BaseOptions(model_asset_path=str(MODELS / name))


def make_detectors(hands_only: bool) -> dict:
    """One set of MediaPipe detectors. Never share these across processes."""
    detectors = {
        "hands": vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=_base("hand_landmarker.task"),
                num_hands=2,
                running_mode=vision.RunningMode.IMAGE,
            )
        )
    }
    if hands_only:
        return detectors
    detectors["pose"] = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=_base("pose_landmarker_lite.task"),
            running_mode=vision.RunningMode.IMAGE,
        )
    )
    detectors["face"] = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=_base("face_landmarker.task"),
            running_mode=vision.RunningMode.IMAGE,
        )
    )
    return detectors


def hands_of(detector, image, mirror: bool) -> tuple[np.ndarray, np.ndarray]:
    """Anatomical right and left hand, zero-filled when a hand is absent.

    `mirror` says the labels describe a mirrored view, so MediaPipe's "left" is the signer's
    right. It is a parameter and not a comment because this corpus is filmed in the third
    person, which is exactly the case the hardcoded inversion elsewhere gets wrong.
    """
    found = detector.detect(image)
    right = np.zeros((HAND_LANDMARKS, 3), dtype=np.float32)
    left = np.zeros((HAND_LANDMARKS, 3), dtype=np.float32)
    for index, points in enumerate(found.hand_landmarks):
        array = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
        label = found.handedness[index][0].category_name.lower()
        if (label == "left") == mirror:
            right = array
        else:
            left = array
    return right, left


def pose_of(detector, image) -> np.ndarray:
    found = detector.detect(image)
    if not found.pose_landmarks:
        return np.zeros((POSE_LANDMARKS, 3), dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in found.pose_landmarks[0]], dtype=np.float32)


def face_of(detector, image) -> np.ndarray:
    found = detector.detect(image)
    if not found.face_landmarks:
        return np.zeros((FACE_COUNT, 3), dtype=np.float32)
    mesh = np.array([[p.x, p.y, p.z] for p in found.face_landmarks[0]], dtype=np.float32)
    return select(mesh)


@contextlib.contextmanager
def local_copy(video: str, source: Path):
    """A real path for `video`, unpacked from the archive if `source` is a .zip.

    cv2 cannot seek a deflate stream, and unpacking all 4.7 GiB to keep 273 files around is
    wasteful when each one is read exactly once.
    """
    if source.is_dir():
        yield source / f"{video}.mp4"
        return
    staging = DATA / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=staging, suffix=".mp4", delete=False)
    try:
        with zipfile.ZipFile(source) as archive:
            with archive.open(f"{MEMBER_DIR}/{video}.mp4") as member:
                shutil.copyfileobj(member, handle)
        handle.close()
        yield Path(handle.name)
    finally:
        handle.close()
        os.unlink(handle.name)


def cached_pass(path: Path) -> str | None:
    """"hands" or "full" if the cache entry is readable and complete, else None."""
    try:
        with np.load(path) as bundle:
            keys = set(bundle.files)
            if not {"right", "left", "fps", "frames", "inverted"} <= keys:
                return None
            frames = int(bundle["frames"][0])
            if len(bundle["right"]) != frames or len(bundle["left"]) != frames:
                return None
            if "pose" not in keys or "face" not in keys:
                return "hands"
            if len(bundle["pose"]) != frames or len(bundle["face"]) != frames:
                return None
            return "full"
    except Exception:
        return None


_WORKER: dict = {}


def _start_worker(hands_only: bool, mirror: bool, source: str) -> None:
    cv2.setNumThreads(1)
    _WORKER.update(
        detectors=make_detectors(hands_only),
        hands_only=hands_only,
        mirror=mirror,
        source=Path(source),
    )


def extract(video: str) -> dict:
    """Read one video into the cache. Returns timings and what was written."""
    hands_only = _WORKER["hands_only"]
    detectors = _WORKER["detectors"]
    import mediapipe as mp

    started = time.perf_counter()
    rights: list[np.ndarray] = []
    lefts: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    faces: list[np.ndarray] = []

    with local_copy(video, _WORKER["source"]) as path:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return {"video": video, "error": "cv2 no pudo abrir el video"}
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        announced = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            right, left = hands_of(detectors["hands"], image, _WORKER["mirror"])
            rights.append(right)
            lefts.append(left)
            if not hands_only:
                poses.append(pose_of(detectors["pose"], image))
                faces.append(face_of(detectors["face"], image))
        capture.release()

    frames = len(rights)
    if frames == 0:
        return {"video": video, "error": "cero fotogramas legibles"}

    bundle = {
        "right": np.stack(rights),
        "left": np.stack(lefts),
        "fps": np.array([fps], dtype=np.float32),
        "frames": np.array([frames], dtype=np.int32),
        "inverted": np.array([1 if _WORKER["mirror"] else 0], dtype=np.int32),
    }
    if not hands_only:
        bundle["pose"] = np.stack(poses)
        bundle["face"] = np.stack(faces)

    CACHE.mkdir(parents=True, exist_ok=True)
    final = CACHE / f"{video}.npz"
    # Written aside and renamed: a run interrupted mid-write used to leave a truncated .npz
    # that the resume check happily accepted. `savez` appends ".npz" to a path, so the
    # staging file is handed over already open.
    staged = final.with_name(f"{video}.part")
    with open(staged, "wb") as handle:
        np.savez_compressed(handle, **bundle)
    os.replace(staged, final)

    elapsed = time.perf_counter() - started
    return {
        "video": video,
        "frames": frames,
        "fps": fps,
        "seconds": elapsed,
        "pass": "hands" if hands_only else "full",
        # The container's own count, not videos.csv, is what proves nothing was dropped:
        # videos.csv rounds to whole seconds and drifts up to a second from the files.
        "short_by": announced - frames,
    }


def write_index() -> dict:
    """Rebuild `cache/index.json` from whatever is on disk, so a resumed run stays honest."""
    index: dict[str, dict] = {}
    for path in sorted(CACHE.glob("*.npz")):
        with np.load(path) as bundle:
            index[path.stem] = {
                "frames": int(bundle["frames"][0]),
                "fps": float(bundle["fps"][0]),
                "pass": "hands" if "pose" not in bundle.files else "full",
                "inverted": int(bundle["inverted"][0]),
            }
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / "index.json", "w") as handle:
        json.dump(index, handle, indent=1, sort_keys=True)
    return index


def slot_stats(array: np.ndarray) -> tuple[int, float]:
    """Frames with a hand, and total fingertip travel — the segmenter's own motion measure."""
    present = int(sum(1 for frame in array if frame.any()))
    motion = 0.0
    for index in range(1, len(array)):
        motion += sim.motion_between(array[index], array[index - 1])
    return present, motion


def pose_agreement(right: np.ndarray, left: np.ndarray, pose: np.ndarray) -> tuple[int, int, int]:
    """Frames where pose sees a third-person view, and where it confirms the hand slots.

    This is the referee: if the cached `right` slot really holds the signer's right hand, its
    wrist must sit closer to pose landmark 16 than to 15. Motion shares can be argued with;
    this cannot, and it needs no declared handedness at all.
    """
    third_person = 0
    agree = 0
    checked = 0
    for index in range(len(pose)):
        points = pose[index]
        if not points.any():
            continue
        if points[POSE_SHOULDER_LEFT, 0] > points[POSE_SHOULDER_RIGHT, 0]:
            third_person += 1
        anatomical = {
            "right": points[POSE_WRIST_RIGHT, :2],
            "left": points[POSE_WRIST_LEFT, :2],
        }
        for slot, hands in (("right", right), ("left", left)):
            wrist = hands[index][0, :2]
            if not hands[index].any():
                continue
            nearer = min(anatomical, key=lambda k: float(np.linalg.norm(wrist - anatomical[k])))
            checked += 1
            agree += nearer == slot
    return third_person, agree, checked


def report(meta: dict[str, dict[str, str]], wanted: list[str] | None) -> None:
    """Validate the cache and settle the laterality question against declared handedness."""
    index = write_index()
    names = [v for v in index if wanted is None or v in wanted]
    if not names:
        print("cache vacio: no hay nada que validar")
        return

    rows: list[dict] = []
    print(f"validando {len(names)} videos del cache...")
    for video in sorted(names):
        path = CACHE / f"{video}.npz"
        level = cached_pass(path)
        with np.load(path) as bundle:
            right, left = bundle["right"], bundle["left"]
            frames = int(bundle["frames"][0])
            fps = float(bundle["fps"][0])
            inverted = int(bundle["inverted"][0])
            pose = bundle["pose"] if "pose" in bundle.files else None
        right_present, right_motion = slot_stats(right)
        left_present, left_motion = slot_stats(left)
        third, agree, checked = (0, 0, 0)
        if pose is not None:
            third, agree, checked = pose_agreement(right, left, pose)
        declared = meta.get(video, {})
        rows.append(
            {
                "video": video,
                "signer": declared.get("signer", "?"),
                "handedness": declared.get("handedness", "?"),
                "declared_seconds": float(declared.get("seconds") or 0.0),
                "frames": frames,
                "fps": fps,
                "pass": level,
                "inverted": inverted,
                "right_present": right_present,
                "left_present": left_present,
                "right_motion": right_motion,
                "left_motion": left_motion,
                "third_person": third,
                "pose_agree": agree,
                "pose_checked": checked,
            }
        )

    _print_durations(rows)
    _print_laterality(rows)
    _print_pose_referee(rows)


def _print_durations(rows: list[dict]) -> None:
    print()
    print("duracion cacheada frente a videos.csv")
    header = f"{'video':>12} {'sig':>4} {'pasada':>7} {'fps':>6} {'frames':>7}"
    print(f"{header} {'esperados':>9} {'error':>7} {'deriva':>8}")
    worst = 0.0
    worst_drift = 0.0
    for row in rows:
        expected = row["declared_seconds"] * row["fps"]
        error = abs(row["frames"] - expected) / expected if expected else float("nan")
        drift = abs(row["frames"] / row["fps"] - row["declared_seconds"])
        worst = max(worst, error if error == error else 0.0)
        worst_drift = max(worst_drift, drift)
        print(
            f"{row['video']:>12} {row['signer']:>4} {row['pass'] or 'ROTO':>7} "
            f"{row['fps']:6.2f} {row['frames']:7d} {expected:9.0f} {error * 100:6.2f}% "
            f"{drift:7.2f}s"
        )
    # `videos.csv` stores whole seconds and does not agree with the containers even at that
    # resolution, so a percentage of a 15 s clip is dominated by its rounding. The absolute
    # drift is the number that means something: a real decode failure loses seconds, not one.
    print(f"peor error relativo: {worst * 100:.2f}%   peor deriva absoluta: {worst_drift:.2f}s")
    print("(videos.csv redondea a segundos enteros; por debajo de 1s la deriva es su "
          "resolucion, no un fallo de lectura)")
    broken = [r["video"] for r in rows if r["pass"] is None]
    if broken:
        print(f"ENTRADAS ROTAS: {', '.join(broken)}")


def _print_laterality(rows: list[dict]) -> None:
    print()
    print("lateralidad: presencia y movimiento por slot ANATOMICO")
    print(
        f"{'signante':>9} {'declarada':>10} {'vids':>5} {'frames':>8} "
        f"{'pres.der':>9} {'pres.izq':>9} {'mov.der':>8} {'mov.izq':>8}"
    )

    by_signer: dict[str, list[dict]] = {}
    for row in rows:
        by_signer.setdefault(row["signer"], []).append(row)

    verdicts: dict[str, tuple[str, float]] = {}
    def order(signer: str) -> tuple[int, int, str]:
        return (0, int(signer), "") if signer.isdigit() else (1, 0, signer)

    for signer in sorted(by_signer, key=order):
        group = by_signer[signer]
        frames = sum(r["frames"] for r in group)
        right_motion = sum(r["right_motion"] for r in group)
        left_motion = sum(r["left_motion"] for r in group)
        total = right_motion + left_motion
        share = right_motion / total if total else 0.0
        declared = group[0]["handedness"]
        verdicts[signer] = (declared, share)
        print(
            f"{signer:>9} {declared:>10} {len(group):5d} {frames:8d} "
            f"{sum(r['right_present'] for r in group) / frames * 100:8.1f}% "
            f"{sum(r['left_present'] for r in group) / frames * 100:8.1f}% "
            f"{share * 100:7.1f}% {(1 - share) * 100:7.1f}%"
        )

    inverted = {r["inverted"] for r in rows}
    how = {frozenset({1}): "se invirtio", frozenset({0}): "sin invertir"}
    print()
    print(f"convencion cacheada: inverted={sorted(inverted)}  "
          f"({how.get(frozenset(inverted), 'MEZCLA')})")

    right_ok = [s for s, (d, sh) in verdicts.items() if d == "Right" and sh > 0.5]
    right_bad = [s for s, (d, sh) in verdicts.items() if d == "Right" and sh <= 0.5]
    left_ok = [s for s, (d, sh) in verdicts.items() if d == "Left" and sh < 0.5]
    left_bad = [s for s, (d, sh) in verdicts.items() if d == "Left" and sh >= 0.5]

    print(f"diestros que dominan el slot derecho: {len(right_ok)}/{len(right_ok) + len(right_bad)}"
          f"   zurdos que dominan el izquierdo: {len(left_ok)}/{len(left_ok) + len(left_bad)}")
    if not (left_ok or left_bad):
        print("VEREDICTO: INCONCLUYENTE — no hay ningun signante zurdo (6 o 7) en el cache.")
    elif not right_bad and not left_bad:
        print("VEREDICTO: la convencion por defecto es CORRECTA para este corpus.")
    elif not right_ok and not left_ok:
        print("VEREDICTO: LA CONVENCION ESTA INVERTIDA. Todos los signantes caen en el slot")
        print("           contrario a su lateralidad declarada: hay que cambiar el defecto")
        print("           (--no-mirror) y revisar check_calse.py y extract_raw.py.")
    else:
        print("VEREDICTO: INCOHERENTE — unos signantes concuerdan y otros no, asi que el slot")
        print(f"           no lo decide la lateralidad. Discrepan: diestros {right_bad}, "
              f"zurdos {left_bad}.")


def _print_pose_referee(rows: list[dict]) -> None:
    checked = sum(r["pose_checked"] for r in rows)
    if not checked:
        print()
        print("arbitro de pose: no disponible (el cache es de solo manos; hace falta la "
              "pasada completa)")
        return
    agree = sum(r["pose_agree"] for r in rows)
    third = sum(r["third_person"] for r in rows)
    frames = sum(r["frames"] for r in rows)
    print()
    print("arbitro de pose (independiente de la etiqueta de MediaPipe y de videos.csv)")
    print(f"  vista en tercera persona: {third / frames * 100:.1f}% de los fotogramas")
    print(f"  el slot cacheado coincide con la muneca anatomica de pose: "
          f"{agree}/{checked} = {agree / checked * 100:.1f}%")
    if agree / checked >= 0.9:
        print("  -> los slots del cache SON anatomicos.")
    elif agree / checked <= 0.1:
        print("  -> los slots del cache ESTAN CAMBIADOS: invierte la convencion.")
    else:
        print("  -> INCONCLUYENTE: pose no distingue las manos en este material.")


def throughput(results: list[dict], workers: int, hands_only: bool, total: int) -> None:
    frames = sum(r["frames"] for r in results)
    worker_seconds = sum(r["seconds"] for r in results)
    if not frames or not worker_seconds:
        return
    per_worker = frames / worker_seconds
    corpus_frames = 974600
    print()
    print(f"rendimiento medido ({'solo manos' if hands_only else 'pasada completa'})")
    print(f"  fotogramas: {frames}   segundos-worker: {worker_seconds:.1f}")
    print(f"  {per_worker:.1f} fotogramas/s por worker   {per_worker * workers:.1f} con {workers}")
    hours = corpus_frames / (per_worker * workers) / 3600
    print(f"  extrapolacion a los {total} videos (~{corpus_frames} fotogramas): {hours:.1f} h")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hands-only", action="store_true",
                        help="solo manos: es lo unico que necesita el segmentador")
    parser.add_argument("--workers", type=int, default=8, help="procesos en paralelo (def. 8)")
    parser.add_argument("--videos", type=int, default=0, help="limitar a los N primeros videos")
    parser.add_argument("--only", default="", help="lista de ids separados por comas")
    parser.add_argument("--mirror", dest="mirror", action="store_true", default=False,
                        help="invertir la etiqueta de MediaPipe (asume grabacion en espejo)")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false",
                        help="tomar la etiqueta tal cual: lo medido en este corpus (defecto)")
    parser.add_argument("--source", type=Path, default=ARCHIVE,
                        help="zip de los videos o directorio ya extraido")
    parser.add_argument("--report", action="store_true",
                        help="no extrae: valida el cache existente e imprime lateralidad")
    arguments = parser.parse_args()

    meta = catalogue()
    wanted = [v.strip() for v in arguments.only.split(",") if v.strip()] or None
    if wanted:
        missing = [v for v in wanted if v not in meta]
        if missing:
            parser.error(f"ids ausentes de videos.csv: {', '.join(missing)}")

    if arguments.report:
        report(meta, wanted)
        return

    if not arguments.source.exists():
        parser.error(f"no existe {arguments.source}")

    todo = wanted if wanted else sorted(meta)
    if arguments.videos:
        todo = todo[: arguments.videos]

    level = "hands" if arguments.hands_only else "full"
    pending = []
    for video in todo:
        done = cached_pass(CACHE / f"{video}.npz")
        if done == "full" or done == level:
            continue
        pending.append(video)
    skipped = len(todo) - len(pending)
    print(f"{len(todo)} videos pedidos, {skipped} ya en cache, {len(pending)} por extraer "
          f"({'solo manos' if arguments.hands_only else 'pasada completa'}, "
          f"mirror={arguments.mirror})")

    results: list[dict] = []
    if pending:
        started = time.perf_counter()
        workers = max(1, min(arguments.workers, len(pending)))
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            workers,
            initializer=_start_worker,
            initargs=(arguments.hands_only, arguments.mirror, str(arguments.source)),
        ) as pool:
            for done, result in enumerate(pool.imap_unordered(extract, pending), start=1):
                if "error" in result:
                    print(f"[{done}/{len(pending)}] {result['video']}: {result['error']}")
                    continue
                results.append(result)
                short = f"  FALTAN {result['short_by']}" if result["short_by"] else ""
                print(f"[{done}/{len(pending)}] {result['video']} "
                      f"{result['frames']} fotogramas a {result['fps']:.2f} fps "
                      f"en {result['seconds']:.1f}s "
                      f"({result['frames'] / result['seconds']:.1f} fps de proceso){short}")
        wall = time.perf_counter() - started
        print(f"\n{len(results)} videos extraidos en {wall:.1f}s de reloj con {workers} workers")
        throughput(results, workers, arguments.hands_only, len(meta))

    index = write_index()
    print(f"cache/index.json actualizado: {len(index)} videos")


if __name__ == "__main__":
    main()
