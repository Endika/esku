"""Freeze a signer-disjoint train/test split, so retraining cannot eat the only honest bench.

LSE-Health is the only material in LSE that measures Esku on real coarticulated signing, and
its annotations are also the only training set for that regime. Both uses are legitimate; using
the same signers for both is not. A random split over instances would put the same signer either
side and report a number that says "this model has seen this person", which is exactly the kind
of measurement this project has already been burned by twice.

The choice of who to hold out is a rule here rather than a judgement call, so it is inspectable
and reproducible:

- the **smallest left-handed** signer goes to test. Two signers of ten are left-handed (6 and 7)
  and holding out both would leave training with no left-handed signing at all — a worse problem
  than the one it solves, since the app's `dominantHand` reads whichever hand is working.
- then the **smallest Deaf right-handed** signers, ascending, until test holds at least 15% of
  instances. Smallest-first keeps as much data as possible in training; Deaf-first is because
  Deaf signers are who the app is for, and three of the ten are hearing interpreters.

One video, `Vo4C-3pzFFo`, is annotated to signers "3,2" — two people in one recording — so it
would contaminate whichever side it landed on and is excluded from both.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import health_bench as hb

VIDEOS = hb.CACHE.parent / "videos.csv"
OUTPUT = hb.CACHE.parent / "split.json"
TEST_SHARE = 0.15


def load_meta() -> dict[str, dict[str, str]]:
    with VIDEOS.open() as handle:
        return {row["video"]: row for row in csv.DictReader(handle)}


def choose_test_signers(
    per_signer: collections.Counter, meta: dict[str, dict[str, str]], total: int
) -> list[str]:
    profile: dict[str, tuple[str, str]] = {}
    for row in meta.values():
        profile.setdefault(row["signer"], (row["handedness"], row["deaf"]))

    left = [s for s in per_signer if profile.get(s, ("", ""))[0] == "Left"]
    if not left:
        raise SystemExit("no hay signantes zurdos en el corpus; revisa videos.csv")
    chosen = [min(left, key=lambda s: per_signer[s])]

    right_deaf = sorted(
        (s for s in per_signer if profile.get(s) == ("Right", "Deaf")),
        key=lambda s: per_signer[s],
    )
    # Take the prefix whose share lands *closest* to the target, not the first one to clear it:
    # signer sizes range from 226 to 3,070 instances, so "first over 15%" can overshoot to 29%
    # and hand a third of the corpus, plus a large Deaf signer, to a set nothing trains on.
    options = [list(chosen)]
    for signer in right_deaf:
        chosen.append(signer)
        options.append(list(chosen))
    return min(
        options,
        key=lambda group: abs(sum(per_signer[s] for s in group) / total - TEST_SHARE),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--include-starred",
        action="store_true",
        help="incluir las glosas con prefijo * (no sabemos que marca; por defecto fuera)",
    )
    parser.add_argument(
        "--test-signers",
        help="lista de signantes separados por comas, en vez de aplicar la regla",
    )
    arguments = parser.parse_args()

    annotations, gloss_rows, segment_rows = hb.load_annotations()
    if gloss_rows != hb.EXPECTED_GLOSSES or segment_rows != hb.EXPECTED_SEGMENTS:
        raise SystemExit("el parseo del xlsx no cuadra con el corpus publicado; no continuo")
    meta = load_meta()

    multi = sorted(video for video, row in meta.items() if "," in row["signer"])
    usable = {v: rows for v, rows in annotations.items() if v not in multi and v in meta}

    kept: dict[str, list] = {}
    starred = 0
    for video, rows in usable.items():
        selected = []
        for start, end, label in rows:
            if label.strip().startswith("*"):
                starred += 1
                if not arguments.include_starred:
                    continue
            selected.append((start, end, label))
        kept[video] = selected

    per_signer = collections.Counter()
    for video, rows in kept.items():
        per_signer[meta[video]["signer"]] += len(rows)
    total = sum(per_signer.values())

    if arguments.test_signers:
        test_signers = [s.strip() for s in arguments.test_signers.split(",") if s.strip()]
        unknown = [s for s in test_signers if s not in per_signer]
        if unknown:
            raise SystemExit(f"signantes desconocidos: {unknown}")
    else:
        test_signers = choose_test_signers(per_signer, meta, total)

    test_videos = sorted(v for v in kept if meta[v]["signer"] in test_signers)
    train_videos = sorted(v for v in kept if meta[v]["signer"] not in test_signers)

    def labels_of(videos: list[str]) -> collections.Counter:
        out = collections.Counter()
        for video in videos:
            for *_, label in kept[video]:
                out[label] += 1
        return out

    train_labels, test_labels = labels_of(train_videos), labels_of(test_videos)
    orphans = sorted(set(test_labels) - set(train_labels))

    print("Particion disjunta por signante de LSE-Health-UVigo")
    print()
    print(f"   glosas anotadas en total        : {gloss_rows}")
    print(f"   con prefijo *                   : {starred} "
          f"({'incluidas' if arguments.include_starred else 'EXCLUIDAS'})")
    print(f"   video multi-signante excluido   : {', '.join(multi) or 'ninguno'}")
    print(f"   instancias utilizables          : {total}")
    print()
    print(f"   {'signante':>9} {'lateralidad':>12} {'sordo/oyente':>13} "
          f"{'instancias':>11} {'conjunto':>10}")
    profile = {}
    for row in meta.values():
        profile.setdefault(row["signer"], (row["handedness"], row["deaf"]))
    for signer, count in per_signer.most_common():
        hand, deaf = profile.get(signer, ("?", "?"))
        where = "TEST" if signer in test_signers else "entrena"
        print(f"   {signer:>9} {hand:>12} {deaf:>13} {count:>11} {where:>10}")
    print()
    test_count = sum(per_signer[s] for s in test_signers)
    print(f"   test : {len(test_videos):>3} videos  {test_count:>6} instancias "
          f"({test_count / total:.1%})  {len(test_labels)} clases")
    print(f"   train: {len(train_videos):>3} videos  {total - test_count:>6} instancias "
          f"({1 - test_count / total:.1%})  {len(train_labels)} clases")
    print()
    if orphans:
        print(f"   AVISO: {len(orphans)} clases aparecen en test y no en entrenamiento, asi que")
        print("   son irrecuperables por construccion y quedan marcadas como tales:")
        print(f"   {', '.join(orphans)}")
    else:
        print("   toda clase del test tiene ejemplos en entrenamiento")
    overlap = set(test_videos) & set(train_videos)
    if overlap:
        raise SystemExit(f"un video cae en los dos conjuntos: {sorted(overlap)}")

    OUTPUT.write_text(
        json.dumps(
            {
                "test_signers": sorted(test_signers),
                "test_videos": test_videos,
                "train_videos": train_videos,
                "excluded_videos": multi,
                "starred_included": arguments.include_starred,
                "orphan_labels": orphans,
                "instances": {"train": total - test_count, "test": test_count},
            },
            indent=1,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print()
    print(f"   escrito {OUTPUT}")


if __name__ == "__main__":
    main()
