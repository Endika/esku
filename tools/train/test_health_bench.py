"""Tests for the IoU pairing that `health_bench.py` reports recall and precision from.

The pairing rule is where a benchmark can flatter itself without anyone noticing: allow one
window to claim every gloss it straddles and a segmenter that emits one window per sentence
scores perfect recall; allow one gloss to absorb a burst of windows and over-firing vanishes
from the precision. So the one-to-one constraint is asserted here on synthetic intervals,
where the right answer is countable by hand, rather than inferred from corpus output.

Run it before trusting any number the bench prints:

    .venv/bin/python test_health_bench.py
"""

from __future__ import annotations

import unittest
import zipfile

import numpy as np

import health_bench as bench
import simulate_app as sim


class IouTest(unittest.TestCase):
    def test_identical_spans_are_one(self) -> None:
        self.assertEqual(bench.iou((0, 1000), (0, 1000)), 1.0)

    def test_disjoint_spans_are_zero(self) -> None:
        self.assertEqual(bench.iou((0, 400), (400, 800)), 0.0)
        self.assertEqual(bench.iou((0, 400), (900, 1000)), 0.0)

    def test_containment_is_the_length_ratio(self) -> None:
        self.assertAlmostEqual(bench.iou((0, 250), (0, 1000)), 0.25)

    def test_is_symmetric(self) -> None:
        self.assertEqual(bench.iou((100, 700), (400, 900)), bench.iou((400, 900), (100, 700)))


class MatchTest(unittest.TestCase):
    def test_exact_overlap_pairs(self) -> None:
        pairs = bench.match([(0, 500)], [(0, 500)], 0.5)
        self.assertEqual([(0, 0)], [(g, w) for g, w, _ in pairs])
        self.assertEqual(1.0, pairs[0][2])

    def test_partial_overlap_just_above_threshold_pairs(self) -> None:
        # 600 of 1400 shared -> 600/1400 = 0.4286
        self.assertAlmostEqual(bench.iou((0, 1000), (400, 1400)), 600 / 1400)
        self.assertEqual(1, len(bench.match([(0, 1000)], [(400, 1400)], 0.42)))

    def test_partial_overlap_just_below_threshold_does_not_pair(self) -> None:
        self.assertEqual([], bench.match([(0, 1000)], [(400, 1400)], 0.43))

    def test_window_covering_two_glosses_claims_only_one(self) -> None:
        glosses = [(0, 400), (400, 900)]
        pairs = bench.match(glosses, [(0, 900)], 0.1)
        self.assertEqual(1, len(pairs))
        # The longer gloss wins: 500/900 beats 400/900.
        self.assertEqual(1, pairs[0][0])

    def test_gloss_covered_by_two_windows_claims_only_one(self) -> None:
        windows = [(0, 900), (0, 1000)]
        pairs = bench.match([(0, 1000)], windows, 0.1)
        self.assertEqual(1, len(pairs))
        self.assertEqual(1, pairs[0][1])

    def test_no_overlap_pairs_nothing(self) -> None:
        self.assertEqual([], bench.match([(0, 400)], [(500, 900)], 0.5))

    def test_zero_threshold_still_requires_real_overlap(self) -> None:
        """A threshold of 0 must not pair intervals that merely touch."""
        self.assertEqual([], bench.match([(0, 400)], [(400, 800)], 0.0))

    def test_pairing_is_globally_greedy_not_first_come(self) -> None:
        """The best IoU wins the window even when a worse gloss is considered first."""
        glosses = [(0, 1000), (500, 1000)]
        pairs = bench.match(glosses, [(500, 1000)], 0.3)
        self.assertEqual([(1, 0)], [(g, w) for g, w, _ in pairs])

    def test_two_clean_signs_pair_one_to_one(self) -> None:
        glosses = [(0, 400), (1000, 1400)]
        windows = [(0, 400), (1000, 1400)]
        self.assertEqual(2, len(bench.match(glosses, windows, 0.5)))

    def test_over_firing_caps_precision(self) -> None:
        """Four windows over one gloss: recall 1 of 1, precision 1 of 4."""
        windows = [(0, 400), (0, 400), (0, 400), (0, 400)]
        self.assertEqual(1, len(bench.match([(0, 400)], windows, 0.5)))

    def test_is_deterministic_across_runs(self) -> None:
        glosses = [(0, 500), (300, 800), (700, 1200)]
        windows = [(100, 600), (250, 900), (650, 1300)]
        first = bench.match(glosses, windows, 0.1)
        self.assertEqual(first, bench.match(glosses, windows, 0.1))

    def test_empty_inputs(self) -> None:
        self.assertEqual([], bench.match([], [(0, 400)], 0.5))
        self.assertEqual([], bench.match([(0, 400)], [], 0.5))


class AnnotationTest(unittest.TestCase):
    """The published row counts. A truncated parse would shrink every recall denominator."""

    def test_row_counts_match_the_corpus(self) -> None:
        if not bench.WORKBOOK.is_file():
            self.skipTest(f"no workbook at {bench.WORKBOOK}")
        by_video, glosses, segments = bench.load_annotations()
        self.assertEqual(bench.EXPECTED_GLOSSES, glosses)
        self.assertEqual(bench.EXPECTED_SEGMENTS, segments)
        self.assertEqual(glosses, sum(len(v) for v in by_video.values()))

    def test_sheets_are_resolved_by_name(self) -> None:
        if not bench.WORKBOOK.is_file():
            self.skipTest(f"no workbook at {bench.WORKBOOK}")
        with zipfile.ZipFile(bench.WORKBOOK) as book:
            header = bench.sheet_rows(book, "GlossesContent")[0]
            self.assertEqual("Gloss", header["D"])
            with self.assertRaises(KeyError):
                bench.sheet_rows(book, "NoSuchSheet")

    def test_gloss_durations_are_the_documented_distribution(self) -> None:
        if not bench.WORKBOOK.is_file():
            self.skipTest(f"no workbook at {bench.WORKBOOK}")
        by_video, _, _ = bench.load_annotations()
        spans = [end - start for v in by_video.values() for start, end, _ in v]
        median, p25, p75, top = bench.percentiles(spans)
        self.assertEqual((440.0, 325.0, 600.0, 2880.0), (median, p25, p75, top))


class RateTest(unittest.TestCase):
    def test_per_minute(self) -> None:
        self.assertAlmostEqual(60.0, bench.rate_per_minute(60, 60000))
        self.assertEqual(0.0, bench.rate_per_minute(5, 0))


class FloorTest(unittest.TestCase):
    """The shortest emittable window, which is never MIN_SIGN_MS itself."""

    def test_twenty_five_fps(self) -> None:
        # 40 ms frames: first span past 850 is 880, plus one more frame of deceleration.
        self.assertAlmostEqual(920.0, bench.floor_ms(25.0))

    def test_twenty_fps(self) -> None:
        # 50 ms frames: 850 lands exactly on the grid, and one 50 ms hold satisfies the wait.
        self.assertAlmostEqual(850.0, bench.floor_ms(20.0))

    def test_floor_is_never_below_min_sign_ms(self) -> None:
        for fps in (15.0, 20.0, 24.0, 25.0, 30.0, 50.0, 60.0):
            self.assertGreaterEqual(bench.floor_ms(fps), sim.MIN_SIGN_MS, f"{fps} fps")

    def test_a_real_segmenter_run_never_undercuts_the_floor(self) -> None:
        """The floor must bound actual output, or the diagnostic in block 3 lies.

        Every window except the last: the end-of-recording flush in `sim.windows` closes on
        MIN_MS alone, so a final short window is legitimate and is counted separately.
        """
        fps = 25.0
        frames = 600
        right = np.zeros((frames, 21, 3), dtype=np.float32)
        right[:, :, 0] = np.linspace(0.4, 0.6, 21)
        right[:, 5, :2], right[:, 17, :2] = (0.40, 0.5), (0.50, 0.5)
        step = np.sin(np.arange(frames) / 2.0) * 0.08
        for tip in (4, 8, 12, 16, 20):
            right[:, tip, 0] += step
        spans = bench.window_spans(right, np.zeros_like(right), fps)
        self.assertGreater(len(spans), 1, "constant motion should emit several windows")
        lowest = bench.floor_ms(fps)
        for start, end in spans[:-1]:
            self.assertGreaterEqual(end - start, lowest - 1.0)
        self.assertGreaterEqual(spans[-1][1] - spans[-1][0], sim.MIN_MS)

    def test_the_floor_is_the_mode_of_a_constant_motion_run(self) -> None:
        """Steady motion has no real sign boundaries, so the floor is what the rules pick."""
        fps = 25.0
        right = np.zeros((600, 21, 3), dtype=np.float32)
        right[:, :, 0] = np.linspace(0.4, 0.6, 21)
        right[:, 5, :2], right[:, 17, :2] = (0.40, 0.5), (0.50, 0.5)
        step = np.sin(np.arange(600) / 2.0) * 0.08
        for tip in (4, 8, 12, 16, 20):
            right[:, tip, 0] += step
        spans = bench.window_spans(right, np.zeros_like(right), fps)
        lengths = [end - start for start, end in spans]
        self.assertEqual(bench.floor_ms(fps), min(lengths[:-1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
