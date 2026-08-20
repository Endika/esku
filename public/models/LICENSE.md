# Licence of the shipped models

**The application's source code is MIT (see [`LICENSE`](../../LICENSE)). The trained vocabulary
model in this directory is not.**

## `lse-vocabulary.bin` and `lse-vocabulary.json` — CC BY-NC 4.0

The vocabulary model is trained on two corpora, and one of them is non-commercial:

| corpus | licence | DOI |
| --- | --- | --- |
| **SWL-LSE** | CC BY 4.0 | [`10.5281/zenodo.13691887`][swl] |
| **LSE-Health-UVigo** | **CC BY-NC 4.0** | [`10.5281/zenodo.10234465`][health] |

[swl]: https://doi.org/10.5281/zenodo.13691887
[health]: https://doi.org/10.5281/zenodo.10234465

So these weights are distributed under **Creative Commons Attribution-NonCommercial 4.0
International**, with attribution as recorded in [`NOTICE.md`](../../NOTICE.md).

Using Esku is unaffected: the app is free and stays free. What this does mean is that **a
commercial fork cannot ship this model**. Retraining on SWL-LSE alone produces an MIT-clean model,
at the cost of the accuracy that LSE-Health's co-articulated annotations buy.

This is the cautious reading. UVigo's reply pointing us at the corpus placed LSE-Health under
CC BY 4.0 while its Zenodo record declares CC BY-NC 4.0; until that is settled we assume the
stricter of the two. If it is confirmed as CC BY, this file goes away.

## `hand_landmarker.task`, `pose_landmarker_lite.task`, `face_landmarker.task` — Apache 2.0

Google's MediaPipe Tasks Vision bundles, vendored so they are served same-origin.
<https://github.com/google-ai-edge/mediapipe>

## What is not here

No video and no landmark file from any corpus is redistributed — only weights derived from them.
That matters twice over for LSE-Health, whose recordings are of identifiable Deaf signers, and
whose sibling corpus LSE_Lex40 was never published because the university's data protection
assessment did not clear it.
