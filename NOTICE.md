# Notices and attributions

Esku's own source code is MIT (see `LICENSE`). The recognition models are trained on
third-party data with its own terms, listed here — and since 1.14 **the vocabulary model is no
longer MIT**: see `public/models/LICENSE.md`.

## SWL-LSE — SignaMed Word-Level LSE

The isolated-sign vocabulary model is trained on the **SWL-LSE** dataset: 8,000 sign
sequences over 300 Spanish Sign Language health-domain sign classes, performed by 124
signers (deaf signers, interpreters and L2 students).

- Dataset: <https://zenodo.org/records/13691887> (DOI `10.5281/zenodo.13691887`)
- Licence: **Creative Commons Attribution 4.0 International (CC-BY-4.0)**
- Paper: Docío-Fernández et al., *SWL-LSE: A Dataset of Health-Related Signs in Spanish
  Sign Language with an ISLR Baseline Method*, Technologies 12(10), 2024.
  <https://doi.org/10.3390/technologies12100205>

CC-BY-4.0 permits redistribution and derivative works, including the trained weights
shipped in `public/models/`, provided attribution is given. This file is that
attribution, and it is also surfaced in the app's "Acerca de" screen.

No dataset video or landmark file is redistributed in this repository — only weights
derived from it.

## LSE-Health-UVigo

The vocabulary model is also trained on **LSE-Health-UVigo**: 273 videos, 10.8 hours of
continuous health-domain discourse from ten signers (seven Deaf, three interpreters), with
15,098 hand-annotated gloss occurrences carrying start and end times. Those annotations are
what let the model see co-articulated signing at all — trained on SWL-LSE alone it wrote the
right word for 0.8% of signs in continuous use.

- Dataset: <https://zenodo.org/records/10234465> (DOI `10.5281/zenodo.10234465`)
- Licence: **Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC-4.0)**,
  as declared on Zenodo. The group's own reply placed it under CC-BY-4.0; until that is
  settled we assume the stricter reading, which is why `public/models/LICENSE.md` exists.
- Authors: Alba-Castro, Jose L.; Vázquez-Enríquez, Manuel; Pérez-Pérez, Ania; Mariño-Pérez,
  Flora; Lema-Álvarez, Manuel L.; Cabeza-Pereiro, Carmen; Rodríguez-Banga, Eduardo;
  Docío-Fernández, Laura; Torres-Guijarro, Soledad; Caderno-Fernández, Alba; Cid-Álvarez, Sol
- Paper: Vázquez Enríquez, M.; Alba Castro, J. L.; Docío Fernández, L.; Jacques Junior, J. C. S.;
  Escalera, S. *ECCV 2022 Sign Spotting Challenge: Dataset, Design and Results*. Lecture Notes
  in Computer Science 13808.
- The videos originate from the Servizo Galego de Saúde YouTube channel under a permissive
  licence; the corpus's own value is the annotation.

**Because CC-BY-NC forbids commercial use, the weights cannot be MIT.** The code still is. A
commercial fork has to retrain on SWL-LSE alone.

As with SWL-LSE, no video and no landmark file is redistributed here. That matters more for this
corpus: the recordings are of identifiable Deaf signers, and its sibling corpus LSE_Lex40 was
never published because the university's data protection assessment did not clear it.

## LSE-FS-UVigo (fingerspelling weights)

`public/models/lse-alphabet.*` is trained on this corpus and **only** on this corpus, which is
why it is CC BY 4.0 and the vocabulary model is not. See
[`public/models/LICENSE.md`](public/models/LICENSE.md).

- Source: <https://zenodo.org/records/15797079>, DOI `10.5281/zenodo.15797079`
- Licence: **CC BY 4.0** — attribution, no commercial restriction
- Authors: Ruanova Lea, Jose Luis; Alba-Castro, Jose L.; Docio-Fernandez, Laura;
  Pérez Pérez, Ania; Longa Alonso, Beatriz
- Paper: DOI `10.1109/ACCESS.2025.3631400`
- Reference code: <https://github.com/lruanova/LSE-FS-UVigo-train>

3,044 sequences of continuous fingerspelling, labelled by the spelled word with no per-letter
timings — which is why the model is trained with CTC. As with the other corpora, no video and no
landmark file is redistributed here; only derived weights.

## MediaPipe Tasks (hand and pose landmarks)

Landmark extraction uses Google's MediaPipe Tasks Vision bundles, vendored into
`public/models/` and served same-origin.

- Licence: **Apache License 2.0**
- Source: <https://github.com/google-ai-edge/mediapipe>

## ONNX Runtime Web

- Licence: **MIT**
- Source: <https://github.com/microsoft/onnxruntime>

## Not used

**sign.mt** (<https://sign.mt>) is the closest prior art and was studied as a reference,
but it is licensed **CC BY-NC-SA 4.0** — non-commercial and share-alike. No sign.mt code,
weights or assets are vendored here, so Esku stays MIT-licensable.

**LSE_Lex40_UVIGO** is not used, and cannot be: UVigo never published it, because their data
protection officer did not find sufficient guarantees under GDPR. The group pointed us at
LSE-Health-UVigo, LSE-FS-UVigo and LSE-METI-UVigo instead; the first two are now both in use.

**LSE-METI-UVigo** (<https://zenodo.org/records/20035734>) is not used yet. ~5,000 continuous
sentences with gloss sequences and no temporal alignment. That was recorded here as needing
"CTC-style training rather than this architecture" — and the fingerspelling head above is now
exactly that, so the blocker is no longer architectural. It remains CC-BY-NC-4.0, so weights
derived from it would inherit the vocabulary model's commercial restriction.
