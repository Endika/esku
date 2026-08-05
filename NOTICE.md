# Notices and attributions

Esku's own source code is MIT (see `LICENSE`). The recognition models are trained on
third-party data with its own terms, listed here.

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

**LSE_Lex40_UVIGO** is not used: it is distributed on request via the University of Vigo
GTM group rather than by open download, and publishes no redistribution licence.
