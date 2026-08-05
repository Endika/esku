# Security policy

## Reporting a vulnerability

Open a [private security advisory](https://github.com/Endika/esku/security/advisories/new).
Please do not file a public issue for anything exploitable.

Expect a first reply within 7 days.

## Threat model

Esku is a static, client-only PWA served from GitHub Pages. It has no backend, no
accounts, no API keys and no network calls at runtime beyond fetching its own assets.
That removes most of the usual surface, and concentrates what is left:

| Asset | Exposure | Control |
| --- | --- | --- |
| Camera frames | Highest-value input. Video may show a person's face, home and health situation. | Frames are read into a `<video>`/`WebGL` pipeline, converted to landmarks and discarded. No frame is stored, uploaded or logged. Nothing leaves the device. |
| Hand/pose landmarks | Derived biometric-ish data. | Held in memory for a sliding window only. Never persisted, never transmitted. |
| Taught custom signs | Stored locally, may include health vocabulary. | IndexedDB on the user's device, under the origin. Deletable from the UI. Not synced. |
| Transcript text | May contain health information the user signed. | In-memory; only persisted if the user explicitly saves. Never transmitted. |
| Model files (`.task`, `.onnx`, `.wasm`) | Supply chain — code and weights that execute locally. | Vendored into `public/` and served same-origin from Pages. No third-party CDN at runtime, so no external host can swap them. Updates go through a reviewed PR. |

### Why there is no CDN

Loading MediaPipe or ONNX Runtime from a CDN would mean a third party could serve
arbitrary WASM into a page that has camera permission. Every model and runtime is
self-hosted for that reason, not only for offline support.

### Privacy note

Because all inference is on-device and no personal data is transmitted, Esku acts as
neither a controller nor a processor of personal data in the GDPR sense for the
recognition pipeline — there is no processing outside the user's own browser. This is a
design constraint, not an implementation detail: **any change that sends frames,
landmarks or transcripts off-device is a breaking change to this policy** and must be
called out explicitly in a release note.

## Supported versions

Only the latest release, deployed at <https://endika.github.io/esku/>.
