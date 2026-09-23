# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Four audiences, all confirmed, all on a phone almost every time (~99% of use):

- **A hearing person reading someone else**, often a health worker, holding the phone with the
  rear camera pointed at a Deaf person who signs.
- **A Deaf person signing**, with the front camera, turning their own signs into text to show to
  someone who does not sign.
- **Learners and the curious**, practising LSE or testing what the app recognises.
- **Researchers and demos**: the UVigo group, the Deaf community and anyone evaluating the model.

The camera is open while the app is in use, often in a waiting room or a clinic, with the phone
held up in front of another person.

## Product Purpose

Esku recognises Spanish Sign Language (LSE) signs and fingerspelled letters from the phone
camera, **one at a time**, and appends them to a running, editable text. Success is a sequence
of recognised signs that is useful to the people in the conversation, with its errors visible
and fixable, not a translated sentence.

## Positioning

Honest, on-device sign recognition. Everything runs in the browser: no frame, landmark or
transcript leaves the device, because there is no backend. It states its accuracy and its
limits in the interface itself, in figures.

## Operating Context

- Used live, camera on, at arm's length, often while another person watches the screen.
- Two cameras with different jobs: front (the signer is the user), rear (the user reads the
  person in front of them). Only the front view is mirrored.
- Installable PWA that works offline once the models are cached (~30 MB stored).
- The transcript is editable (undo, clear) because the model is fallible.

## Capabilities and Constraints

- Three engines behind one port: **Alphabet** (27 fingerspelled letters, seven unreliable),
  **Vocabulary** (286 health-domain LSE signs plus a "nobody is signing" class), and **Taught**
  (any sign the user records 3+ times, in any sign language, stored only on the device).
- Live feedback while running: which body parts are tracked (hands, face, neck, torso, arms),
  the top live guess over the video, and a status line explaining why nothing is being written.
- Secondary panels: teach a sign, on-device storage (download and clear the models), diagnostics,
  and what it recognises.
- **Diagnostics is for users too**, not only for debugging: it explains why a word does not
  appear, so it must stay reachable while the camera runs.
- Accuracy to state plainly: about 2 in 3 signs right one by one, far less signing continuously.
- Copy is Spanish (`lang="es"`).
- Vanilla TypeScript and CSS with Vite, deployed to GitHub Pages; no UI framework.

## Brand Commitments

- Name **Esku** and the raised-hand mark (`public/favicon.svg`).
- **Never describe Esku as translating LSE.** No "traductor", no implied sentence-level
  translation. It recognises signs one at a time. The Deaf community is tired of being sold
  hype, and the research group Esku works with is deliberately cautious about it.
- Voice: plain, precise, modest. Limits are stated, never softened into marketing.

## Evidence on Hand

- Measured accuracy figures and their scope in `README.md` ("How good is the vocabulary model,
  really").
- Datasets and licences: SWL-LSE (CC-BY-4.0) and LSE-Health-UVigo (CC-BY-NC-4.0).
- Privacy statement in `PRIVACY.md`.
- No testimonials, users, partners or endorsements exist; none may be invented.

## Product Principles

1. **Honest before impressive.** Every claim of capability carries its limit next to it.
2. **The camera view is the tool.** While running, everything needed to read signs fits one
   screen; nothing essential is a scroll away.
3. **Say why nothing happened.** No hand, unknown shape, or reading are different states and the
   user must be able to tell which one they are in.
4. **Nothing leaves the device**, and the interface says so where it matters, at the camera.

## Accessibility & Inclusion

- Deaf users: every piece of feedback is visual; nothing may depend on sound.
- Read at arm's length and by a second person: the live guess and the transcript must be large
  and high-contrast.
- Touch targets of at least 44 px; one-handed use on a phone.
- Both light and dark schemes; respect reduced motion.
