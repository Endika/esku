# Changelog

## [1.10.2](https://github.com/Endika/esku/compare/v1.10.1...v1.10.2) (2026-08-07)


### Bug Fixes

* measure sign windows in time, so recognition survives any frame rate ([f4ff381](https://github.com/Endika/esku/commit/f4ff38171a4bce34a631a02e60c4abf55ad25105))

## [1.10.1](https://github.com/Endika/esku/compare/v1.10.0...v1.10.1) (2026-08-06)


### Bug Fixes

* stop cancelling Pages deploys, which blocks the deployment queue ([#51](https://github.com/Endika/esku/issues/51)) ([917438a](https://github.com/Endika/esku/commit/917438add3a7ba82fc20f7782c26c881ecfa35ff))

## [1.10.0](https://github.com/Endika/esku/compare/v1.9.3...v1.10.0) (2026-08-06)


### Features

* give the camera its own layout so no control needs a scroll ([#49](https://github.com/Endika/esku/issues/49)) ([488f1bf](https://github.com/Endika/esku/commit/488f1bfe51d2eb881fd15ad066c629c5c23b7e84))

## [1.9.3](https://github.com/Endika/esku/compare/v1.9.2...v1.9.3) (2026-08-06)


### Bug Fixes

* lower the window confidence floor to the value that was actually measured ([#45](https://github.com/Endika/esku/issues/45)) ([87d1a07](https://github.com/Endika/esku/commit/87d1a07e798a914df4ef0788d41224dfdac24781))

## [1.9.2](https://github.com/Endika/esku/compare/v1.9.1...v1.9.2) (2026-08-06)


### Bug Fixes

* close sign windows on deceleration instead of waiting for stillness ([#43](https://github.com/Endika/esku/issues/43)) ([26e4f86](https://github.com/Endika/esku/commit/26e4f86f69dcf5600091f73876a62d9fc2f4c519))

## [1.9.1](https://github.com/Endika/esku/compare/v1.9.0...v1.9.1) (2026-08-06)


### Bug Fixes

* compare the feature vector against medians, not poisoned means ([#41](https://github.com/Endika/esku/issues/41)) ([8b66f6d](https://github.com/Endika/esku/commit/8b66f6dade286ccb1bced3e5170a71f6f6d03568))

## [1.9.0](https://github.com/Endika/esku/compare/v1.8.0...v1.9.0) (2026-08-06)


### Features

* report the feature vector the model was actually fed ([#39](https://github.com/Endika/esku/issues/39)) ([024723d](https://github.com/Endika/esku/commit/024723d0030ee08c2a7c5dcbdb38fb100c55fe0a))

## [1.8.0](https://github.com/Endika/esku/compare/v1.7.6...v1.8.0) (2026-08-06)


### Features

* show what the pipeline did when no word appears ([#37](https://github.com/Endika/esku/issues/37)) ([44a55fc](https://github.com/Endika/esku/commit/44a55fcd2b024ba38ac36dc61b09cc6ea4445905))

## [1.7.6](https://github.com/Endika/esku/compare/v1.7.5...v1.7.6) (2026-08-05)


### Bug Fixes

* stop discarding finished signs while the classifier is busy ([e35c899](https://github.com/Endika/esku/commit/e35c899cd833f676e89b5bb8c1ef9bd5a5e0c25c))

## [1.7.5](https://github.com/Endika/esku/compare/v1.7.4...v1.7.5) (2026-08-05)


### Bug Fixes

* report disk size instead of gzip transfer size ([ec14438](https://github.com/Endika/esku/commit/ec1443878bb248c66ed1480b1ff737fa48f1b94c))

## [1.7.4](https://github.com/Endika/esku/compare/v1.7.3...v1.7.4) (2026-08-05)


### Bug Fixes

* cache the wasm runtime so offline actually works ([0daeeb9](https://github.com/Endika/esku/commit/0daeeb938833c977e1b6e400a207fb54107e4035))

## [1.7.3](https://github.com/Endika/esku/compare/v1.7.2...v1.7.3) (2026-08-05)


### Bug Fixes

* tune segmentation to match training and correct the engine size ([57a0949](https://github.com/Endika/esku/commit/57a09495f3765b53e4f96676d68560615bc074dc))

## [1.7.2](https://github.com/Endika/esku/compare/v1.7.1...v1.7.2) (2026-08-05)


### Bug Fixes

* stop drawing pose landmarks the model never saw ([27cfdb2](https://github.com/Endika/esku/commit/27cfdb29a71baa3e3f81adcfd55af98c016ee874))

## [1.7.1](https://github.com/Endika/esku/compare/v1.7.0...v1.7.1) (2026-08-05)


### Bug Fixes

* recognise signing that never pauses ([1591ebe](https://github.com/Endika/esku/commit/1591ebe3d1fcb16d05038d0e758e813168ff8408))

## [1.7.0](https://github.com/Endika/esku/compare/v1.6.0...v1.7.0) (2026-08-05)


### Features

* read torso, head and face to recognise signs better ([99d79c7](https://github.com/Endika/esku/commit/99d79c766632073e9f34c585c594beed7791dfe0))

## [1.6.0](https://github.com/Endika/esku/compare/v1.5.0...v1.6.0) (2026-08-05)


### Features

* show face, neck, torso and arms tracking on camera ([a700811](https://github.com/Endika/esku/commit/a700811bc04383cc609456ee867563d6f8ee26a1))

## [1.5.0](https://github.com/Endika/esku/compare/v1.4.0...v1.5.0) (2026-08-05)


### Features

* read signs from the rear camera too ([3e974f4](https://github.com/Endika/esku/commit/3e974f447ee50541f243eac515f407c92d696149))

## [1.4.0](https://github.com/Endika/esku/compare/v1.3.0...v1.4.0) (2026-08-05)


### Features

* recognise 238 LSE signs with a trained model ([ab4e20a](https://github.com/Endika/esku/commit/ab4e20ac151ba50bea73b6d05d3765e485d989f0))

## [1.3.0](https://github.com/Endika/esku/compare/v1.2.0...v1.3.0) (2026-08-05)


### Features

* let the user download and free the cached engine ([5f58558](https://github.com/Endika/esku/commit/5f5855894dc25e49b452f652b6856403f5ee9e38))

## [1.2.0](https://github.com/Endika/esku/compare/v1.1.0...v1.2.0) (2026-08-05)


### Features

* teach the app your own signs and recognise them offline ([2fb4fb7](https://github.com/Endika/esku/commit/2fb4fb73c7eb24587250661812d0369bf686be48))

## [1.1.0](https://github.com/Endika/esku/compare/v1.0.0...v1.1.0) (2026-08-05)


### Features

* draw the tracked hand skeleton over the camera ([2f4dfc8](https://github.com/Endika/esku/commit/2f4dfc84fbccbd4abc28ce522a789e8b4f2b4137))

## 1.0.0 (2026-08-05)


### Features

* read fingerspelled LSE letters from the camera ([0d07c31](https://github.com/Endika/esku/commit/0d07c31b46ce1b31f1672b506764c8629bbb95a0))
* scaffold Esku sign-language-to-text PWA with hexagonal domain ([8d108c5](https://github.com/Endika/esku/commit/8d108c591fde954ec75f59e2902c2944141d21ca))


### Bug Fixes

* **deps:** override sharp to clear inherited libvips CVEs ([9d1cf2f](https://github.com/Endika/esku/commit/9d1cf2f02e1b58482bc31795aee0b0f6f8672c24))
