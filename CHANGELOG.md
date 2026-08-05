# Changelog

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
