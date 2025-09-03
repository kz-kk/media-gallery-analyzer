This folder contains vendored client-side dependencies to avoid any CDN usage:

- model-viewer.min.js: @google/model-viewer (module build)
- meshopt_decoder.module.js / .wasm: Meshopt decoder used by GLTF
- draco_wasm_wrapper.js / draco_decoder.wasm: Draco GLTF decoder

All are served locally under /assets/js/vendor so CSP can exclude external CDNs.

