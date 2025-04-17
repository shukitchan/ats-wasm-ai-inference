AI Inference Examples for ATS Wasm Plugin
====
  - We tested with WAMR as the runtime for the ATS Wasm Plugin
  - Currently we need to use rust 1.81 for the compilation. There are [changes](https://releases.rs/docs/1.82.0/) in 1.82 on the WebAssembly target generation that makes the module unable to be loaded in ATS with WAMR runtime
  - Because of the rust 1.81 requirement, we need to downgrade some libraries as well for the compilation
```
cargo update liquid@0.26.11 --precise 0.26.9
cargo update liquid-lib@0.26.11 --precise 0.26.9
cargo update liquid-core@0.26.11 --precise 0.26.9
cargo update liquid-derive@0.26.10 --precise 0.26.8
```
  - To compile, run `cargo build --target=wasm32-wasip1 --release` inside each of the example directory to generate the wasm modules in `target/wasm32-wasip1/release/myfilter.wasm`. Copy it to `/usr/local/var/wasm/`
  - Copy `myfilter.yaml` inside the example directory as well to `/usr/local/var/wasm/`
  - Make sure we have ATS wasm plugin added to ATS
```
wasm.so /usr/local/var/wasm/myfilter.yaml
```
  - Turn on debug log for `wasm` label and restart ATS. Proxy a HTTP request through the ATS and see the inference results in the debug log (`traffic.out`)

Imagenet Example
====
  - ImageNet challenge: The goal is to categorize images and associate them with one of 1000 labels. In other words, recognize a dog, a cat, a rabbit, or a military uniform.
  - MobileNet model is a response to the challenge
  - The basketball image is from unsplash
  - Inspired by Fastly AI example and Sonos tract MobileNet example

Links
====
  - [ATS](https://trafficserver.apache.org)
  - [ATS Wasm Plugin](https://docs.trafficserver.apache.org/en/latest/admin-guide/plugins/wasm.en.html)
  - [Sonos tract](https://github.com/sonos/tract)
  - [Fastly AI example](https://www.fastly.com/documentation/solutions/demos/edgeml/)
