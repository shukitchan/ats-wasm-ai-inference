AI Inference Examples for ATS Wasm Plugin
====
  - These examples are tested to work with wasmtime as the runtime for the ATS Wasm Plugin
  - We also need the ATS wasm plugin with the fix here - https://github.com/proxy-wasm/proxy-wasm-cpp-host/pull/433
  - These examples are also tested to work with Rust 1.88
  - To compile, run `cargo build --target=wasm32-wasip1 --release` inside each of the example directory to generate the wasm modules in `target/wasm32-wasip1/release/myfilter.wasm`. Copy it to `/usr/local/var/wasm/`
  - Copy `myfilter.yaml` inside the example directory as well to `/usr/local/var/wasm/`
  - Make sure we have ATS wasm plugin added to ATS
```
wasm.so /usr/local/var/wasm/myfilter.yaml
```
  - We can have a very simple `remap.config`
```
map http://test.com/ http://httpbin.org/
```  
  - Turn on debug log for `wasm` label and restart ATS. Proxy a HTTP request through the ATS and see the inference results in the debug log (`traffic.out`).

Imagenet Example
====
  - ImageNet challenge: The goal is to categorize images and associate them with one of [1000 labels](https://github.com/anishathalye/imagenet-simple-labels/blob/master/imagenet-simple-labels.json). In other words, recognize a dog, a cat, a rabbit, or a military uniform.
  - MobileNet model is a response to the challenge. The model is copied from Fastly AI example
  - The [basketball image](https://unsplash.com/photos/spalding-basketball-in-court-Gl0jBJJTDWs) is from unsplash
  - Inspired by Fastly AI example and Sonos tract MobileNet example
  - The program needs an image to be posed as part of the request. e.g.
```
curl -s -v -H "Expect:" -H 'Host: test.com' --data-binary '@./images/bball.jpg' -H 'content-type: application/x-www-form-urlencoded' 'http://localhost:8080/anything' > /dev/null
```

MNIST example
====
  - MNIST: The goal is to identify hand-written digits
  - The model comes from https://github.com/onnx/models/
  - The image is from https://github.com/teavanist/MNIST-JPG/
  - The program needs an image to be posted as part of the request. e.g.
```
curl -s -v -H 'Host: test.com' --data-binary '@./images/7.jpg' -H 'content-type: application/x-www-form-urlencoded'  'http://localhost:8080/anything' > /dev/null
```

Sentiment analysis example
====
  - Model and `tokenizer.json` come from huggingface.co
    - https://huggingface.co/tabularisai/robust-sentiment-analysis
    - we want the format to be ONNX and optmized - https://huggingface.co/shukitchan2023/robust-sentiment-analysis-ONNX
    - This can be done through this space - https://huggingface.co/spaces/onnx-community/convert-to-onnx
  - The full model is too large to be included in the rust program. We opt to use the quantized model instead and so we can also skip calling `into_optimized()` on the loaded model.
  - The program needs an `input` request header as the input for sentiment analysis. Otherwise an error response will be printed instead. e.g.
```
curl -v -H 'Host: test.com' -H 'input: "I love Rust and AI!"' 'http://localhost:8080/'
```    

Notes on Rust 1.82 and WAMR
====
  - In Rust 1.82, webassembly target support for `reference-types` is on by default. See [changes](https://releases.rs/docs/1.82.0/) in 1.82
  - WAMR does not support `reference-types` by default till 2.3.0
  - Thus we need to wait for `proxy-wasm` to [support WAMR-2.3.0](https://github.com/proxy-wasm/proxy-wasm-cpp-host/issues/449) if we want to use WAMR as runtime
  - For now, if we want to use WAMR, stay with rust 1.81. But we need to downgrade some libraries as well for the compilation
```
cargo update liquid@0.26.11 --precise 0.26.9
cargo update liquid-lib@0.26.11 --precise 0.26.9
cargo update liquid-core@0.26.11 --precise 0.26.9
cargo update liquid-derive@0.26.10 --precise 0.26.8
```

  - or compile the examples with `wasm32-unknown-unknown` as target with the feature turned off

```
RUSTFLAGS="-C target-cpu=mvp" cargo +nightly build -Z build-std=std,panic_abort --target=wasm32-unknown-unknown --release --verbose
``` 

Links
====
  - [ATS](https://trafficserver.apache.org)
  - [ATS Wasm Plugin](https://docs.trafficserver.apache.org/en/latest/admin-guide/plugins/wasm.en.html)
  - [Sonos tract](https://github.com/sonos/tract)
  - [Fastly AI example](https://www.fastly.com/documentation/solutions/demos/edgeml/)
