// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use log::trace;
use proxy_wasm::traits::*;
use proxy_wasm::types::*;
use primes::is_prime;
use std::io::Cursor;
use tract_flavour::prelude::*;

proxy_wasm::main! {{
    proxy_wasm::set_log_level(LogLevel::Trace);
    proxy_wasm::set_root_context(|_| -> Box<dyn RootContext> { Box::new(HelloWorld) });
}}

struct HelloWorld;

impl Context for HelloWorld {}

impl RootContext for HelloWorld {
    fn on_vm_start(&mut self, _: usize) -> bool {
        trace!("Hello, World!");
        true
    }

    fn get_type(&self) -> Option<ContextType> {
        Some(ContextType::HttpContext)
    }

    fn create_http_context(&self, context_id: u32) -> Option<Box<dyn HttpContext>> {
        Some(Box::new(HttpHeaders { context_id }))
    }

}

struct HttpHeaders {
    context_id: u32,
}

impl Context for HttpHeaders {}

impl HttpContext for HttpHeaders {
    fn on_http_request_headers(&mut self, _: usize, _: bool) -> Action {

    let model_bytes = include_bytes!("../models/mobilenet_v2_1.4_224_frozen.pb");
    let model_bytes_len = model_bytes.len();
    trace!("model length {}", model_bytes_len);
    let model = tract_flavour::tensorflow() // swap in ::nnef() for the tract-nnef package, etc.
        // Load the model.
        .model_for_read(&mut Cursor::new(model_bytes)).expect("Shit 1!")
        // Specify input type and shape.
        .with_input_fact(0,InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 224, 224, 3))).expect("Shit 2!")
        // Optimize the model.
        .into_optimized().expect("Shit 3!")
        // Make the model runnable and fix its inputs and outputs.
        .into_runnable().expect("Shit 4!");

    //let image_bytes = include_bytes!("../images/charlotte-v2.jpg");
    let image_bytes = include_bytes!("../images/bball.jpg");
    let image_bytes_len = image_bytes.len();
    trace!("image length {}", image_bytes_len);
    let img = image::load_from_memory(image_bytes).expect("Shit 5!").to_rgb8();
    let resized = image::imageops::resize(&img, 224, 224, image::imageops::FilterType::Nearest);
    let img: Tensor = tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into();

    let result = model.run(tvec!(img)).expect("Shit 6!");

    let best = result[0]
        .to_array_view::<f32>().expect("Shit 7!")
        .iter()
        .cloned()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    trace!("result: {best:?}");
    

        for (name, value) in &self.get_http_request_headers() {
            let s3 = format!("In WASM: #{} -> {}: {}",self.context_id, name, value);
            trace!("{}", s3);
        }
        if let Some(ua) = self.get_http_request_header("User-Agent") {
            if ua != "" {
              trace!("UA is {}", ua);
            }
        }

        match self.get_http_request_header("token") {
            Some(token) if token.parse::<u64>().is_ok() && is_prime(token.parse().unwrap()) => {
                trace!("It is prime!!!");
                Action::Continue
            }
            _ => {
                trace!("It is not prime!!! That's true.");
                self.send_http_response(
                    403,
                    vec![("Powered-By", "proxy-wasm")],
                    Some(b"Access forbidden.\n"),
                );
                Action::Pause
            }
        }
    }

}
