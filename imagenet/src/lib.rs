use log::trace;
use proxy_wasm::traits::*;
use proxy_wasm::types::*;
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
        Some(Box::new(HttpHeaders { context_id, predicted_index: None, predicted_confidence: None }))
    }

}

struct HttpHeaders {
    context_id: u32,
    predicted_index: Option<usize>,
    predicted_confidence: Option<f32>,
}

impl Context for HttpHeaders {}

impl HttpContext for HttpHeaders {
    fn on_http_response_headers(&mut self, _: usize, _: bool) -> Action {
        for (name, value) in &self.get_http_response_headers() {
            trace!("#{} <- {}: {}", self.context_id, name, value);
        }

        if let Some(index) = self.predicted_index {
            let index_str = index.to_string();
            self.add_http_response_header("x-predicted-label", index_str.as_str());
        }

        if let Some(confidence) = self.predicted_confidence {
            let confidence_str = confidence.to_string();
            self.add_http_response_header("x-predicted-confidence", confidence_str.as_str());
        }

        Action::Continue
    }

    fn on_http_request_headers(&mut self, _: usize, _: bool) -> Action {
        trace!("context {}", self.context_id);

/*
    let model_bytes = include_bytes!("../models/mobilenet_v2_1.4_224_frozen.pb");
    let model_bytes_len = model_bytes.len();
    trace!("model length {}", model_bytes_len);
    let model = tract_flavour::tensorflow() // swap in ::nnef() for the tract-nnef package, etc.
        // Load the model.
        .model_for_read(&mut Cursor::new(model_bytes)).expect("error in loading model")
        // Specify input type and shape.
        .with_input_fact(0,InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 224, 224, 3))).expect("error specifying input type and shape")
        // Optimize the model.
        .into_optimized().expect("error in optimizing")
        // Make the model runnable and fix its inputs and outputs.
        .into_runnable().expect("error in making modelrunnable");

    let image_bytes = include_bytes!("../images/bball.jpg");
    let image_bytes_len = image_bytes.len();
    trace!("image length {}", image_bytes_len);
    let img = image::load_from_memory(image_bytes).expect("error in loading image").to_rgb8();
    let resized = image::imageops::resize(&img, 224, 224, image::imageops::FilterType::Nearest);
    let img: Tensor = tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into();

    let result = model.run(tvec!(img)).expect("error in generating results");

    let best = result[0]
        .to_array_view::<f32>().expect("error in checking the result")
        .iter()
        .cloned()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    trace!("result: {best:?}");
*/
        Action::Continue
    }

    fn on_http_request_body(&mut self, body_size: usize, end_of_stream: bool) -> Action {
        if !end_of_stream {
            // Wait -- we'll be called again when the complete body is buffered
            return Action::Pause;
        }

        if let Some(body_bytes) = self.get_http_request_body(0, body_size) {

            let image_bytes = &body_bytes;
            let img = image::load_from_memory(image_bytes).expect("error in loading image").to_rgb8();

            let model_bytes = include_bytes!("../models/mobilenet_v2_1.4_224_frozen.pb");
            let model_bytes_len = model_bytes.len();
            trace!("model length {}", model_bytes_len);
            let model = tract_flavour::tensorflow() // swap in ::nnef() for the tract-nnef package, etc.
                // Load the model.
                .model_for_read(&mut Cursor::new(model_bytes)).expect("error in loading model")
                // Specify input type and shape.
                .with_input_fact(0,InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 224, 224, 3))).expect("error specifying input type and shape")
                // Optimize the model.
                .into_optimized().expect("error in optimizing")
                // Make the model runnable and fix its inputs and outputs.
                .into_runnable().expect("error in making modelrunnable");

            let resized = image::imageops::resize(&img, 224, 224, image::imageops::FilterType::Nearest);
            let img: Tensor = tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
                resized[(x as _, y as _)][c] as f32 / 255.0
            })
            .into();

            let result = model.run(tvec!(img)).expect("error in generating results");

            let best = result[0]
                .to_array_view::<f32>().expect("error in checking the result")
                .iter()
                .cloned()
                .zip(1..)
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            trace!("result: {best:?}");

            if let Some((value, label)) = best {
                self.predicted_index = Some(label);
                self.predicted_confidence = Some(value);
                trace!("Predicted digit: {}, confidence: {}", label, value);
            } else {
                trace!("No prediction could be made.");
            }

        }
        Action::Continue
    }
}
