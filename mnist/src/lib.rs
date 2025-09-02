use log::trace;
use proxy_wasm::traits::*;
use proxy_wasm::types::*;
use std::io::Cursor;
use tract_flavour::prelude::tract_ndarray;
use tract_flavour::prelude::Framework;
use tract_flavour::prelude::Datum;
use tract_flavour::prelude::Tensor;
use tract_flavour::prelude::InferenceModelExt;
use tract_flavour::prelude::InferenceFact;
use tract_flavour::prelude::tvec;

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
    trace!("self.predicted_index: {:?}", self.predicted_index);
    trace!("self.predicted_confidence: {:?}", self.predicted_confidence);

        for (name, value) in &self.get_http_response_headers() {
            trace!("#{} <- {}: {}", self.context_id, name, value);
        }

        if let Some(index) = self.predicted_index {
            trace!("index added");
            let index_str = index.to_string();
            self.add_http_response_header("x-predicted-digit", index_str.as_str());
        }
        if let Some(confidence) = self.predicted_confidence {
            trace!("confidence added");
            let confidence_str = confidence.to_string();
            self.add_http_response_header("x-predicted-confidence", confidence_str.as_str());
        }

        Action::Continue
    }

    fn on_http_request_headers(&mut self, _: usize, _: bool) -> Action {
        trace!("context {}", self.context_id);
        Action::Continue
    }

    fn on_http_request_body(&mut self, body_size: usize, end_of_stream: bool) -> Action {
        if !end_of_stream {
            // Wait -- we'll be called again when the complete body is buffered
            // at the host side.
            return Action::Pause;
        }

        // Replace the message body if it contains the text "secret".
        // Since we returned "Pause" previuously, this will return the whole body.
        if let Some(body_bytes) = self.get_http_request_body(0, body_size) {
            trace!("Request body size: {}", body_bytes.len());
            trace!("Request body bytes: {:?}", body_bytes);

            let image_bytes = &body_bytes;
            let img = image::load_from_memory(image_bytes).expect("error in loading image").to_luma8();

            trace!("Image width: {}", img.width());
            trace!("Image height: {}", img.height());

            let model_bytes = include_bytes!("../models/mnist-8.onnx");
            let model_bytes_len = model_bytes.len();
            trace!("model length {}", model_bytes_len);
            let model = tract_flavour::onnx()
                .model_for_read(&mut Cursor::new(model_bytes)).expect("error in loading model")
                .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 1, 28, 28))).expect("error specifying input type and shape")
                // Optimize the model.
                .into_optimized().expect("error in optimizing")
                // Make the model runnable and fix its inputs and outputs.
                .into_runnable().expect("error in making modelrunnable");

            let resized = image::imageops::resize(&img, 28, 28, image::imageops::FilterType::Nearest);
            let tensor = tract_ndarray::Array4::from_shape_vec((1, 1, 28, 28), resized.into_raw())
                .unwrap()
                .mapv(|x| x as f32 / 255.0);
            let input_tensor: Tensor = tensor.into();

            let result = model.run(tvec!(input_tensor.into())).expect("error in generating results");

            let output = result[0]
                .to_array_view::<f32>().expect("error in checking the result")
                .iter()
                .cloned()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            trace!("result: {:?}", output);

            if let Some((index, value)) = output {
                self.predicted_index = Some(index);
                self.predicted_confidence = Some(value);
                trace!("Predicted digit: {}, confidence: {}", index, value);
            } else {
                trace!("No prediction could be made.");
            }

            trace!("self.predicted_index: {:?}", self.predicted_index);
            trace!("self.predicted_confidence: {:?}", self.predicted_confidence);
        }
        Action::Continue
    }
}
