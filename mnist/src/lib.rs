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
        Some(Box::new(HttpHeaders { context_id }))
    }

}

struct HttpHeaders {
    context_id: u32,
}

impl Context for HttpHeaders {}

impl HttpContext for HttpHeaders {
    fn on_http_request_headers(&mut self, _: usize, _: bool) -> Action {

    trace!("context {}", self.context_id);

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

    let image_bytes = include_bytes!("../images/7.jpg");
    let image_bytes_len = image_bytes.len();
    trace!("image length {}", image_bytes_len);
    let img = image::load_from_memory(image_bytes).expect("error in loading image").to_luma8();
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

    Action::Continue

    }
}
