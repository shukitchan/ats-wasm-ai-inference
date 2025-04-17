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

    Action::Continue

    }
}
