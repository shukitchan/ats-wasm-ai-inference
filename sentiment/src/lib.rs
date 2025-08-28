use log::trace;
use proxy_wasm::traits::*;
use proxy_wasm::types::*;
use std::io::Cursor;
use tract_onnx::prelude::*;
use tokenizers::Tokenizer;

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

        // get input
        let input_text: String = if let Some(input) = self.get_http_request_header("input") {
            if input != "" {
                input
            } else {
                self.send_http_response(
                    400,
                    vec![("Powered-By", "proxy-wasm")],
                    Some(b"Empty Input\r\n"),
                );
                return Action::Continue
                //"I really love Rust and AI!".to_string()
            }
        } else {
            self.send_http_response(
                400,
                vec![("Powered-By", "proxy-wasm")],
                Some(b"No Input\r\n"),
            );
            return Action::Continue
            //"I really really love Rust and AI!".to_string()
        };

        // Load ONNX model
        let model_bytes = include_bytes!("../models/bert_sentiment.onnx");
        let model = tract_onnx::onnx()
            .model_for_read(&mut Cursor::new(model_bytes)).expect("error loading model")
//            .into_optimized().expect("error optimizing model")
            .into_runnable().expect("error making model runnable");

        // Load tokenizer
        let tokenizer_bytes = include_bytes!("../models/tokenizer.json");
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes).expect("error loading tokenizer");

        //let input_text = "I love Rust and AI!".to_string();
        let encoding = tokenizer.encode(input_text.as_str(), true).unwrap();

        // Prepare input tensors
        let input_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();

        let input_ids_tensor = tract_ndarray::Array2::from_shape_vec((1, input_ids.len()), input_ids).unwrap().into_dyn();
        let attention_mask_tensor = tract_ndarray::Array2::from_shape_vec((1, attention_mask.len()), attention_mask).unwrap().into_dyn();

        // Run inference
        let outputs = model.run(tvec![
            Tensor::from(input_ids_tensor),
            Tensor::from(attention_mask_tensor)
        ]).expect("error running inference");

        // Get logits and apply softmax
        let logits = outputs[0].to_array_view::<f32>().unwrap();
        let probs = softmax(logits.as_slice().unwrap());
        trace!("Logits: {:?}", logits);
        trace!("Probabilities: {:?}", probs);

        // Find the index of the element with the largest value
        let max_index = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        trace!("Max probability index: {}", max_index);

        // Map to sentiment labels
        let labels = ["very negative", "negative", "neutral", "positive", "very positive"];
        let predicted_label = labels[max_index];
        trace!("Predicted label: {}", predicted_label);

        self.send_http_response(
                    418,
                    vec![("Powered-By", "proxy-wasm"), ("Predicted-Label", format!("{}", predicted_label).as_str())],
                    Some(format!("Input: {} \r\nOk\r\n", input_text).as_bytes()),
                );
        Action::Continue
    }
}

// Simple softmax implementation
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

