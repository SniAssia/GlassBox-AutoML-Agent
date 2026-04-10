use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

#[wasm_bindgen]
pub async fn run_automl_bridge(csv_base64: String, target: String, task: String) -> Result<JsValue, JsValue> {
    let mut opts = RequestInit::new();
    opts.method("POST");
    opts.mode(RequestMode::Cors);

    // Préparation du corps JSON pour FastAPI
    let body = obj_to_json(&csv_base64, &target, &task);
    opts.body(Some(&JsValue::from_str(&body)));

    let url = "http://localhost:8000/run-automl";
    let request = Request::new_with_str_and_init(&url, &opts)?;

    request.headers().set("Content-Type", "application/json")?;

    let window = web_sys::window().ok_or("No window found")?;
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_value.dyn_into()?;

    // On récupère le résultat du "Zoo de modèles"
    let json = JsFuture::from(resp.json()?).await?;
    Ok(json)
}

fn obj_to_json(csv: &str, target: &str, task: &str) -> String {
    format!(
        r#"{{"csv_base64": "{}", "target_column": "{}", "task_type": "{}"}}"#,
        csv, target, task
    )
}