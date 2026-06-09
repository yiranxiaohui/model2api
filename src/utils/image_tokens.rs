//! Port of `utils/image_tokens.py` — image token accounting (input patch/tile
//! counting and generated-image token estimates). Pure math; image dimensions
//! are read header-only via the `imagesize` crate.

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use serde_json::{json, Value};

pub const DEFAULT_IMAGE_SIZE: (i64, i64) = (1024, 1024);
const IMAGE_INPUT_TOKEN_MODEL: &str = "gpt-5.4-mini";

const PATCH_SIZE: f64 = 32.0;
const TILE_SIZE: f64 = 512.0;
const TILE_HIGH_SHORT_SIDE: f64 = 768.0;

const PATCH_1536_MODELS: &[&str] = &[
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5-codex-mini",
    "gpt-5.1-codex-mini",
    "gpt-5.2-codex",
    "gpt-5.2-chat-latest",
    "o4-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
];

fn patch_multiplier(model: &str) -> f64 {
    let name = model_name(model);
    // Order matters only in that the first matching prefix wins; the Python
    // dict preserves insertion order.
    for (prefix, mult) in [
        ("gpt-5.4-mini", 1.62),
        ("gpt-5.4-nano", 2.46),
        ("gpt-5-mini", 1.62),
        ("gpt-5-nano", 2.46),
        ("gpt-4.1-mini", 1.62),
        ("gpt-4.1-nano", 2.46),
        ("o4-mini", 1.72),
    ] {
        if name.starts_with(prefix) {
            return mult;
        }
    }
    1.0
}

fn model_name(model: &str) -> String {
    model.trim().to_ascii_lowercase()
}

pub fn image_size_from_bytes(data: &[u8]) -> Option<(i64, i64)> {
    if data.is_empty() {
        return None;
    }
    match imagesize::blob_size(data) {
        Ok(sz) if sz.width > 0 && sz.height > 0 => Some((sz.width as i64, sz.height as i64)),
        _ => None,
    }
}

fn decode_data_url(value: &str) -> Option<Vec<u8>> {
    let text = value.trim();
    let payload = if text.starts_with("data:") && text.contains(',') {
        text.splitn(2, ',').nth(1).unwrap_or("")
    } else {
        text
    };
    BASE64_STANDARD.decode(payload).ok()
}

pub fn image_size_from_data_url(value: &str) -> Option<(i64, i64)> {
    image_size_from_bytes(&decode_data_url(value)?)
}

pub fn parse_image_size(size: &Value) -> (i64, i64) {
    if let Some(arr) = size.as_array() {
        if arr.len() >= 2 {
            let w = arr[0].as_i64().or_else(|| arr[0].as_f64().map(|f| f as i64));
            let h = arr[1].as_i64().or_else(|| arr[1].as_f64().map(|f| f as i64));
            if let (Some(w), Some(h)) = (w, h) {
                if w > 0 && h > 0 {
                    return (w, h);
                }
            }
        }
    }
    let s = match size {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    let re = regex::Regex::new(r"(\d{2,5})\D+(\d{2,5})").unwrap();
    if let Some(cap) = re.captures(&s) {
        let w: i64 = cap[1].parse().unwrap_or(0);
        let h: i64 = cap[2].parse().unwrap_or(0);
        if w > 0 && h > 0 {
            return (w, h);
        }
    }
    DEFAULT_IMAGE_SIZE
}

fn patch_count(width: f64, height: f64) -> i64 {
    ((width / PATCH_SIZE).ceil() * (height / PATCH_SIZE).ceil()) as i64
}

fn patch_limits(model: &str, detail: &str) -> Option<(i64, i64)> {
    let name = model_name(model);
    if PATCH_1536_MODELS.iter().any(|p| name.starts_with(p)) {
        return Some((1536, 2048));
    }
    if name.starts_with("gpt-5.5") {
        return Some(if detail == "auto" || detail == "original" {
            (10000, 6000)
        } else {
            (2500, 2048)
        });
    }
    if name.starts_with("gpt-5.4") {
        return Some(if detail == "original" {
            (10000, 6000)
        } else {
            (2500, 2048)
        });
    }
    None
}

fn patch_tokens(width: i64, height: i64, model: &str, detail: &str) -> i64 {
    let multiplier = patch_multiplier(model);
    if detail == "low" {
        return (256.0 * multiplier).ceil() as i64;
    }
    let Some((patch_budget, max_dimension)) = patch_limits(model, detail) else {
        return 0;
    };
    let (w, h) = (width as f64, height as f64);
    let scale = (max_dimension as f64 / w.max(h)).min(1.0);
    let mut resized_w = w * scale;
    let mut resized_h = h * scale;

    if patch_count(resized_w, resized_h) > patch_budget {
        let shrink = ((PATCH_SIZE * PATCH_SIZE * patch_budget as f64) / (resized_w * resized_h)).sqrt();
        let width_units = resized_w * shrink / PATCH_SIZE;
        let height_units = resized_h * shrink / PATCH_SIZE;
        let adj = shrink
            * f64::min(
                if width_units != 0.0 {
                    width_units.floor() / width_units
                } else {
                    1.0
                },
                if height_units != 0.0 {
                    height_units.floor() / height_units
                } else {
                    1.0
                },
            );
        resized_w *= adj;
        resized_h *= adj;
    }

    let tokens = patch_count(resized_w.max(1.0), resized_h.max(1.0)).min(patch_budget);
    (tokens as f64 * multiplier).ceil() as i64
}

fn tile_rates(model: &str) -> (i64, i64) {
    let name = model_name(model);
    if name == "gpt-5" || name == "gpt-5-chat-latest" {
        return (70, 140);
    }
    if name.starts_with("gpt-4o-mini") {
        return (2833, 5667);
    }
    if name.starts_with("o1") || name.starts_with("o1-pro") || name.starts_with("o3") {
        return (75, 150);
    }
    if name.starts_with("computer-use-preview") {
        return (65, 129);
    }
    (85, 170)
}

#[allow(dead_code)]
fn tile_tokens(width: i64, height: i64, model: &str, detail: &str) -> i64 {
    let (base_tokens, tile_tok) = tile_rates(model);
    if detail == "low" {
        return base_tokens;
    }
    let (w, h) = (width as f64, height as f64);
    let scale = (2048.0 / w.max(h)).min(1.0);
    let mut resized_w = w * scale;
    let mut resized_h = h * scale;
    let short_side = resized_w.min(resized_h);
    if short_side > 0.0 {
        let s = TILE_HIGH_SHORT_SIDE / short_side;
        resized_w *= s;
        resized_h *= s;
    }
    let tiles = (resized_w / TILE_SIZE).ceil() as i64 * (resized_h / TILE_SIZE).ceil() as i64;
    base_tokens + tiles * tile_tok
}

pub fn count_image_input_tokens(width: i64, height: i64, _model: &str, detail: &str) -> i64 {
    if width <= 0 || height <= 0 {
        return 0;
    }
    let detail = {
        let d = detail.trim().to_ascii_lowercase();
        if d.is_empty() {
            "auto".to_string()
        } else {
            d
        }
    };
    patch_tokens(width, height, IMAGE_INPUT_TOKEN_MODEL, &detail)
}

fn part_size(part: &Value) -> Option<(i64, i64)> {
    let w = part.get("width").and_then(|v| v.as_i64()).unwrap_or(0);
    let h = part.get("height").and_then(|v| v.as_i64()).unwrap_or(0);
    if w > 0 && h > 0 {
        return Some((w, h));
    }
    // `image_url` data URL
    if let Some(image_url) = part.get("image_url") {
        let url = match image_url {
            Value::Object(m) => m
                .get("url")
                .or_else(|| m.get("image_url"))
                .and_then(|v| v.as_str())
                .map(String::from),
            Value::String(s) => Some(s.clone()),
            _ => None,
        };
        if let Some(u) = url {
            if u.starts_with("data:") {
                return image_size_from_data_url(&u);
            }
        }
    }
    // `source` (Anthropic-style base64)
    if let Some(source) = part.get("source").and_then(|v| v.as_object()) {
        if source.get("type").and_then(|v| v.as_str()) == Some("base64") {
            let data = source.get("data").and_then(|v| v.as_str()).unwrap_or("");
            if let Ok(bytes) = BASE64_STANDARD.decode(data) {
                return image_size_from_bytes(&bytes);
            }
        }
    }
    None
}

pub fn count_image_content_tokens(content: &Value, model: &str, default_detail: &str) -> i64 {
    let Some(parts) = content.as_array() else {
        return 0;
    };
    let mut total = 0;
    for part in parts {
        let Some(obj) = part.as_object() else { continue };
        let part_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("").trim();
        let has_source = obj.contains_key("source");
        if !matches!(part_type, "image" | "image_url" | "input_image") && !has_source {
            continue;
        }
        let Some((w, h)) = part_size(part) else { continue };
        let detail = obj
            .get("detail")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(default_detail);
        total += count_image_input_tokens(w, h, model, detail);
    }
    total
}

pub fn count_generated_image_tokens(width: i64, height: i64, quality: &str) -> i64 {
    let patches = patch_count(width as f64, height as f64) as f64;
    match quality.trim().to_ascii_lowercase().as_str() {
        "low" => (patches * 17.0 / 64.0).ceil() as i64,
        "high" | "hd" => (patches * 65.0 / 16.0).ceil() as i64,
        _ => (patches * 33.0 / 32.0).ceil() as i64,
    }
}

pub fn count_image_output_tokens(size: &Value, quality: &str, count: i64) -> i64 {
    let (w, h) = parse_image_size(size);
    count.max(0) * count_generated_image_tokens(w, h, quality)
}

pub fn count_image_output_items_tokens(items: &Value, size: &Value, quality: &str) -> i64 {
    let Some(arr) = items.as_array() else { return 0 };
    if arr.is_empty() {
        return 0;
    }
    let fallback = parse_image_size(size);
    let mut total = 0;
    for item in arr {
        let mut image_size = None;
        if let Some(b64) = item.get("b64_json").and_then(|v| v.as_str()) {
            let b64 = b64.trim();
            if !b64.is_empty() {
                if let Ok(bytes) = BASE64_STANDARD.decode(b64) {
                    image_size = image_size_from_bytes(&bytes);
                }
            }
        }
        let (w, h) = image_size.unwrap_or(fallback);
        total += count_generated_image_tokens(w, h, quality);
    }
    total
}

/// Build the OpenAI-style usage block (responses API).
pub fn token_usage(
    input_text: i64,
    input_image: i64,
    output_text: i64,
    output_image: i64,
) -> Value {
    let it = input_text.max(0);
    let ii = input_image.max(0);
    let ot = output_text.max(0);
    let oi = output_image.max(0);
    let input_tokens = it + ii;
    let output_tokens = ot + oi;
    json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": { "text_tokens": it, "image_tokens": ii, "cached_tokens": 0 },
        "output_tokens_details": { "text_tokens": ot, "image_tokens": oi, "reasoning_tokens": 0 },
    })
}

pub fn image_usage(input_text: i64, input_image: i64, output: i64) -> Value {
    token_usage(input_text, input_image, 0, output)
}

/// Convert a responses-style usage block into a chat-completions usage block.
pub fn chat_usage_from_image_usage(usage: &Value) -> Value {
    let input_tokens = usage.get("input_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
    let output_tokens = usage.get("output_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
    let id = usage.get("input_tokens_details");
    let od = usage.get("output_tokens_details");
    let g = |o: Option<&Value>, k: &str| o.and_then(|v| v.get(k)).and_then(|v| v.as_i64()).unwrap_or(0);
    json!({
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "prompt_tokens_details": {
            "text_tokens": g(id, "text_tokens"),
            "image_tokens": g(id, "image_tokens"),
            "cached_tokens": g(id, "cached_tokens"),
        },
        "completion_tokens_details": {
            "text_tokens": g(od, "text_tokens"),
            "image_tokens": g(od, "image_tokens"),
            "reasoning_tokens": g(od, "reasoning_tokens"),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_tokens_default_quality() {
        // 1024x1024 -> patch_count = 32*32 = 1024 patches; default = ceil(1024*33/32)=1056
        assert_eq!(count_generated_image_tokens(1024, 1024, "auto"), 1056);
        assert_eq!(count_generated_image_tokens(1024, 1024, "low"), (1024.0_f64 * 17.0 / 64.0).ceil() as i64);
        assert_eq!(count_generated_image_tokens(1024, 1024, "high"), (1024.0_f64 * 65.0 / 16.0).ceil() as i64);
    }

    #[test]
    fn parse_size_from_string_and_array() {
        assert_eq!(parse_image_size(&json!("1024x768")), (1024, 768));
        assert_eq!(parse_image_size(&json!([512, 512])), (512, 512));
        assert_eq!(parse_image_size(&json!("garbage")), DEFAULT_IMAGE_SIZE);
    }

    #[test]
    fn input_tokens_low_detail() {
        // low detail -> ceil(256 * multiplier(gpt-5.4-mini=1.62)) = ceil(414.72)=415
        assert_eq!(count_image_input_tokens(2048, 2048, "gpt-5.4-mini", "low"), 415);
    }
}
