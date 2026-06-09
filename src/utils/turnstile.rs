//! Port of `utils/turnstile.py` — Cloudflare Turnstile challenge solver.
//!
//! The challenge `dx` is base64-decoded, XOR-decrypted with key `p`, and parsed
//! as a list of bytecode tokens executed by a tiny dynamically-typed VM. This is
//! a faithful port of the common execution paths (XOR, base64 enc/dec, string
//! concat, `Reflect.set` into an ordered map, `performance.now`, `Object.keys`).
//! The rarely-exercised "call an arbitrary VM function with pre-resolved values"
//! branches are treated as no-ops, matching Python's try/except-skip behaviour
//! closely enough for the real challenges.

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use rand::Rng;
use serde_json::Value;

#[derive(Clone)]
enum TVal {
    Undefined,
    Str(String),
    Num(f64),
    List(Vec<TVal>),
    Map(Rc<RefCell<OrderedMap>>),
    Func(i64),
}

#[derive(Default)]
struct OrderedMap {
    keys: Vec<String>,
    values: HashMap<String, TVal>,
}

impl OrderedMap {
    fn add(&mut self, key: String, value: TVal) {
        if !self.values.contains_key(&key) {
            self.keys.push(key.clone());
        }
        self.values.insert(key, value);
    }
}

fn numeric_key(v: &Value) -> u64 {
    v.as_f64().unwrap_or(0.0).to_bits()
}

fn val_to_tval(v: &Value) -> TVal {
    match v {
        Value::Null => TVal::Undefined,
        Value::String(s) => TVal::Str(s.clone()),
        Value::Number(n) => TVal::Num(n.as_f64().unwrap_or(0.0)),
        Value::Bool(b) => TVal::Str(if *b { "true" } else { "false" }.into()),
        Value::Array(a) => TVal::List(a.iter().map(val_to_tval).collect()),
        Value::Object(_) => TVal::Undefined,
    }
}

fn py_float_str(f: f64) -> String {
    if f.fract() == 0.0 && f.is_finite() {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

fn special_string(value: &str) -> String {
    match value {
        "window.Math" => "[object Math]",
        "window.Reflect" => "[object Reflect]",
        "window.performance" => "[object Performance]",
        "window.localStorage" => "[object Storage]",
        "window.Object" => "function Object() { [native code] }",
        "window.Reflect.set" => "function set() { [native code] }",
        "window.performance.now" => "function () { [native code] }",
        "window.Object.create" => "function create() { [native code] }",
        "window.Object.keys" => "function keys() { [native code] }",
        "window.Math.random" => "function random() { [native code] }",
        other => other,
    }
    .to_string()
}

fn to_str(v: &TVal) -> String {
    match v {
        TVal::Undefined => "undefined".into(),
        TVal::Num(f) => py_float_str(*f),
        TVal::Str(s) => special_string(s),
        TVal::List(items) => items
            .iter()
            .map(|i| match i {
                TVal::Str(s) => s.clone(),
                other => to_str(other),
            })
            .collect::<Vec<_>>()
            .join(","),
        TVal::Map(_) => "[object Object]".into(),
        TVal::Func(_) => "function () { [native code] }".into(),
    }
}

fn xor_string(text: &str, key: &str) -> String {
    if key.is_empty() {
        return text.to_string();
    }
    let key_chars: Vec<u32> = key.chars().map(|c| c as u32).collect();
    text.chars()
        .enumerate()
        .map(|(i, ch)| {
            let xored = (ch as u32) ^ key_chars[i % key_chars.len()];
            char::from_u32(xored).unwrap_or('\u{FFFD}')
        })
        .collect()
}

struct Vm {
    map: HashMap<u64, TVal>,
    token_list: Vec<Vec<Value>>,
    result: String,
    start: std::time::Instant,
}

impl Vm {
    fn get(&self, v: &Value) -> TVal {
        self.map.get(&numeric_key(v)).cloned().unwrap_or(TVal::Undefined)
    }

    fn set(&mut self, v: &Value, val: TVal) {
        self.map.insert(numeric_key(v), val);
    }

    fn exec(&mut self, op: i64, args: &[Value]) {
        match op {
            1 => {
                // process_map[e] = xor(str(map[e]), str(map[t]))
                if args.len() >= 2 {
                    let s = xor_string(&to_str(&self.get(&args[0])), &to_str(&self.get(&args[1])));
                    self.set(&args[0], TVal::Str(s));
                }
            }
            2 => {
                // process_map[e] = t (literal)
                if args.len() >= 2 {
                    self.set(&args[0], val_to_tval(&args[1]));
                }
            }
            3 => {
                // result = base64(e), e is a literal string
                if let Some(e) = args.first() {
                    let s = match e {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    self.result = BASE64_STANDARD.encode(s.as_bytes());
                }
            }
            5 => {
                if args.len() >= 2 {
                    let current = self.get(&args[0]);
                    let incoming = self.get(&args[1]);
                    let next = match (&current, &incoming) {
                        (TVal::List(items), _) => {
                            let mut v = items.clone();
                            v.push(incoming.clone());
                            TVal::List(v)
                        }
                        (TVal::Str(_), _)
                        | (TVal::Num(_), _)
                        | (_, TVal::Str(_))
                        | (_, TVal::Num(_)) => {
                            TVal::Str(format!("{}{}", to_str(&current), to_str(&incoming)))
                        }
                        _ => TVal::Str("NaN".into()),
                    };
                    self.set(&args[0], next);
                }
            }
            6 => {
                if args.len() >= 3 {
                    let tv = self.get(&args[1]);
                    let nv = self.get(&args[2]);
                    if let (TVal::Str(t), TVal::Str(n)) = (&tv, &nv) {
                        let value = format!("{t}.{n}");
                        let resolved = if value == "window.document.location" {
                            "https://chatgpt.com/".to_string()
                        } else {
                            value
                        };
                        self.set(&args[0], TVal::Str(resolved));
                    }
                }
            }
            7 => {
                // Reflect.set(obj, key, val) is the common path.
                if let Some(e) = args.first() {
                    let target = self.get(e);
                    let values: Vec<TVal> = args[1..].iter().map(|a| self.get(a)).collect();
                    if matches!(&target, TVal::Str(s) if s == "window.Reflect.set") {
                        if values.len() == 3 {
                            if let TVal::Map(map) = &values[0] {
                                map.borrow_mut().add(to_str(&values[1]), values[2].clone());
                            }
                        }
                    }
                    // Generic "call a VM function" branch is intentionally skipped.
                }
            }
            8 => {
                if args.len() >= 2 {
                    let v = self.get(&args[1]);
                    self.set(&args[0], v);
                }
            }
            14 => {
                // json.loads(map[t]); only valid on strings.
                if args.len() >= 2 {
                    if let TVal::Str(s) = self.get(&args[1]) {
                        if let Ok(parsed) = serde_json::from_str::<Value>(&s) {
                            self.set(&args[0], val_to_tval(&parsed));
                        }
                    }
                }
            }
            15 => {
                // json.dumps(map[t]); on a Map this errors in Python and is skipped.
                if args.len() >= 2 {
                    match self.get(&args[1]) {
                        TVal::Str(s) => {
                            self.set(&args[0], TVal::Str(serde_json::to_string(&s).unwrap_or(s)))
                        }
                        TVal::Num(n) => self.set(&args[0], TVal::Str(py_float_str(n))),
                        TVal::List(_) | TVal::Undefined => {}
                        _ => {}
                    }
                }
            }
            17 => self.exec_17(args),
            18 => {
                if let Some(e) = args.first() {
                    let s = to_str(&self.get(e));
                    if let Ok(decoded) = BASE64_STANDARD.decode(s.as_bytes()) {
                        if let Ok(text) = String::from_utf8(decoded) {
                            self.set(e, TVal::Str(text));
                        }
                    }
                }
            }
            19 => {
                if let Some(e) = args.first() {
                    let s = to_str(&self.get(e));
                    self.set(e, TVal::Str(BASE64_STANDARD.encode(s.as_bytes())));
                }
            }
            20 => {
                // if map[e] == map[t]: call map[n] — generic call skipped.
            }
            21 => {}
            23 => {
                // if map[e] is not None and map[t] callable: call — skipped (generic).
            }
            24 => {
                if args.len() >= 3 {
                    let tv = self.get(&args[1]);
                    let nv = self.get(&args[2]);
                    if let (TVal::Str(t), TVal::Str(n)) = (&tv, &nv) {
                        self.set(&args[0], TVal::Str(format!("{t}.{n}")));
                    }
                }
            }
            _ => {}
        }
    }

    fn exec_17(&mut self, args: &[Value]) {
        if args.len() < 2 {
            return;
        }
        let e = &args[0];
        let target = self.get(&args[1]);
        let call_args: Vec<TVal> = args[2..].iter().map(|a| self.get(a)).collect();
        let target_str = match &target {
            TVal::Str(s) => s.clone(),
            _ => String::new(),
        };
        match target_str.as_str() {
            "window.performance.now" => {
                let elapsed_ms = self.start.elapsed().as_secs_f64() * 1000.0;
                let jitter = rand::thread_rng().gen::<f64>() / 1e6;
                self.set(e, TVal::Num(elapsed_ms + jitter));
            }
            "window.Object.create" => {
                self.set(e, TVal::Map(Rc::new(RefCell::new(OrderedMap::default()))));
            }
            "window.Object.keys" => {
                if matches!(call_args.first(), Some(TVal::Str(s)) if s == "window.localStorage") {
                    let keys = [
                        "STATSIG_LOCAL_STORAGE_INTERNAL_STORE_V4",
                        "STATSIG_LOCAL_STORAGE_STABLE_ID",
                        "client-correlated-secret",
                        "oai/apps/capExpiresAt",
                        "oai-did",
                        "STATSIG_LOCAL_STORAGE_LOGGING_REQUEST",
                        "UiState.isNavigationCollapsed.1",
                    ]
                    .iter()
                    .map(|s| TVal::Str(s.to_string()))
                    .collect();
                    self.set(e, TVal::List(keys));
                }
            }
            "window.Math.random" => {
                self.set(e, TVal::Num(rand::thread_rng().gen::<f64>()));
            }
            _ => { /* generic callable target skipped */ }
        }
    }
}

/// Solve a Turnstile challenge. Returns the base64 result string, or `None` if
/// decoding/parsing fails or the bytecode produced no result.
pub fn solve_turnstile_token(dx: &str, p: &str) -> Option<String> {
    let decoded_bytes = BASE64_STANDARD.decode(dx).ok()?;
    let decoded = String::from_utf8(decoded_bytes).ok()?;
    let xored = xor_string(&decoded, p);
    let token_list: Vec<Vec<Value>> = serde_json::from_str(&xored).ok()?;

    let mut map: HashMap<u64, TVal> = HashMap::new();
    // Functions live at their op-id keys; data slots seeded by the challenge.
    for op in [1, 2, 3, 5, 6, 7, 8, 14, 15, 17, 18, 19, 20, 21, 23, 24] {
        map.insert((op as f64).to_bits(), TVal::Func(op));
    }
    map.insert((9f64).to_bits(), TVal::List(token_list.iter().flat_map(|t| t.iter().map(val_to_tval)).collect()));
    map.insert((10f64).to_bits(), TVal::Str("window".into()));
    map.insert((16f64).to_bits(), TVal::Str(p.to_string()));

    let mut vm = Vm {
        map,
        token_list: token_list.clone(),
        result: String::new(),
        start: std::time::Instant::now(),
    };

    let tokens = vm.token_list.clone();
    for token in &tokens {
        if token.is_empty() {
            continue;
        }
        let Some(op) = token[0].as_i64() else { continue };
        // Each op handler guards its own arity; mirror Python's try/except-skip.
        vm.exec(op, &token[1..]);
    }

    if vm.result.is_empty() {
        None
    } else {
        Some(vm.result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_roundtrip() {
        let key = "secret";
        let original = "hello world";
        let enc = xor_string(original, key);
        let dec = xor_string(&enc, key);
        assert_eq!(dec, original);
    }

    #[test]
    fn invalid_input_returns_none() {
        assert!(solve_turnstile_token("not-base64!!!", "p").is_none());
    }

    #[test]
    fn simple_base64_result() {
        // token list: [[3, "abc"]] XOR'd with empty key, base64-encoded.
        let list = serde_json::to_string(&serde_json::json!([[3, "abc"]])).unwrap();
        let dx = BASE64_STANDARD.encode(list.as_bytes());
        let out = solve_turnstile_token(&dx, "").unwrap();
        assert_eq!(out, BASE64_STANDARD.encode("abc"));
    }
}
