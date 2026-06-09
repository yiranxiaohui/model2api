//! OpenAI OAuth / Auth0 constants shared by the OAuth-bridge login service
//! (Phase 5) and the registration flow (Phase 9). Port of the module-level
//! constants in `services/register/openai_register.py`.

pub const AUTH_BASE: &str = "https://auth.openai.com";
pub const PLATFORM_BASE: &str = "https://platform.openai.com";
pub const PLATFORM_OAUTH_CLIENT_ID: &str = "app_2SKx67EdpoN0G6j64rFvigXD";
pub const PLATFORM_OAUTH_AUDIENCE: &str = "https://api.openai.com/v1";
pub const PLATFORM_AUTH0_CLIENT: &str = "eyJuYW1lIjoiYXV0aDAtc3BhLWpzIiwidmVyc2lvbiI6IjEuMjEuMCJ9";

pub fn platform_oauth_redirect_uri() -> String {
    format!("{PLATFORM_BASE}/auth/callback")
}

pub const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36";
pub const SEC_CH_UA: &str = "\"Google Chrome\";v=\"137\", \"Not?A_Brand\";v=\"8\", \"Chromium\";v=\"137\"";
pub const SEC_CH_UA_FULL_VERSION_LIST: &str =
    "\"Chromium\";v=\"137.0.0.0\", \"Not:A-Brand\";v=\"99.0.0.0\", \"Google Chrome\";v=\"137.0.0.0\"";

/// The `common_headers` dict (auth/oauth JSON requests).
pub fn common_headers() -> Vec<(&'static str, String)> {
    vec![
        ("accept", "application/json".to_string()),
        ("accept-encoding", "gzip, deflate, br".to_string()),
        ("accept-language", "en-US,en;q=0.9".to_string()),
        ("cache-control", "no-cache".to_string()),
        ("connection", "keep-alive".to_string()),
        ("content-type", "application/json".to_string()),
        ("dnt", "1".to_string()),
        ("origin", AUTH_BASE.to_string()),
        ("priority", "u=1, i".to_string()),
        ("sec-gpc", "1".to_string()),
        ("sec-ch-ua", SEC_CH_UA.to_string()),
        ("sec-ch-ua-arch", "\"x86_64\"".to_string()),
        ("sec-ch-ua-bitness", "\"64\"".to_string()),
        ("sec-ch-ua-full-version-list", SEC_CH_UA_FULL_VERSION_LIST.to_string()),
        ("sec-ch-ua-mobile", "?0".to_string()),
        ("sec-ch-ua-model", "\"\"".to_string()),
        ("sec-ch-ua-platform", "\"Windows\"".to_string()),
        ("sec-ch-ua-platform-version", "\"10.0.0\"".to_string()),
        ("sec-fetch-dest", "empty".to_string()),
        ("sec-fetch-mode", "cors".to_string()),
        ("sec-fetch-site", "same-origin".to_string()),
        ("user-agent", USER_AGENT.to_string()),
    ]
}
