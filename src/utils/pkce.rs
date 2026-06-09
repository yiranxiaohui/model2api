//! Port of `utils/pkce.py` — PKCE code verifier/challenge (S256) for OAuth.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rand::RngCore;
use sha2::{Digest, Sha256};

/// Generate `(code_verifier, code_challenge)`. The verifier is 64 random bytes
/// base64url-encoded without padding; the challenge is the S256 of the verifier.
pub fn generate_pkce() -> (String, String) {
    let mut bytes = [0u8; 64];
    rand::thread_rng().fill_bytes(&mut bytes);
    let code_verifier = URL_SAFE_NO_PAD.encode(bytes);

    let digest = Sha256::digest(code_verifier.as_bytes());
    let code_challenge = URL_SAFE_NO_PAD.encode(digest);

    (code_verifier, code_challenge)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_and_challenge_are_unpadded_urlsafe() {
        let (v, c) = generate_pkce();
        assert!(!v.contains('='));
        assert!(!v.contains('+') && !v.contains('/'));
        // 64 bytes -> ceil(64/3)*4 = 88 chars, minus padding (=) -> 86
        assert_eq!(v.len(), 86);
        // sha256 = 32 bytes -> 43 chars unpadded
        assert_eq!(c.len(), 43);
        // Challenge must be the S256 of the verifier.
        let expect = URL_SAFE_NO_PAD.encode(Sha256::digest(v.as_bytes()));
        assert_eq!(c, expect);
    }
}
