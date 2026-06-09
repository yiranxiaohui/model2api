//! Account registration subsystem (port of `services/register/`). The mail
//! providers and the full registration orchestration land in Phase 9; the
//! shared OAuth/Auth0 [`constants`] are defined here early because the
//! OAuth-bridge login service (Phase 5) depends on them.

pub mod constants;
pub mod mail_provider;
pub mod openai_register;
