//! Protocol translation layer (port of `services/protocol/`). Translates
//! OpenAI/Anthropic API requests into internal conversation calls and formats
//! the responses.

pub mod anthropic_v1_messages;
pub mod chat_completion_cache;
pub mod conversation;
pub mod openai_search;
pub mod openai_v1_chat_complete;
pub mod openai_v1_image_edit;
pub mod openai_v1_image_generations;
pub mod openai_v1_models;
pub mod openai_v1_response;
