//! Shared application state (`AppState`), the Rust equivalent of the module-level
//! singletons in the Python project. Cloneable `Arc` handle passed to all axum
//! handlers.

use std::path::PathBuf;
use std::sync::Arc;

use crate::config::Config;
use crate::services::account_service::AccountService;
use crate::services::auth_service::AuthService;
use crate::services::backup_service::BackupService;
use crate::services::content_filter::ContentFilter;
use crate::services::cpa_service::CpaService;
use crate::services::editable_file_task_service::EditableFileTaskService;
use crate::services::image_service::ImageService;
use crate::services::image_storage_service::ImageStorageService;
use crate::services::image_tags_service::ImageTagsService;
use crate::services::image_task_service::ImageTaskService;
use crate::services::log_service::LogService;
use crate::services::oauth_login_service::OAuthLoginService;
use crate::services::protocol::conversation::ConvDeps;
use crate::services::register_service::RegisterService;
use crate::services::storage::{create_storage_backend, StorageBackend};
use crate::services::sub2api_service::Sub2apiService;

pub type SharedState = Arc<AppState>;

pub struct AppState {
    pub config: Config,
    pub base_dir: PathBuf,
    pub web_dist_dir: PathBuf,
    pub storage: Arc<dyn StorageBackend>,
    pub log: LogService,
    pub accounts: Arc<AccountService>,
    pub auth: Arc<AuthService>,
    pub oauth_login: Arc<OAuthLoginService>,
    pub image_storage: ImageStorageService,
    pub image_tags: ImageTagsService,
    pub image_service: ImageService,
    pub image_tasks: ImageTaskService,
    pub editable_tasks: EditableFileTaskService,
    pub content_filter: ContentFilter,
    pub backup: BackupService,
    pub cpa: CpaService,
    pub sub2api: Sub2apiService,
    pub register: RegisterService,
    pub conv: ConvDeps,
}

impl AppState {
    pub fn new(base_dir: PathBuf, config: Config) -> anyhow::Result<SharedState> {
        let storage: Arc<dyn StorageBackend> =
            Arc::from(create_storage_backend(config.data_dir())?);
        let web_dist_dir = base_dir.join("web_dist");

        let log = LogService::new(config.data_dir().join("logs.jsonl"));
        let accounts = Arc::new(AccountService::new(
            Arc::clone(&storage),
            config.clone(),
            log.clone(),
        ));
        let auth = Arc::new(AuthService::new(Arc::clone(&storage), config.clone()));
        let oauth_login = Arc::new(OAuthLoginService::new(config.clone()));

        let image_storage = ImageStorageService::new(config.clone());
        let image_tags = ImageTagsService::new(config.clone());
        let image_service =
            ImageService::new(config.clone(), image_storage.clone(), image_tags.clone());

        let conv = ConvDeps {
            config: config.clone(),
            accounts: Arc::clone(&accounts),
            image_storage: image_storage.clone(),
        };
        let image_tasks = ImageTaskService::new(conv.clone());
        let editable_tasks = EditableFileTaskService::new(config.clone());

        let content_filter = ContentFilter::new(config.clone());
        let backup = BackupService::new(config.clone());
        let cpa = CpaService::new(config.clone(), Arc::clone(&accounts));
        let sub2api = Sub2apiService::new(config.clone(), Arc::clone(&accounts));
        let register = RegisterService::new(config.clone(), Arc::clone(&accounts));

        Ok(Arc::new(AppState {
            config,
            base_dir,
            web_dist_dir,
            storage,
            log,
            accounts,
            auth,
            oauth_login,
            image_storage,
            image_tags,
            image_service,
            image_tasks,
            editable_tasks,
            content_filter,
            backup,
            cpa,
            sub2api,
            register,
            conv,
        }))
    }
}
