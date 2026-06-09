//! Port of `services/storage/git_storage.py` — store accounts/auth-keys as JSON
//! files in a private Git repo, syncing via clone/pull/commit/push (`git2`).

use std::path::{Path, PathBuf};

use git2::{Cred, FetchOptions, PushOptions, RemoteCallbacks, Repository};
use serde_json::{json, Value};

use super::StorageBackend;

pub struct GitStorageBackend {
    repo_url: String,
    auth_repo_url: String,
    token: String,
    branch: String,
    file_path: String,
    auth_keys_file_path: String,
    local_cache_dir: PathBuf,
}

fn build_auth_url(repo_url: &str, token: &str) -> String {
    if token.is_empty() {
        return repo_url.to_string();
    }
    if let Some(rest) = repo_url.strip_prefix("https://") {
        return format!("https://{token}@{rest}");
    }
    if let Some(rest) = repo_url.strip_prefix("git@") {
        let https = format!("https://{}", rest.replacen(".com:", ".com/", 1));
        if let Some(r) = https.strip_prefix("https://") {
            return format!("https://{token}@{r}");
        }
    }
    repo_url.to_string()
}

impl GitStorageBackend {
    pub fn new(
        repo_url: &str,
        token: &str,
        branch: &str,
        file_path: &str,
        auth_keys_file_path: &str,
        local_cache_dir: PathBuf,
    ) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&local_cache_dir)?;
        Ok(Self {
            repo_url: repo_url.to_string(),
            auth_repo_url: build_auth_url(repo_url, token),
            token: token.to_string(),
            branch: branch.to_string(),
            file_path: file_path.to_string(),
            auth_keys_file_path: auth_keys_file_path.to_string(),
            local_cache_dir,
        })
    }

    fn credentials_callbacks(&self) -> RemoteCallbacks<'_> {
        let mut cb = RemoteCallbacks::new();
        let token = self.token.clone();
        cb.credentials(move |_url, username, _allowed| {
            if token.is_empty() {
                Cred::default()
            } else {
                // GitHub PAT over HTTPS: token as username, empty password.
                Cred::userpass_plaintext(&token, "")
            }
            .or_else(|_| Cred::username(username.unwrap_or("git")))
        });
        cb
    }

    fn clone_or_pull(&self) -> anyhow::Result<Repository> {
        let repo_path = self.local_cache_dir.join("repo");
        if repo_path.join(".git").exists() {
            match self.pull_existing(&repo_path) {
                Ok(repo) => return Ok(repo),
                Err(_) => {
                    let _ = std::fs::remove_dir_all(&repo_path);
                }
            }
        }
        let mut fo = FetchOptions::new();
        fo.remote_callbacks(self.credentials_callbacks());
        let mut builder = git2::build::RepoBuilder::new();
        builder.branch(&self.branch);
        builder.fetch_options(fo);
        let repo = builder.clone(&self.auth_repo_url, &repo_path)?;
        Ok(repo)
    }

    fn pull_existing(&self, repo_path: &Path) -> anyhow::Result<Repository> {
        let repo = Repository::open(repo_path)?;
        {
            let mut remote = repo.find_remote("origin")?;
            let mut fo = FetchOptions::new();
            fo.remote_callbacks(self.credentials_callbacks());
            remote.fetch(&[&self.branch], Some(&mut fo), None)?;
            let fetch_head = repo.find_reference("FETCH_HEAD")?;
            let fetch_commit = repo.reference_to_annotated_commit(&fetch_head)?;
            let (analysis, _) = repo.merge_analysis(&[&fetch_commit])?;
            if analysis.is_fast_forward() {
                let refname = format!("refs/heads/{}", self.branch);
                if let Ok(mut reference) = repo.find_reference(&refname) {
                    reference.set_target(fetch_commit.id(), "fast-forward")?;
                    repo.set_head(&refname)?;
                    repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
                } else {
                    repo.reference(&refname, fetch_commit.id(), true, "setting branch")?;
                    repo.set_head(&refname)?;
                    repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
                }
            }
        }
        Ok(repo)
    }

    fn load_json_value(&self, file_path: &str) -> anyhow::Result<Value> {
        let repo = self.clone_or_pull()?;
        let workdir = repo.workdir().ok_or_else(|| anyhow::anyhow!("no workdir"))?;
        let full = workdir.join(file_path);
        if !full.exists() {
            return Ok(Value::Null);
        }
        let text = std::fs::read_to_string(&full)?;
        Ok(serde_json::from_str(&text)?)
    }

    fn save_json_file(&self, file_path: &str, items: Value, message: &str) -> anyhow::Result<()> {
        let repo = self.clone_or_pull()?;
        let workdir = repo.workdir().ok_or_else(|| anyhow::anyhow!("no workdir"))?.to_path_buf();
        let full = workdir.join(file_path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(&items)? + "\n";
        std::fs::write(&full, content)?;

        let mut index = repo.index()?;
        index.add_path(Path::new(file_path))?;
        index.write()?;
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let sig = repo
            .signature()
            .or_else(|_| git2::Signature::now("model2api", "model2api@local"))?;
        let parent_commit = repo.head().ok().and_then(|h| h.peel_to_commit().ok());

        // Only commit if the tree changed.
        let changed = match &parent_commit {
            Some(c) => c.tree_id() != tree_id,
            None => true,
        };
        if !changed {
            return Ok(());
        }
        let parents: Vec<&git2::Commit> = parent_commit.iter().collect();
        repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &parents)?;

        let mut remote = repo.find_remote("origin")?;
        let mut po = PushOptions::new();
        po.remote_callbacks(self.credentials_callbacks());
        let refspec = format!("refs/heads/{}:refs/heads/{}", self.branch, self.branch);
        remote.push(&[&refspec], Some(&mut po))?;
        Ok(())
    }

    fn last_commit_short(&self) -> Option<String> {
        let repo = self.clone_or_pull().ok()?;
        let commit = repo.head().ok()?.peel_to_commit().ok()?;
        Some(commit.id().to_string()[..8].to_string())
    }
}

impl StorageBackend for GitStorageBackend {
    fn load_accounts(&self) -> Vec<Value> {
        match self.load_json_value(&self.file_path) {
            Ok(Value::Array(a)) => a,
            _ => vec![],
        }
    }

    fn save_accounts(&self, accounts: &[Value]) -> anyhow::Result<()> {
        self.save_json_file(&self.file_path, json!(accounts), "Update accounts data")
    }

    fn load_auth_keys(&self) -> Vec<Value> {
        match self.load_json_value(&self.auth_keys_file_path) {
            Ok(Value::Array(a)) => a,
            Ok(Value::Object(o)) => o.get("items").and_then(|v| v.as_array()).cloned().unwrap_or_default(),
            _ => vec![],
        }
    }

    fn save_auth_keys(&self, auth_keys: &[Value]) -> anyhow::Result<()> {
        self.save_json_file(&self.auth_keys_file_path, json!({"items": auth_keys}), "Update auth keys data")
    }

    fn health_check(&self) -> Value {
        match self.last_commit_short() {
            Some(commit) => json!({
                "status": "healthy",
                "backend": "git",
                "repo_url": mask_token(&self.repo_url),
                "branch": self.branch,
                "file_path": self.file_path,
                "auth_keys_file_path": self.auth_keys_file_path,
                "last_commit": commit,
            }),
            None => json!({"status": "unhealthy", "backend": "git", "error": "clone/pull failed"}),
        }
    }

    fn get_backend_info(&self) -> Value {
        json!({
            "type": "git",
            "description": "Git 私有仓库存储",
            "repo_url": mask_token(&self.repo_url),
            "branch": self.branch,
            "file_path": self.file_path,
            "auth_keys_file_path": self.auth_keys_file_path,
        })
    }
}

fn mask_token(url: &str) -> String {
    if url.contains('@') && url.contains("://") {
        if let Some((protocol, rest)) = url.split_once("://") {
            if let Some((_, host)) = rest.split_once('@') {
                return format!("{protocol}://****@{host}");
            }
        }
    }
    url.to_string()
}
