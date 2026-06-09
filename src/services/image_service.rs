//! Port of `services/image_service.py` — the image-gallery service layer that
//! sits on top of [`ImageStorageService`] (raw byte storage / WebDAV) and
//! [`ImageTagsService`] (per-image tags).
//!
//! This module composes those two already-built services rather than
//! re-instantiating them. It provides:
//!   * thumbnail generation (`ensure_thumbnail` / `get_thumbnail`) using the
//!     `image` crate (EXIF-orientation aware, fit-within-320×320, never
//!     upscaling, re-encoded as PNG) with results cached on disk under
//!     `image_thumbnails/`;
//!   * raw image / download responses returned as an [`ImageFile`]
//!     (bytes + content-type + optional download filename) instead of FastAPI
//!     `FileResponse`/`Response` types — CORS headers are expected to be applied
//!     by middleware in the Axum layer;
//!   * listing images (reconciled + tagged + grouped by date);
//!   * deleting images (removing thumbnails and tags alongside the bytes);
//!   * exporting selected images as an in-memory ZIP archive.
//!
//! Differences from the Python original are intentional and noted inline:
//!   * thumbnail decode/encode uses the `image` crate instead of Pillow;
//!   * byte-reading methods are `async` because [`ImageStorageService`] I/O is
//!     async, and the `parking_lot` thumbnail-generation lock is only held over
//!     the synchronous decode+encode, never across an `.await`;
//!   * the two duplicate `download_images_zip` definitions in the Python file
//!     are merged into the fuller variant that falls back to WebDAV via
//!     `storage.get_bytes` for non-local images;
//!   * the background auto-cleanup scheduler, `storage_stats`, `compress_images`
//!     and `delete_to_target` (which rely on `shutil.disk_usage` / a thread) are
//!     out of scope for this module and are not ported here.

use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use axum::http::StatusCode;
use image::{DynamicImage, ImageDecoder, ImageFormat};
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::config::Config;
use crate::error::AppError;

use super::image_storage_service::ImageStorageService;
use super::image_tags_service::ImageTagsService;

/// Largest thumbnail edge, in pixels (mirrors `THUMBNAIL_SIZE = (320, 320)`).
const THUMBNAIL_SIZE: u32 = 320;

/// A raw file payload returned to the HTTP layer, replacing FastAPI's
/// `FileResponse` / `Response`. `filename`, when set, is the suggested
/// `Content-Disposition: attachment` name for downloads.
#[derive(Clone, Debug)]
pub struct ImageFile {
    pub bytes: Vec<u8>,
    pub content_type: String,
    pub filename: Option<String>,
}

/// Cloneable handle to the image-gallery service. Composes the storage and tag
/// services built elsewhere.
#[derive(Clone)]
pub struct ImageService {
    inner: Arc<Inner>,
}

struct Inner {
    config: Config,
    storage: ImageStorageService,
    tags: ImageTagsService,
    /// Serializes thumbnail decode+encode so two concurrent requests for the
    /// same thumbnail don't race on the output file. Never held across `.await`.
    gen_lock: Mutex<()>,
}

// ---------------------------------------------------------------------------
// free helpers (mirror the module-level helpers in the Python file)
// ---------------------------------------------------------------------------

fn thumb_err() -> AppError {
    AppError::message(StatusCode::UNPROCESSABLE_ENTITY, "failed to create thumbnail")
}

/// `_safe_relative_path` — reject empty / traversal paths, normalise to posix.
fn safe_relative_path(path: &str) -> Result<String, AppError> {
    let value = path.trim().replace('\\', "/");
    let value = value.trim_start_matches('/');
    let parts: Vec<&str> = value.split('/').filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        return Err(AppError::not_found("image not found"));
    }
    if parts.iter().any(|p| *p == "." || *p == "..") {
        return Err(AppError::not_found("image not found"));
    }
    Ok(parts.join("/"))
}

/// Last path segment of a posix relative path.
fn file_name_of(rel: &str) -> String {
    rel.rsplit('/').next().unwrap_or(rel).to_string()
}

/// `Path.stem` / `Path.suffix` split (suffix keeps its leading dot; a leading
/// dot is treated as part of the stem).
fn split_stem_suffix(name: &str) -> (String, String) {
    match name.rfind('.') {
        Some(i) if i > 0 => (name[..i].to_string(), name[i..].to_string()),
        _ => (name.to_string(), String::new()),
    }
}

/// Guess a content-type from a file name's extension (defaults to image/png).
fn content_type_for(name: &str) -> String {
    let ext = match name.rsplit_once('.') {
        Some((_, e)) if !e.is_empty() => e.to_ascii_lowercase(),
        _ => String::new(),
    };
    match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => "image/png",
    }
    .to_string()
}

/// Recursively collect every regular file under `root`.
fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.is_file() {
                out.push(p);
            }
        }
    }
    out
}

/// Recursively collect every directory strictly under `root` (excludes `root`).
fn walk_dirs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                out.push(p.clone());
                stack.push(p);
            }
        }
    }
    out
}

fn rel_posix(root: &Path, path: &Path) -> Option<String> {
    let r = path.strip_prefix(root).ok()?;
    Some(
        r.components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("/"),
    )
}

/// `_cleanup_empty_dirs` — remove empty directories deepest-first.
fn cleanup_empty_dirs(root: &Path) {
    let mut dirs = walk_dirs(root);
    dirs.sort_by(|a, b| b.components().count().cmp(&a.components().count()));
    for d in dirs {
        // `remove_dir` only succeeds on empty directories; errors are ignored.
        let _ = std::fs::remove_dir(&d);
    }
}

/// Decode image bytes, applying the EXIF orientation transform (mirrors
/// `ImageOps.exif_transpose`).
fn decode_image(data: &[u8]) -> Result<DynamicImage, AppError> {
    let reader = image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|_| thumb_err())?;
    let mut decoder = reader.into_decoder().map_err(|_| thumb_err())?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder).map_err(|_| thumb_err())?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// Coerce to an RGB / RGBA buffer so the PNG re-encode is well-defined (mirrors
/// the Pillow `convert("RGBA"|"RGB")` step).
fn normalize_mode(img: DynamicImage) -> DynamicImage {
    if img.color().has_alpha() {
        DynamicImage::ImageRgba8(img.to_rgba8())
    } else {
        DynamicImage::ImageRgb8(img.to_rgb8())
    }
}

/// De-duplicate a ZIP entry name against already-used names (`name`, then
/// `name_2`, `name_3`, …), recording the chosen name.
fn dedup_name(base: &str, used: &mut Vec<String>) -> String {
    if !used.iter().any(|n| n == base) {
        used.push(base.to_string());
        return base.to_string();
    }
    let (stem, suffix) = split_stem_suffix(base);
    let mut counter = 2;
    loop {
        let cand = format!("{stem}_{counter}{suffix}");
        if !used.iter().any(|n| n == &cand) {
            used.push(cand.clone());
            return cand;
        }
        counter += 1;
    }
}

// ---------------------------------------------------------------------------
// ImageService
// ---------------------------------------------------------------------------

impl ImageService {
    /// Compose the gallery service from the already-built storage and tag
    /// services (rather than constructing fresh ones).
    pub fn new(config: Config, storage: ImageStorageService, tags: ImageTagsService) -> Self {
        Self {
            inner: Arc::new(Inner {
                config,
                storage,
                tags,
                gen_lock: Mutex::new(()),
            }),
        }
    }

    /// `_safe_image_path` — resolve a relative path to a real file under the
    /// images root, rejecting traversal and missing files.
    fn safe_image_path(&self, relative_path: &str) -> Result<PathBuf, AppError> {
        let safe = safe_relative_path(relative_path)?;
        let root = self.inner.config.images_dir();
        let root = root.canonicalize().unwrap_or(root);
        let path = root
            .join(&safe)
            .canonicalize()
            .map_err(|_| AppError::not_found("image not found"))?;
        if !path.starts_with(&root) {
            return Err(AppError::not_found("image not found"));
        }
        if !path.is_file() {
            return Err(AppError::not_found("image not found"));
        }
        Ok(path)
    }

    /// `_thumbnail_path` — on-disk cache location for a thumbnail.
    fn thumbnail_path(&self, relative_path: &str) -> Result<PathBuf, AppError> {
        let safe = safe_relative_path(relative_path)?;
        Ok(self
            .inner
            .config
            .image_thumbnails_dir()
            .join(format!("{safe}.png")))
    }

    /// `thumbnail_url` — public URL for an image's thumbnail.
    pub fn thumbnail_url(&self, base_url: &str, relative_path: &str) -> Result<String, AppError> {
        let safe = safe_relative_path(relative_path)?;
        Ok(format!(
            "{}/image-thumbnails/{}",
            base_url.trim_end_matches('/'),
            safe
        ))
    }

    /// `get_image_response` — bytes + content-type for serving an image inline.
    pub async fn get_image(&self, relative_path: &str) -> Result<ImageFile, AppError> {
        if self.inner.storage.has_local(relative_path) {
            let path = self.safe_image_path(relative_path)?;
            let bytes = std::fs::read(&path).map_err(|_| AppError::not_found("image not found"))?;
            Ok(ImageFile {
                bytes,
                content_type: content_type_for(&file_name_of(relative_path)),
                filename: None,
            })
        } else {
            let bytes = self.inner.storage.get_bytes(relative_path).await?;
            Ok(ImageFile {
                bytes,
                content_type: "image/png".to_string(),
                filename: None,
            })
        }
    }

    /// `get_image_download_response` — bytes + content-type + download filename.
    pub async fn get_image_download(&self, relative_path: &str) -> Result<ImageFile, AppError> {
        if self.inner.storage.has_local(relative_path) {
            let path = self.safe_image_path(relative_path)?;
            let name = path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let bytes = std::fs::read(&path).map_err(|_| AppError::not_found("image not found"))?;
            Ok(ImageFile {
                content_type: content_type_for(&name),
                filename: Some(name),
                bytes,
            })
        } else {
            let rel = safe_relative_path(relative_path)?;
            let name = file_name_of(&rel);
            let bytes = self.inner.storage.get_bytes(&rel).await?;
            Ok(ImageFile {
                bytes,
                content_type: "image/png".to_string(),
                filename: Some(name),
            })
        }
    }

    /// `ensure_thumbnail` — generate (or reuse a fresh cached) thumbnail and
    /// return its on-disk path.
    pub async fn ensure_thumbnail(&self, relative_path: &str) -> Result<PathBuf, AppError> {
        let target = self.thumbnail_path(relative_path)?;

        let source_path = if self.inner.storage.has_local(relative_path) {
            Some(self.safe_image_path(relative_path)?)
        } else {
            None
        };
        let source_mtime: Option<SystemTime> = source_path
            .as_ref()
            .and_then(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());

        if let Ok(meta) = std::fs::metadata(&target) {
            if let Ok(target_mtime) = meta.modified() {
                // Fresh when there's no source mtime, or the cache is newer.
                let fresh = match source_mtime {
                    None => true,
                    Some(sm) => target_mtime >= sm,
                };
                if fresh {
                    return Ok(target);
                }
            }
        }

        // Read the source bytes (locally if available, else via storage) before
        // taking the synchronous generation lock.
        let data = match &source_path {
            Some(p) => std::fs::read(p).map_err(|_| thumb_err())?,
            None => self.inner.storage.get_bytes(relative_path).await?,
        };

        {
            let _guard = self.inner.gen_lock.lock();
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(|_| thumb_err())?;
            }
            let img = normalize_mode(decode_image(&data)?);
            // Fit within THUMBNAIL_SIZE preserving aspect ratio; never upscale
            // (matches PIL `Image.thumbnail`).
            let thumb = if img.width() > THUMBNAIL_SIZE || img.height() > THUMBNAIL_SIZE {
                img.resize(
                    THUMBNAIL_SIZE,
                    THUMBNAIL_SIZE,
                    image::imageops::FilterType::Lanczos3,
                )
            } else {
                img
            };
            thumb
                .save_with_format(&target, ImageFormat::Png)
                .map_err(|_| thumb_err())?;
        }

        Ok(target)
    }

    /// `get_thumbnail_response` — thumbnail bytes (PNG) for serving inline.
    pub async fn get_thumbnail(&self, relative_path: &str) -> Result<ImageFile, AppError> {
        let target = self.ensure_thumbnail(relative_path).await?;
        let bytes = std::fs::read(&target).map_err(|_| thumb_err())?;
        Ok(ImageFile {
            bytes,
            content_type: "image/png".to_string(),
            filename: None,
        })
    }

    /// `cleanup_image_thumbnails` — drop thumbnails whose source image no longer
    /// exists (or whose name isn't a `.png`). Returns the number removed.
    pub fn cleanup_image_thumbnails(&self) -> usize {
        let root = self.inner.config.image_thumbnails_dir();
        let mut removed = 0;
        for path in walk_files(&root) {
            let Some(rel) = rel_posix(&root, &path) else {
                continue;
            };
            let keep = rel.ends_with(".png")
                && self.inner.storage.exists(&rel[..rel.len() - 4]);
            if !keep && std::fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
        cleanup_empty_dirs(&root);
        removed
    }

    /// `list_images` — reconcile + tag + group images by date. Returns
    /// `{"items": [...], "groups": [{"date", "items"}, ...]}`.
    pub fn list_images(&self, base_url: &str, start_date: &str, end_date: &str) -> Value {
        self.inner.config.cleanup_old_images();
        self.cleanup_image_thumbnails();
        let all_tags = self.inner.tags.load_tags();

        let raw = self.inner.storage.list_items(base_url, start_date, end_date);
        let mut items: Vec<Value> = Vec::with_capacity(raw.len());
        for it in raw {
            let Value::Object(mut obj) = it else {
                continue;
            };
            let path = obj
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let url = obj
                .get("url")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    format!("{}/images/{}", base_url.trim_end_matches('/'), path)
                });
            obj.insert("url".into(), json!(url));
            obj.insert(
                "thumbnail_url".into(),
                json!(self.thumbnail_url(base_url, &path).unwrap_or_default()),
            );
            let tags = all_tags.get(&path).cloned().unwrap_or_else(|| json!([]));
            obj.insert("tags".into(), tags);
            items.push(Value::Object(obj));
        }

        // Group by date, preserving first-seen order (like a Python dict).
        let mut groups: Vec<(String, Vec<Value>)> = Vec::new();
        for it in &items {
            let date = it
                .get("date")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            match groups.iter_mut().find(|(k, _)| *k == date) {
                Some((_, v)) => v.push(it.clone()),
                None => groups.push((date, vec![it.clone()])),
            }
        }
        let groups_json: Vec<Value> = groups
            .into_iter()
            .map(|(date, value)| json!({ "date": date, "items": value }))
            .collect();

        json!({ "items": items, "groups": groups_json })
    }

    /// `delete_images` — delete the selected (or all date-matching) images,
    /// also removing their thumbnails and tags. Returns `{"removed": n}`.
    pub async fn delete_images(
        &self,
        paths: Option<Vec<String>>,
        start_date: &str,
        end_date: &str,
        all_matching: bool,
    ) -> Result<Value, AppError> {
        let images_root = self.inner.config.images_dir();
        let thumbs_root = self.inner.config.image_thumbnails_dir();

        let targets: Vec<String> = if all_matching {
            self.inner
                .storage
                .list_items("", start_date, end_date)
                .into_iter()
                .filter_map(|it| {
                    it.get("path")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                })
                .collect()
        } else {
            paths.unwrap_or_default()
        };

        let mut removed = 0i64;
        for item in &targets {
            // Traversal guard (Python guards via `resolve()` + `relative_to`).
            let Ok(safe) = safe_relative_path(item) else {
                continue;
            };
            if self.inner.storage.delete(item).await? {
                removed += 1;
            }
            // Both the `{rel}.png` thumbnail and a bare `{rel}` copy.
            for tp in [thumbs_root.join(format!("{safe}.png")), thumbs_root.join(&safe)] {
                if tp.is_file() {
                    let _ = std::fs::remove_file(&tp);
                }
            }
            self.inner.tags.remove_tags(item);
        }

        cleanup_empty_dirs(&images_root);
        cleanup_empty_dirs(&thumbs_root);
        Ok(json!({ "removed": removed }))
    }

    /// `download_images_zip` — build an in-memory ZIP of the selected images,
    /// falling back to WebDAV for images not present locally. Returns the raw
    /// ZIP bytes, or 404 if nothing was added.
    pub async fn download_images_zip(&self, paths: &[String]) -> Result<Vec<u8>, AppError> {
        let root = self.inner.config.images_dir();
        let mut buf = Cursor::new(Vec::new());
        let mut added = 0;
        let mut used_names: Vec<String> = Vec::new();

        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);

            for item in paths {
                let Ok(rel) = safe_relative_path(item) else {
                    continue;
                };
                let path = root.join(&rel);
                let payload: Vec<u8> = if path.is_file() {
                    match std::fs::read(&path) {
                        Ok(p) => p,
                        Err(_) => continue,
                    }
                } else {
                    match self.inner.storage.get_bytes(&rel).await {
                        Ok(p) => p,
                        Err(_) => continue,
                    }
                };
                let name = dedup_name(&file_name_of(&rel), &mut used_names);
                zip.start_file(name, options)
                    .map_err(|e| AppError::internal(format!("zip: {e}")))?;
                zip.write_all(&payload)
                    .map_err(|e| AppError::internal(format!("zip: {e}")))?;
                added += 1;
            }

            zip.finish()
                .map_err(|e| AppError::internal(format!("zip: {e}")))?;
        }

        if added == 0 {
            return Err(AppError::not_found("no images found"));
        }
        Ok(buf.into_inner())
    }
}
