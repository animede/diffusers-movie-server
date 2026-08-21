"""gateway 側アセット管理(POST /api/v1/assets)。

アップロードを gateway/data/assets/ に保存して asset_id を発行する。
generate 時に asset_ids で参照し、
- ltx25 向け: バックエンドの /api/assets へ転送してバックエンド側 ID に変換
- h3 向け: gateway が multipart を組み立てて同期 API へ添付
する(変換・添付は jobs.py / modes.py 側)。

メタデータは <id>.json のサイドカーに保存する(gateway 再起動後も参照可能)。
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

GATEWAY_DIR = Path(__file__).resolve().parent
ASSET_DIR = GATEWAY_DIR / "data" / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

MAX_ASSET_SIZE_MB = 500

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".gif", ".avi"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


class AssetError(ValueError):
    """400/415 相当のアセットエラー。"""


def classify_suffix(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    raise AssetError(
        f"サポートしていない拡張子です: {suffix!r}"
        f"(画像 {sorted(IMAGE_SUFFIXES)} / 動画 {sorted(VIDEO_SUFFIXES)} / "
        f"音声 {sorted(AUDIO_SUFFIXES)})")


def save_asset(filename: str, reader) -> dict:
    """reader(同期 read(size) を持つファイルオブジェクト)を保存してメタを返す。"""
    suffix = Path(filename or "").suffix.lower()
    kind = classify_suffix(suffix)  # 未対応拡張子は AssetError

    asset_id = uuid.uuid4().hex
    limit = MAX_ASSET_SIZE_MB * 1024 * 1024
    tmp_path = ASSET_DIR / f".{asset_id}{suffix}.part"
    dest_path = ASSET_DIR / f"{asset_id}{suffix}"
    size = 0
    try:
        with tmp_path.open("wb") as out:
            while chunk := reader.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise AssetError(f"アップロードサイズが上限 {MAX_ASSET_SIZE_MB}MB を超えています")
                out.write(chunk)
        if size == 0:
            raise AssetError("アップロードされたファイルが空です")
        tmp_path.replace(dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    meta = {
        "id": asset_id,
        "kind": kind,
        "filename": filename or dest_path.name,
        "suffix": suffix,
        "size": size,
        "created_at": time.time(),
    }
    (ASSET_DIR / f"{asset_id}.json").write_text(json.dumps(meta, ensure_ascii=False))
    return meta


def get_asset(asset_id: str) -> dict:
    """asset_id → メタ dict(path: Path 付き)。無ければ AssetError。"""
    if not asset_id or "/" in asset_id or ".." in asset_id:
        raise AssetError(f"不正な asset_id です: {asset_id!r}")
    meta_path = ASSET_DIR / f"{asset_id}.json"
    if not meta_path.is_file():
        raise AssetError(f"アセットが見つかりません: {asset_id!r}")
    meta = json.loads(meta_path.read_text())
    path = ASSET_DIR / f"{asset_id}{meta['suffix']}"
    if not path.is_file():
        raise AssetError(f"アセットの実体ファイルが見つかりません: {asset_id!r}")
    meta["path"] = path
    return meta
