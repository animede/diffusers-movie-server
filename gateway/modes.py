"""統一 mode → バックエンド別エンドポイント + パラメータ変換(宣言的テーブル)。

統一 mode(Phase 2):
- 両対応: t2v / flf2v / ref2v / t2i / ref2i
- i2v: ltx25 はネイティブ、h3 は fl2va の先頭フレームのみ指定で代替(実機仕様:
  /api/fl2va は image または last_image の片方だけでも受け付ける)
- ltx25 固有: a2v / extend / retake / iclora / refine_image(h3 指定時は 400)

共通パラメータ(params): prompt / negative_prompt / width / height / seconds /
num_frames / fps / steps / guidance_scale / seed。バックエンド固有名へマップする。
- h3: steps → num_inference_steps。negative_prompt / guidance_scale / fps は
  h3 API に存在しないため無視する(警告としてジョブの result 生成時ではなく
  変換時に notes へ記録)。num_frames のみ指定時は seconds = num_frames / 24 に換算。
  width/height は常に明示送信する(resolution プリセットは使わない —
  512x512 等のプリセット外サイズが 400 になるため。Phase 1 の発見事項 2)。
- ltx25: steps → steps。seconds のみ指定時は num_frames = 8n+1 へ丸めて送る
  (fps 既定 24)。

extra(dict)はバックエンド固有パラメータの素通し(既知キーのみのホワイトリスト方式)。
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# 定義テーブル
# ---------------------------------------------------------------------------

class ModeError(ValueError):
    """400 に変換されるモード/パラメータ変換エラー。"""


COMMON_PARAM_KEYS = {
    "prompt", "negative_prompt", "width", "height", "seconds", "num_frames",
    "fps", "steps", "guidance_scale", "seed",
}

# 統一mode → h3 エンドポイント
H3_MODE_TABLE = {
    "t2v": "/api/t2va",
    "i2v": "/api/fl2va",     # 先頭フレームのみ指定の fl2va で代替
    "flf2v": "/api/fl2va",
    "ref2v": "/api/ref2va",
    "t2i": "/api/t2i",
    "ref2i": "/api/ref2va",  # still=1
}

# 統一mode → ltx25 の GenerateRequest.mode
LTX25_MODE_TABLE = {
    "t2v": "t2av",
    "i2v": "i2v",
    "flf2v": "flf2v",
    "ref2v": "condition",
    "t2i": "t2i",
    "ref2i": "ref2i",
    # ltx25 固有(統一名 = バックエンド名)
    "a2v": "a2v",
    "extend": "extend",
    "retake": "retake",
    "iclora": "iclora",
    "refine_image": "refine_image",
}

UNIFIED_MODES = sorted(set(H3_MODE_TABLE) | set(LTX25_MODE_TABLE))

# extra ホワイトリスト(バックエンド固有パラメータの素通し)
H3_EXTRA_KEYS = {
    "cache", "cache_threshold", "attn", "turbo", "mute",  # 即反映パラメータ
    "upscale",            # t2va のアップスケール
    "frames",             # t2i / ref2i の still フレーム数(22|5)
    # ref2va/ref2i の参照画像正規化短辺(未指定ならバックエンド側の
    # H3_REF_IMAGE_SHORT_EDGE 既定=2048を使う。backends/minimax-h3/app.py の
    # /api/ref2va 等が受け付ける任意パラメータをそのまま素通しする)。
    "reference_image_short_edge",
    # ref2va の Vocal Lock をリクエスト単位で切替(未指定ならバックエンド側の
    # H3_VOCAL_LOCK 環境変数の値を使う。/api/ref2va のみ対応、ref2va_batch/ref2i_batch
    # は未対応 -- backends/minimax-h3/app.py 参照)。
    "vocal_lock",
}
LTX25_EXTRA_KEYS = {
    "upscale", "upscale_method", "temporal_upscale", "decoder",
    "min_seconds", "max_seconds", "enhance_prompt",
    "retake_start", "retake_end", "regenerate_video", "regenerate_audio",
    "extend_direction", "extend_seconds", "extend_context_seconds",
    "audio_start", "audio_duration",
    # a2v リップシンク調整ノブ(2026-09-03、backends/ltx2_5/app/schemas.py 参照)
    "audio_guidance_scale", "audio_modality_scale",
    "loras", "strength", "frame_position", "session_number",
    "conditions",  # ConditionInput の index/strength 上書き(asset_ids と同順の dict 配列)
}

H3_FPS = 24.0  # minimax-h3 は 24fps 固定(core/runner.py の FPS)


def _suffix_content_type(suffix: str) -> str:
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".bmp": "image/bmp",
        ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
        ".mkv": "video/x-matroska", ".gif": "image/gif",
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
        ".flac": "audio/flac", ".ogg": "audio/ogg", ".aac": "audio/aac",
    }.get(suffix.lower(), "application/octet-stream")


def _check_params(params: dict[str, Any]) -> None:
    unknown = set(params) - COMMON_PARAM_KEYS
    if unknown:
        raise ModeError(
            f"params に未知のキーがあります: {sorted(unknown)}"
            f"(共通パラメータ: {sorted(COMMON_PARAM_KEYS)}。"
            "バックエンド固有パラメータは extra に入れてください)")


def _check_extra(backend: str, extra: dict[str, Any]) -> None:
    allowed = H3_EXTRA_KEYS if backend == "h3" else LTX25_EXTRA_KEYS
    unknown = set(extra) - allowed
    if unknown:
        raise ModeError(
            f"extra に許可されていないキーがあります(backend={backend}): "
            f"{sorted(unknown)}(有効: {sorted(allowed)})")


def _seconds_to_frames_8n1(seconds: float, fps: float) -> int:
    """ltx25 の num_frames(8n+1、9〜481)へ丸める。"""
    n = int(round(float(seconds) * float(fps)))
    n = max(9, min(481, n))
    return ((n - 1 + 7) // 8) * 8 + 1 if (n - 1) % 8 else n


# ---------------------------------------------------------------------------
# h3 変換
# ---------------------------------------------------------------------------

def build_h3_request(mode: str, params: dict[str, Any], extra: dict[str, Any],
                     assets: list[dict]) -> tuple[str, dict, list, list[str]]:
    """h3 の同期エンドポイント向け multipart を組み立てる。

    assets: gateway 側アセットメタ(path/kind/filename)の配列(asset_ids と同順)。
    戻り値: (path, form_data, files, notes)。files は httpx の files 形式
    (フィールド名, (filename, bytes, content_type)) のリスト。
    notes は無視した共通パラメータ等の注記。
    """
    if mode not in H3_MODE_TABLE:
        if mode in LTX25_MODE_TABLE:
            raise ModeError(
                f"mode {mode!r} は ltx25 固有です(h3 では使えません)")
        raise ModeError(f"未知の mode です: {mode!r}(有効: {UNIFIED_MODES})")
    _check_params(params)
    _check_extra("h3", extra)

    notes: list[str] = []
    for ignored in ("negative_prompt", "guidance_scale", "fps"):
        if params.get(ignored) is not None:
            notes.append(f"h3 は {ignored} をサポートしないため無視しました")

    prompt = params.get("prompt")
    if not prompt or not str(prompt).strip():
        raise ModeError("params.prompt は必須です")

    data: dict[str, Any] = {"prompt": str(prompt)}

    # 解像度: 常に height/width を明示(resolution プリセットは使わない)
    width = params.get("width")
    height = params.get("height")
    if (width is None) != (height is None):
        raise ModeError("width と height は両方指定するか両方省略してください")
    if width is not None:
        data["width"] = int(width)
        data["height"] = int(height)
    elif mode in ("t2v", "i2v", "flf2v", "t2i"):
        # h3 の既定キャンバス(768x768)を明示送信(ref2v/ref2i は省略時に
        # サーバが参照から 16:9 キャンバスを解決するため送らない)
        data["width"] = 768
        data["height"] = 768

    # 尺
    seconds = params.get("seconds")
    if seconds is None and params.get("num_frames") is not None:
        seconds = float(params["num_frames"]) / H3_FPS
        notes.append(f"num_frames={params['num_frames']} を seconds={seconds:.2f} に換算しました(24fps)")
    if mode in ("t2v", "i2v", "flf2v", "ref2v") and seconds is not None:
        data["seconds"] = float(seconds)

    if params.get("steps") is not None:
        data["num_inference_steps"] = int(params["steps"])
    if params.get("seed") is not None:
        data["seed"] = int(params["seed"])

    # extra 素通し
    for key, value in extra.items():
        if key == "frames" and mode not in ("t2i", "ref2i"):
            raise ModeError("extra.frames は t2i / ref2i でのみ使えます")
        if key == "upscale" and mode != "t2v":
            raise ModeError("extra.upscale(h3)は t2v でのみ使えます")
        data[key] = value
    if isinstance(data.get("mute"), bool):
        data["mute"] = "true" if data["mute"] else "false"
    if isinstance(data.get("vocal_lock"), bool):
        data["vocal_lock"] = "true" if data["vocal_lock"] else "false"

    # アセット → ファイルフィールド
    files: list[tuple[str, tuple[str, bytes, str]]] = []

    def _file_tuple(asset: dict) -> tuple[str, bytes, str]:
        path = asset["path"]
        return (asset["filename"], path.read_bytes(),
                _suffix_content_type(path.suffix))

    image_assets = [a for a in assets if a["kind"] == "image"]

    if mode == "t2v" or mode == "t2i":
        if assets:
            raise ModeError(f"mode {mode!r}(h3)はアセットを受け付けません")
    elif mode == "i2v":
        if len(image_assets) != 1 or len(assets) != 1:
            raise ModeError("i2v(h3)は画像アセットちょうど1件が必要です(先頭フレーム)")
        files.append(("image", _file_tuple(image_assets[0])))
    elif mode == "flf2v":
        if len(image_assets) != 2 or len(assets) != 2:
            raise ModeError(
                "flf2v(h3)は画像アセットちょうど2件が必要です"
                "(asset_ids の順に 先頭フレーム, 末尾フレーム)")
        files.append(("image", _file_tuple(image_assets[0])))
        files.append(("last_image", _file_tuple(image_assets[1])))
    elif mode in ("ref2v", "ref2i"):
        if not assets:
            raise ModeError(f"{mode}(h3)は参照アセットが1件以上必要です")
        for asset in assets:
            files.append(("references", _file_tuple(asset)))
        if mode == "ref2i":
            data["still"] = 1

    return H3_MODE_TABLE[mode], data, files, notes


# ---------------------------------------------------------------------------
# ltx25 変換
# ---------------------------------------------------------------------------

def build_ltx25_request(mode: str, params: dict[str, Any], extra: dict[str, Any],
                        backend_assets: list[dict]) -> tuple[dict, list[str]]:
    """ltx25 の POST /api/jobs 向け JSON を組み立てる。

    backend_assets: ltx25 側へ転送済みのアセット
    ({"id": <ltx25側asset_id>, "kind": "image"|"video"|"audio"} の配列、asset_ids と同順)。
    戻り値: (json_body, notes)。
    """
    if mode not in LTX25_MODE_TABLE:
        raise ModeError(f"未知の mode です: {mode!r}(有効: {UNIFIED_MODES})")
    _check_params(params)
    _check_extra("ltx25", extra)

    notes: list[str] = []
    prompt = params.get("prompt")
    if not prompt or not str(prompt).strip():
        raise ModeError("params.prompt は必須です")

    body: dict[str, Any] = {"mode": LTX25_MODE_TABLE[mode], "prompt": str(prompt)}

    if params.get("negative_prompt") is not None:
        body["negative_prompt"] = str(params["negative_prompt"])
    for src, dst in (("width", "width"), ("height", "height"),
                     ("fps", "fps"), ("steps", "steps"),
                     ("guidance_scale", "guidance_scale"), ("seed", "seed")):
        if params.get(src) is not None:
            body[dst] = params[src]

    fps = float(params.get("fps") or 24.0)
    if params.get("num_frames") is not None:
        body["num_frames"] = int(params["num_frames"])
    elif params.get("seconds") is not None:
        body["num_frames"] = _seconds_to_frames_8n1(float(params["seconds"]), fps)
        notes.append(
            f"seconds={params['seconds']} を num_frames={body['num_frames']} に換算しました"
            f"(fps={fps:g}、8n+1 丸め)")

    # extra 素通し(conditions は後で個別処理)
    conditions_override = extra.get("conditions")
    for key, value in extra.items():
        if key == "conditions":
            continue
        body[key] = value

    # アセット → conditions / audio_asset_id
    image_assets = [a for a in backend_assets if a["kind"] == "image"]
    video_assets = [a for a in backend_assets if a["kind"] == "video"]
    audio_assets = [a for a in backend_assets if a["kind"] == "audio"]
    visual_assets = [a for a in backend_assets if a["kind"] in ("image", "video")]

    def _cond(asset: dict, index: int = 0, strength: float = 1.0) -> dict:
        return {"asset_id": asset["id"], "kind": asset["kind"],
                "index": index, "strength": strength}

    conditions: list[dict] = []
    if mode == "t2v" or mode == "t2i":
        if backend_assets:
            raise ModeError(f"mode {mode!r}(ltx25)はアセットを受け付けません")
    elif mode == "i2v":
        if len(image_assets) != 1 or len(backend_assets) != 1:
            raise ModeError("i2v(ltx25)は画像アセットちょうど1件が必要です(先頭フレーム)")
        conditions = [_cond(image_assets[0], index=0)]
    elif mode == "flf2v":
        if len(image_assets) != 2 or len(backend_assets) != 2:
            raise ModeError(
                "flf2v(ltx25)は画像アセットちょうど2件が必要です"
                "(asset_ids の順に 先頭フレーム, 末尾フレーム)")
        conditions = [_cond(image_assets[0], index=0),
                      _cond(image_assets[1], index=-1)]
    elif mode == "ref2v":
        if not visual_assets:
            raise ModeError("ref2v(ltx25)は画像/動画アセットが1件以上必要です")
        conditions = [_cond(a) for a in visual_assets]
    elif mode == "ref2i":
        if not image_assets or len(image_assets) != len(backend_assets):
            raise ModeError("ref2i(ltx25)は画像アセットが1件以上必要です(画像のみ)")
        conditions = [_cond(a) for a in image_assets]
    elif mode == "refine_image":
        if len(image_assets) != 1 or len(backend_assets) != 1:
            raise ModeError("refine_image(ltx25)は画像アセットちょうど1件が必要です")
        conditions = [_cond(image_assets[0], index=0)]
    elif mode in ("extend", "retake"):
        if len(video_assets) != 1 or len(backend_assets) != 1:
            raise ModeError(f"{mode}(ltx25)は動画アセットちょうど1件が必要です")
        conditions = [_cond(video_assets[0], index=0)]
    elif mode == "iclora":
        if len(visual_assets) != 1:
            raise ModeError("iclora(ltx25)は画像/動画アセットちょうど1件が必要です(参照)")
        conditions = [_cond(visual_assets[0], index=1)]
        if not body.get("loras"):
            raise ModeError("iclora(ltx25)は extra.loras に IC-LoRA を1つ指定してください")
    elif mode == "a2v":
        if len(audio_assets) != 1:
            raise ModeError("a2v(ltx25)は音声アセットちょうど1件が必要です")
        # 2026-08-31拡張: 末尾フレーム画像(2枚目)も受け付ける(バックエンド
        # ltx2_5 e063d2f で index=-1 の画像条件が解禁済み)。asset_ids の画像順は
        # [先頭フレーム, 末尾フレーム]。実測: 音声同期を保ったまま両端が参照画像に
        # 錨止めされ、遷移はマッチディゾルブになる(mv_studio_V3 の
        # docs/HANDOFF.md「FLF長尺連鎖」プローブ参照)。従来の画像0〜1枚は完全互換。
        if len(image_assets) > 2 or len(video_assets) > 0:
            raise ModeError(
                "a2v(ltx25)は音声1件 + 任意の先頭フレーム画像1件"
                "(+任意の末尾フレーム画像1件)のみです")
        body["audio_asset_id"] = audio_assets[0]["id"]
        if image_assets:
            conditions = [_cond(image_assets[0], index=0)]
        if len(image_assets) == 2:
            conditions.append(_cond(image_assets[1], index=-1))

    # extra.conditions で index/strength を上書き(asset_ids と同順の dict 配列)
    if conditions_override:
        if not isinstance(conditions_override, list) or len(conditions_override) != len(conditions):
            raise ModeError(
                "extra.conditions は生成された conditions と同じ長さの配列で指定してください"
                f"(現在 {len(conditions)} 件)")
        for cond, override in zip(conditions, conditions_override):
            if not isinstance(override, dict):
                raise ModeError("extra.conditions の要素は dict で指定してください")
            for key in ("index", "strength"):
                if key in override:
                    cond[key] = override[key]

    if conditions:
        body["conditions"] = conditions
    return body, notes
