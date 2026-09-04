from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ConditionInput(BaseModel):
    asset_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    kind: Literal["image", "video"]
    index: int = Field(default=0, ge=-1, le=60)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)


class LoraInput(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+\.safetensors$")
    strength: float = Field(default=1.0, ge=-2.0, le=2.0)


class LoraResponse(BaseModel):
    id: str
    name: str
    size: int
    kind: Literal["standard", "iclora"] = "standard"
    model_version: str | None = None
    reference_downscale_factor: float | None = None
    generic_iclora_compatible: bool = False


class ConcatRequest(BaseModel):
    job_ids: list[str] = Field(min_length=2, max_length=20)

    @model_validator(mode="after")
    def validate_unique_jobs(self):
        if len(set(self.job_ids)) != len(self.job_ids):
            raise ValueError("job_ids must be unique")
        if any(len(job_id) != 32 or not all(char in "0123456789abcdef" for char in job_id) for job_id in self.job_ids):
            raise ValueError("invalid job id")
        return self


class AssetResponse(BaseModel):
    id: str
    kind: Literal["image", "video", "audio"]
    filename: str
    size: int


STILL_IMAGE_MODES = {"t2i", "refine_image", "ref2i"}
REF2I_NUM_FRAMES = {25, 41, 49}


class GenerateRequest(BaseModel):
    session_number: int | None = Field(default=None, ge=1)
    mode: Literal[
        "t2av", "i2v", "flf2v", "condition", "iclora", "retake", "extend", "a2v",
        "t2i", "refine_image", "ref2i",
    ] = "t2av"
    # `quality` is retained for older API clients. New clients should control the
    # rendering pipeline explicitly with `upscale` and `decoder`.
    quality: Literal["high", "draft"] | None = Field(
        default=None, json_schema_extra={"deprecated": True}
    )
    upscale: bool = True
    upscale_method: Literal["latent", "pixel"] = "latent"
    temporal_upscale: bool = False
    # Per-request override of the decode path (None = follow LTX25_DECODER setting).
    # The diffusion decoder is currently available only after the 2x refine path.
    decoder: Literal["vae", "diffusion"] | None = None
    prompt: str = Field(min_length=1, max_length=8000)
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted", max_length=4000
    )
    width: int = Field(default=768, ge=256, le=1920, multiple_of=32)
    height: int = Field(default=512, ge=256, le=1920, multiple_of=32)
    num_frames: int | None = Field(default=121, ge=9, le=481)
    min_seconds: float = Field(default=1.0, ge=1.0, le=19.0)
    max_seconds: float = Field(default=8.0, gt=1.0, le=20.0)
    fps: float = Field(default=24.0, ge=8.0, le=60.0)
    steps: int = Field(default=30, ge=1, le=100)
    guidance_scale: float = Field(default=3.0, ge=0.0, le=20.0)
    # リップシンク調整ノブ(2026-09-03 追加、2026-09-04 訂正。既定 None = 1.0 固定)。
    # 【重要な訂正】「映像が音声をどれだけ聞くか」を決めるのは **modality_scale(映像側)**。
    # pipeline_ltx2.py の実装: video_modality_delta = (modality_scale-1) ×
    # (音声あり予測 − 音声なし予測)。audio_modality_scale は**音声予測**にしか効かず、
    # a2v では音声ラテントが凍結されるため出力に対して完全な no-op(追加 forward の
    # コスト ~+0.9s@8step だけ払う)。同一seed比較で出力が小数点まで一致することを
    # 実測済み(2026-09-04)。リップシンク目的では modality_scale を使うこと。
    # audio_guidance_scale も同様に a2v では実効なしの疑いが強い(音声CFG)。
    modality_scale: float | None = Field(default=None, ge=0.0, le=15.0)
    audio_guidance_scale: float | None = Field(default=None, ge=0.0, le=15.0)
    audio_modality_scale: float | None = Field(default=None, ge=0.0, le=15.0)
    seed: int = Field(default=42, ge=0, le=2**63 - 1)
    enhance_prompt: bool = False
    conditions: list[ConditionInput] = Field(default_factory=list, max_length=8)
    retake_start: float | None = Field(default=None, ge=0.0)
    retake_end: float | None = Field(default=None, gt=0.0)
    regenerate_video: bool = True
    regenerate_audio: bool = True
    extend_direction: Literal["end", "start"] = "end"
    extend_seconds: float = Field(default=5.0, ge=1.0, le=20.0)
    extend_context_seconds: float = Field(default=3.0, ge=0.5, le=20.0)
    audio_asset_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    audio_start: float = Field(default=0.0, ge=0.0)
    audio_duration: float | None = Field(default=None, ge=1.0, le=20.0)
    loras: list[LoraInput] = Field(default_factory=list, max_length=4)
    # Still-image modes (t2i / refine_image / ref2i)
    strength: float = Field(default=1.0, ge=0.1, le=1.0)  # refine_image reference strength
    frame_position: Literal["last", "center"] = "last"  # ref2i extracted frame

    @model_validator(mode="after")
    def validate_duration(self):
        if self.quality is not None:
            legacy_upscale = self.quality == "high"
            if "upscale" in self.model_fields_set and self.upscale != legacy_upscale:
                raise ValueError("quality and upscale specify conflicting render modes")
            self.upscale = legacy_upscale
        if self.max_seconds <= self.min_seconds:
            raise ValueError("max_seconds must be greater than min_seconds")
        if self.num_frames is not None and (self.num_frames - 1) % 8 != 0:
            raise ValueError("num_frames must be 8n+1 (for example 9, 121, 241 or 481)")
        if self.upscale and self.width * self.height > 960 * 544:
            raise ValueError("2x upscale base resolution cannot exceed 960x544 pixels")
        if self.upscale_method == "pixel" and not self.upscale:
            raise ValueError("pixel upscale method requires upscale=true")
        if self.mode in STILL_IMAGE_MODES:
            # Probe-verified recipes (scratch_t2i_probe): t2i/refine_image are always
            # two-stage (distilled 8-sigma -> 2x latent upsample -> 3-sigma refine),
            # ref2i is single-stage at base resolution.
            if self.mode == "ref2i":
                if not self.conditions:
                    raise ValueError("ref2i mode requires at least one image reference condition")
                if any(condition.kind != "image" for condition in self.conditions):
                    raise ValueError("ref2i mode accepts image references only")
                if self.num_frames is None or "num_frames" not in self.model_fields_set:
                    self.num_frames = 49
                if self.num_frames not in REF2I_NUM_FRAMES:
                    raise ValueError("ref2i num_frames must be one of 25, 41 or 49")
                self.upscale = False
                self.upscale_method = "latent"
                self.temporal_upscale = False
                # decoder "diffusion" falls back to VAE with a warning in the generator
                # (image-conditioned diffusion decode blurs; probe results).
            else:
                if self.mode == "t2i" and self.conditions:
                    raise ValueError("t2i mode does not accept visual conditions")
                if self.mode == "refine_image" and (
                    len(self.conditions) != 1
                    or self.conditions[0].kind != "image"
                    or self.conditions[0].index != 0
                ):
                    raise ValueError("refine_image mode requires one image condition at index 0")
                self.num_frames = 9  # fixed recipe; t2i decoder="diffusion" promotes to 17 internally
                self.upscale = True
                if self.upscale_method == "pixel":
                    raise ValueError("Pixel Spatial Upscaler is currently available for video output only")
                self.temporal_upscale = False
            if self.width * self.height > 960 * 544 and self.mode != "ref2i":
                raise ValueError("still-image base resolution cannot exceed 960x544 pixels")
            if len({item.id for item in self.loras}) != len(self.loras):
                raise ValueError("the same LoRA cannot be selected more than once")
            return self
        if not (self.upscale or self.temporal_upscale) and self.decoder == "diffusion":
            raise ValueError("diffusion decoder currently requires a latent upscale/refine stage")
        if self.mode == "t2av" and self.conditions:
            raise ValueError("t2av mode does not accept visual conditions")
        if self.mode == "i2v":
            if len(self.conditions) != 1 or self.conditions[0].kind != "image" or self.conditions[0].index != 0:
                raise ValueError("i2v mode requires one image condition at index 0")
        if self.mode == "flf2v":
            expected = {(condition.kind, condition.index) for condition in self.conditions}
            if len(self.conditions) != 2 or expected != {("image", 0), ("image", -1)}:
                raise ValueError("flf2v mode requires first and last image conditions at indices 0 and -1")
        if self.mode == "condition" and not self.conditions:
            raise ValueError("condition mode requires at least one image or video condition")
        if self.mode == "iclora":
            if len(self.conditions) != 1 or self.conditions[0].index != 1:
                raise ValueError("iclora mode requires one image or video reference at index 1")
            if len(self.loras) != 1:
                raise ValueError("iclora mode requires exactly one IC-LoRA")
            if self.num_frames is None:
                raise ValueError("iclora mode does not support automatic duration")
            # Generic IC-LoRA references must remain attached throughout stage 1.
            self.upscale = False
            self.upscale_method = "latent"
            self.temporal_upscale = False
            self.decoder = "vae"
        if self.mode == "retake":
            if len(self.conditions) != 1 or self.conditions[0].kind != "video":
                raise ValueError("retake mode requires one source video")
            if self.retake_start is None or self.retake_end is None or self.retake_start >= self.retake_end:
                raise ValueError("retake_start must be less than retake_end")
            if not (self.regenerate_video or self.regenerate_audio):
                raise ValueError("retake must regenerate video, audio, or both")
            # Official Retake is a single-stage masked denoise at source resolution.
            self.upscale = False
            self.upscale_method = "latent"
            self.temporal_upscale = False
            self.decoder = "vae"
        if self.mode == "extend":
            if len(self.conditions) != 1 or self.conditions[0].kind != "video":
                raise ValueError("extend mode requires one source video")
            self.upscale = False
            self.upscale_method = "latent"
            self.temporal_upscale = False
            self.decoder = "vae"
        if self.mode == "a2v":
            if not self.audio_asset_id:
                raise ValueError("a2v mode requires an audio asset")
            # 2026-08-31 拡張(mv_studio_V3 の FLF長尺連鎖プローブ): 末尾フレーム
            # (index=-1)の画像条件も受け付ける。generator.py の a2v 実装は
            # 「音声凍結フックを通常パイプラインへ被せ、conditions は素通し」の
            # 構造で flf2v と同じ条件経路に乗るため、スキーマ緩和だけで
            # 「a2v(音声同期)+先頭/末尾フレーム条件」が成立する見込み。
            # 従来の画像0〜1枚(index=0)は完全後方互換。
            a2v_allowed = {("image", 0), ("image", -1)}
            a2v_got = [(c.kind, c.index) for c in self.conditions]
            if (
                len(self.conditions) > 2
                or any(pair not in a2v_allowed for pair in a2v_got)
                or len(set(a2v_got)) != len(a2v_got)
            ):
                raise ValueError(
                    "a2v mode accepts at most a first-frame (index 0) and a last-frame (index -1) image condition"
                )
        if len({item.id for item in self.loras}) != len(self.loras):
            raise ValueError("the same LoRA cannot be selected more than once")
        return self


class JobResponse(BaseModel):
    id: str
    session_number: int
    status: str
    progress: float = 0
    error: str | None = None
    video_url: str | None = None
    image_url: str | None = None
    request: GenerateRequest
    created_at: str
    updated_at: str
    generation_seconds: float | None = None
    peak_vram_gb: float | None = None


class SessionResponse(BaseModel):
    session_number: int


class PromptEnhanceRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    mode: Literal[
        "t2av", "i2v", "flf2v", "condition", "iclora", "retake", "extend", "a2v",
        "t2i", "refine_image", "ref2i",
    ] = "t2av"
    shots: list[str] = Field(default_factory=list, max_length=12)


class PromptEnhanceResponse(BaseModel):
    prompt: str
    model: str
