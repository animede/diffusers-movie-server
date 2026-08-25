import subprocess
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import Body, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from safetensors import safe_open

from .config import settings
from .generator import interrupt_controller
from .jobs import JobManager
from .schemas import (
    AssetResponse,
    ConcatRequest,
    GenerateRequest,
    JobResponse,
    LoraResponse,
    PromptEnhanceRequest,
    PromptEnhanceResponse,
    SessionResponse,
)

BASE_DIR = Path(__file__).resolve().parent
settings.output_dir.mkdir(parents=True, exist_ok=True)
settings.input_dir.mkdir(parents=True, exist_ok=True)
settings.lora_dir.mkdir(parents=True, exist_ok=True)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".gif"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

app = FastAPI(title="LTX-2.5 Studio", version="1.0.0")
manager = JobManager(settings)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": settings.model_id,
        "loaded": manager.generator._pipe is not None,
        "prompt_enhancer": bool(settings.llm_base_url and settings.llm_model),
    }


@app.post("/api/interrupt")
def api_interrupt(job_id: Optional[str] = Body(None, embed=True)):
    """Request that the currently-running job stop.

    If `job_id` is given and does not match the job currently executing, this is a
    no-op (so a request racing with the start of an unrelated job cannot kill it).
    Only checked at denoise step boundaries (see `generator.GenerationInterrupted`'s
    docstring), so reaction time is bounded by roughly one step's wall-clock cost --
    this is the same contract as the H3 backend's `POST /api/interrupt` (same
    endpoint name/request/response shape), so callers do not need to special-case
    either backend. An interrupted job settles into `status: "interrupted"` when
    polled via `GET /api/jobs/{job_id}` (there is no synchronous HTTP response to
    return an error code on here, since job submission is async/queued -- see
    `POST /api/jobs`)."""
    active = [job.id for job in manager.jobs.values() if job.status == "running"]
    current_job_id = active[0] if active else None
    requested = interrupt_controller.request(job_id)
    return {
        "interrupted": requested,
        "current_job_id": current_job_id,
        "requested_job_id": job_id,
    }


@app.post("/api/admin/unload")
def admin_unload():
    """Release pipelines/VRAM while keeping the process alive (Phase 5a resident
    switching). 409 while any job is queued/running. Subsequent generation requests
    lazy-load again via LTXGenerator.load() (no restart needed)."""
    active = [job.id for job in manager.jobs.values() if job.status in ("queued", "running")]
    if active:
        raise HTTPException(status_code=409, detail=f"Generation in progress ({len(active)} job(s)); cannot unload")
    result = manager.generator.unload()
    return {"result": "unloaded", **result}


@app.post("/api/sessions", response_model=SessionResponse, status_code=201)
def create_session():
    return SessionResponse(session_number=manager.create_session())


@app.get("/api/jobs", response_model=list[JobResponse])
def list_jobs(session_number: int | None = None, limit: int = 50):
    return [job.public() for job in manager.history(session_number, min(max(limit, 1), 200))]


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    try:
        if not manager.delete(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=204)


@app.post("/api/jobs/concat")
def concat_jobs(request: ConcatRequest):
    sources = []
    for job_id in request.job_ids:
        job = manager.get(job_id)
        path = settings.output_dir / f"{job_id}.mp4"
        if job is None or job.status != "completed" or not path.is_file():
            raise HTTPException(status_code=404, detail=f"Completed video not found: {job_id}")
        sources.append(path)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate", "-of", "json", str(sources[0])],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise HTTPException(status_code=400, detail="Could not inspect the first video")
    import json
    stream = json.loads(probe.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    rate = stream.get("avg_frame_rate", "24/1")

    command = ["ffmpeg", "-y", "-v", "error"]
    for source in sources:
        command += ["-i", str(source)]
    filters = []
    concat_inputs = []
    for index in range(len(sources)):
        filters.append(
            f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={rate},format=yuv420p,setpts=PTS-STARTPTS[v{index}]"
        )
        filters.append(
            f"[{index}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=PTS-STARTPTS[a{index}]"
        )
        concat_inputs.append(f"[v{index}][a{index}]")
    filters.append("".join(concat_inputs) + f"concat=n={len(sources)}:v=1:a=1[v][a]")
    output_id = uuid.uuid4().hex
    target = settings.output_dir / f"combined-{output_id}.mp4"
    command += [
        "-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", str(settings.ltx25_video_crf), "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Video concatenation failed: {result.stderr[-500:]}")
    return {"video_url": f"/outputs/{target.name}", "filename": target.name}


@app.get("/api/loras", response_model=list[LoraResponse])
def list_loras():
    items = []
    for path in sorted(settings.lora_dir.glob("*.safetensors"), key=lambda item: item.name.lower()):
        if path.is_file():
            metadata = {}
            try:
                with safe_open(path, framework="pt", device="cpu") as weights:
                    metadata = weights.metadata() or {}
            except Exception:
                pass
            factor = None
            try:
                factor = float(metadata["reference_downscale_factor"])
            except (KeyError, TypeError, ValueError):
                pass
            is_ic = factor is not None or "ic-lora" in path.name.lower() or "iclora" in path.name.lower()
            items.append(LoraResponse(
                id=path.name,
                name=path.stem,
                size=path.stat().st_size,
                kind="iclora" if is_ic else "standard",
                model_version=metadata.get("model_version"),
                reference_downscale_factor=factor,
                generic_iclora_compatible=is_ic and factor == 1.0,
            ))
    return items


@app.post("/api/prompts/enhance", response_model=PromptEnhanceResponse)
def enhance_prompt(request: PromptEnhanceRequest):
    if not settings.llm_base_url or not settings.llm_model:
        raise HTTPException(
            status_code=503,
            detail="External prompt enhancer is not configured (LLM_BASE_URL and LLM_MODEL are required)",
        )
    mode_notes = {
        "t2av": "Create the scene from text only.",
        "i2v": "Preserve the supplied first frame, subject identity, composition, and clothing.",
        "flf2v": "Respect both supplied endpoint frames and describe a plausible transition between them.",
        "condition": "Respect all supplied visual references without inventing conflicting identities.",
        "iclora": "Use the IC-LoRA reference as identity and appearance context. Preserve its characters, props, and location while describing a new action.",
        "retake": "Describe only the replacement action and sound for the selected time region; preserve continuity with the source video.",
        "extend": "Describe only what happens in the newly extended portion, continuing motion, identity, scene, and sound seamlessly.",
        "a2v": "Describe visuals that synchronize precisely with the supplied fixed audio; do not invent replacement dialogue or sound.",
        "t2i": "Describe a single still image: composition, subject, lighting, texture, and mood. No motion or audio.",
        "refine_image": "Describe the supplied reference image faithfully so it can be re-rendered at higher fidelity; preserve identity, composition, and wardrobe.",
        "ref2i": "Keep the referenced character identity, wardrobe, and style while describing a new scene, pose, or lighting for one still image.",
    }
    shot_text = "\n".join(f"Shot {i + 1}: {shot}" for i, shot in enumerate(request.shots) if shot.strip())
    user_text = request.prompt + (f"\n\nRequested shot plan:\n{shot_text}" if shot_text else "")
    payload = {
        "model": settings.llm_model,
        "temperature": 0.3,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Rewrite the user's idea as one production-ready English prompt for LTX-2.5 audio-video "
                    "generation. Preserve intent and do not add unrelated content. Use present tense and concrete "
                    "visual language. Specify shot size, subject and action, environment, lighting, camera movement, "
                    "motion timing, ambience, sound effects, and quoted dialogue when present. For multiple shots, "
                    "write explicit 'Hard cut to' transitions and keep identity, wardrobe, location, lighting, voice, "
                    "and style consistent. Preserve every quoted line verbatim in its original language: never "
                    "translate, paraphrase, romanize, or add words to dialogue. Explicitly state the spoken language "
                    "and speaker before each quoted line. Do not include explanations, headings, markdown, negative prompts, or "
                    f"parameter advice. Generation mode note: {mode_notes[request.mode]}"
                ),
            },
            {"role": "user", "content": user_text},
        ],
    }
    headers = {"content-type": "application/json"}
    if settings.llm_api_key:
        headers["authorization"] = f"Bearer {settings.llm_api_key}"
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    try:
        with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            enhanced = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Prompt enhancement failed: {type(exc).__name__}") from exc
    if not enhanced:
        raise HTTPException(status_code=502, detail="Prompt enhancement returned an empty response")
    return PromptEnhanceResponse(prompt=enhanced, model=settings.llm_model)


@app.post("/api/assets", response_model=AssetResponse, status_code=201)
async def upload_asset(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        kind = "image"
    elif suffix in VIDEO_SUFFIXES:
        kind = "video"
    elif suffix in AUDIO_SUFFIXES:
        kind = "audio"
    else:
        raise HTTPException(status_code=415, detail="Supported inputs: JPG, PNG, WebP, MP4, MOV, WebM, MKV, GIF, WAV, MP3, M4A, FLAC, OGG, AAC")

    asset_id = uuid.uuid4().hex
    temporary = settings.input_dir / f".{asset_id}{suffix}.part"
    destination = settings.input_dir / f"{asset_id}{suffix}"
    limit = settings.max_upload_size_mb * 1024 * 1024
    size = 0
    try:
        with temporary.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"Upload exceeds {settings.max_upload_size_mb} MB")
                output.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if kind == "image":
            with Image.open(temporary) as image:
                image.verify()
        elif kind == "video" and suffix != ".gif":
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(temporary)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if probe.returncode != 0 or not probe.stdout.strip():
                raise HTTPException(status_code=400, detail="Uploaded video could not be decoded")
        elif kind == "audio":
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(temporary)],
                capture_output=True, text=True, timeout=30,
            )
            if probe.returncode != 0 or not probe.stdout.strip():
                raise HTTPException(status_code=400, detail="Uploaded audio could not be decoded")
        temporary.replace(destination)
    except HTTPException:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid {kind} file") from exc
    finally:
        await file.close()
    return AssetResponse(id=asset_id, kind=kind, filename=file.filename or destination.name, size=size)


@app.post("/api/jobs", response_model=JobResponse, status_code=202)
def create_job(request: GenerateRequest):
    try:
        return manager.submit(request).public()
    except Exception as exc:
        if exc.__class__.__name__ == "Full":
            raise HTTPException(status_code=429, detail="Generation queue is full") from exc
        raise


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.public()
