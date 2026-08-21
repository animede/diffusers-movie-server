import time
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, manager, settings
from app.schemas import ConcatRequest, GenerateRequest


class FakeGenerator:
    _pipe = None

    def generate(self, request, target: Path, progress):
        progress(0.5)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake-mp4")
        progress(1.0)


def test_health_and_generation(tmp_path):
    manager.generator = FakeGenerator()
    manager.config.output_dir = tmp_path
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    response = client.post("/api/jobs", json={"prompt": "A crane flies over Tokyo"})
    assert response.status_code == 202
    job_id = response.json()["id"]
    for _ in range(50):
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] == "completed":
            break
        time.sleep(0.01)
    assert result["status"] == "completed"
    assert result["session_number"] >= 1
    assert result["created_at"]
    assert result["generation_seconds"] is not None
    assert (tmp_path / f"{job_id}.mp4").exists()
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert not (tmp_path / f"{job_id}.mp4").exists()
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_validation():
    client = TestClient(app)
    response = client.post("/api/jobs", json={"prompt": "x", "width": 513})
    assert response.status_code == 422

    response = client.post("/api/jobs", json={"mode": "i2v", "prompt": "x"})
    assert response.status_code == 422

    response = client.post("/api/jobs", json={"prompt": "x", "num_frames": 120})
    assert response.status_code == 422

    response = client.post(
        "/api/jobs", json={"prompt": "x", "quality": "high", "width": 1024, "height": 512}
    )
    assert response.status_code == 422

    response = client.post(
        "/api/jobs", json={"prompt": "x", "upscale": False, "decoder": "diffusion"}
    )
    assert response.status_code == 422

    temporal = GenerateRequest(prompt="x", upscale=False, temporal_upscale=True, decoder="diffusion")
    assert temporal.temporal_upscale is True

    direct_1080p = GenerateRequest(prompt="x", width=1920, height=1088, upscale=False)
    assert (direct_1080p.width, direct_1080p.height) == (1920, 1088)
    portrait_1080p = GenerateRequest(prompt="x", width=1088, height=1920, upscale=False)
    assert (portrait_1080p.width, portrait_1080p.height) == (1088, 1920)
    latent_1080p = GenerateRequest(prompt="x", width=960, height=544, upscale=True)
    assert latent_1080p.upscale is True
    pixel_1080p = GenerateRequest(
        prompt="x", width=960, height=544, upscale=True, upscale_method="pixel"
    )
    assert pixel_1080p.upscale_method == "pixel"
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", upscale=False, upscale_method="pixel")
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", mode="t2i", upscale_method="pixel")
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", width=1280, height=704, upscale=True)

    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", loras=[{"id": "style.safetensors"}, {"id": "style.safetensors"}])
    with pytest.raises(ValueError):
        ConcatRequest(job_ids=["a" * 32, "a" * 32])


def test_lora_listing(tmp_path, monkeypatch):
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    (lora_dir / "cinematic.safetensors").write_bytes(b"weights")
    (lora_dir / "ignore.txt").write_text("no")
    monkeypatch.setattr(settings, "lora_dir", lora_dir)
    response = TestClient(app).get("/api/loras")
    assert response.status_code == 200
    assert response.json() == [{
        "id": "cinematic.safetensors", "name": "cinematic", "size": 7,
        "kind": "standard", "model_version": None,
        "reference_downscale_factor": None, "generic_iclora_compatible": False,
    }]


def test_render_options_and_legacy_quality():
    draft = GenerateRequest(prompt="x", quality="draft")
    assert draft.upscale is False

    explicit = GenerateRequest(prompt="x", upscale=False, decoder="vae")
    assert explicit.upscale is False
    assert explicit.quality is None

    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", quality="draft", upscale=True)

    automatic = GenerateRequest(prompt="x", num_frames=None, min_seconds=2, max_seconds=6)
    assert automatic.num_frames is None

    temporal = GenerateRequest(prompt="x", upscale=False, temporal_upscale=True, decoder="diffusion")
    assert temporal.temporal_upscale is True

    retake = GenerateRequest(
        mode="retake",
        prompt="replace the action",
        conditions=[{"asset_id": "a" * 32, "kind": "video"}],
        retake_start=1,
        retake_end=3,
        upscale=True,
    )
    assert retake.upscale is False
    assert retake.decoder == "vae"

    with pytest.raises(ValueError):
        GenerateRequest(
            mode="retake", prompt="x", conditions=[{"asset_id": "a" * 32, "kind": "video"}],
            retake_start=3, retake_end=1,
        )

    extend = GenerateRequest(
        mode="extend", prompt="continue walking",
        conditions=[{"asset_id": "b" * 32, "kind": "video"}],
        extend_direction="start", extend_seconds=4, extend_context_seconds=2,
        upscale=True,
    )
    assert extend.extend_direction == "start"
    assert extend.upscale is False

    a2v = GenerateRequest(prompt="visualize the beat", mode="a2v", audio_asset_id="c" * 32)
    assert a2v.audio_duration is None
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", mode="a2v")

    iclora = GenerateRequest(
        prompt="Generated video: the character walks",
        mode="iclora",
        conditions=[{"asset_id": "d" * 32, "kind": "image", "index": 1}],
        loras=[{"id": "ingredients.safetensors", "strength": 1.4}],
    )
    assert iclora.upscale is False
    assert iclora.decoder == "vae"
    with pytest.raises(ValueError):
        GenerateRequest(
            prompt="x", mode="iclora",
            conditions=[{"asset_id": "d" * 32, "kind": "image", "index": 1}],
        )


def test_sessions_history_and_unconfigured_enhancer(monkeypatch):
    client = TestClient(app)
    session_response = client.post("/api/sessions")
    assert session_response.status_code == 201
    session_number = session_response.json()["session_number"]
    history = client.get("/api/jobs", params={"session_number": session_number})
    assert history.status_code == 200
    assert history.json() == []
    monkeypatch.setattr(settings, "llm_base_url", None)
    enhancer = client.post("/api/prompts/enhance", json={"prompt": "A person walks"})
    assert enhancer.status_code == 503


def test_upload_and_i2v_request(tmp_path):
    settings.input_dir = tmp_path / "inputs"
    settings.input_dir.mkdir()
    manager.generator = FakeGenerator()
    manager.config.output_dir = tmp_path / "outputs"
    client = TestClient(app)

    image_file = BytesIO()
    Image.new("RGB", (64, 64), "red").save(image_file, format="PNG")
    response = client.post("/api/assets", files={"file": ("first.png", image_file.getvalue(), "image/png")})
    assert response.status_code == 201
    asset = response.json()
    assert asset["kind"] == "image"
    assert (settings.input_dir / f'{asset["id"]}.png').is_file()

    response = client.post(
        "/api/jobs",
        json={
            "mode": "i2v",
            "prompt": "The red frame begins to move",
            "conditions": [{"asset_id": asset["id"], "kind": "image", "index": 0, "strength": 1}],
        },
    )
    assert response.status_code == 202
    assert response.json()["request"]["mode"] == "i2v"
    job_id = response.json()["id"]
    for _ in range(50):
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "completed":
            break
        time.sleep(0.01)


def test_rejects_unsupported_upload(tmp_path):
    settings.input_dir = tmp_path
    client = TestClient(app)
    response = client.post("/api/assets", files={"file": ("notes.txt", b"not media", "text/plain")})
    assert response.status_code == 415


def test_admin_unload(tmp_path):
    """Phase 5a: /api/admin/unload releases the generator while idle, 409 while busy."""

    class UnloadableFakeGenerator(FakeGenerator):
        unload_calls = 0

        def unload(self):
            type(self).unload_calls += 1
            return {"freed": ["pipe"], "allocated_gb": 0.0}

    manager.generator = UnloadableFakeGenerator()
    manager.config.output_dir = tmp_path
    client = TestClient(app)

    # idle -> unload succeeds
    response = client.post("/api/admin/unload")
    assert response.status_code == 200
    assert response.json()["result"] == "unloaded"
    assert UnloadableFakeGenerator.unload_calls == 1

    # busy (in-memory running job) -> 409, generator untouched
    from app.jobs import Job
    from app.schemas import GenerateRequest as GR

    fake_job = Job(id="busyjob", session_number=1, request=GR(prompt="x"), status="running")
    manager.jobs["busyjob"] = fake_job
    try:
        response = client.post("/api/admin/unload")
        assert response.status_code == 409
        assert UnloadableFakeGenerator.unload_calls == 1
    finally:
        manager.jobs.pop("busyjob", None)
