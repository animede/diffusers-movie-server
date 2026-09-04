"""Video encoding with an explicit x264 CRF.

`diffusers.utils.encode_video` does not expose encoder options (libx264 defaults to
CRF~23), so this is a clone that sets `crf` / `preset` via PyAV stream options.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def encode_video_crf(
    frames: np.ndarray,
    fps: float,
    audio: torch.Tensor,
    audio_sample_rate: int,
    output_path: Path | str,
    crf: int = 18,
    preset: str = "slower",
    encoder: str = "x264",
    nvenc_preset: str = "p7",
) -> None:
    """Encode frames (F, H, W, 3) — float in [0, 1] or uint8 — to H.264 mp4 with audio.

    Mirrors diffusers.utils.encode_video (yuv420p + aac mux) but with controllable
    x264 quality options.

    encoder="nvenc" switches to h264_nvenc (GPU encode). NVENC has no CRF; we map
    the same numeric value to VBR constant-quality (`cq`) with the slowest NVENC
    preset p7 + tune hq. At cq18/p7 perceptual quality is close to x264 crf18 while
    encoding is an order of magnitude faster (PyAV 16 bundles nvenc; verified on
    this venv/GPU).
    """
    import av
    from diffusers.utils.export_utils import _prepare_audio_stream, _write_audio

    # LTX25_STAGE_DEBUG=1: mp4 encode 内部の時間分解(2026-09-03 調査、既定OFFで挙動不変)
    import os as _os
    import time as _time
    _dbg = _os.getenv("LTX25_STAGE_DEBUG", "0").strip() == "1"
    _t = _time.time()

    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
    if _dbg:
        print(f"[ltx25] STAGE_DEBUG mp4.uint8_convert {_time.time() - _t:.3f}s", flush=True)
        _t = _time.time()

    container = av.open(str(output_path), mode="w")
    codec = "h264_nvenc" if encoder == "nvenc" else "libx264"
    try:
        stream = container.add_stream(codec, rate=int(round(fps)))
    except Exception:
        if codec == "libx264":
            raise
        # NVENC が使えない環境(PyAV ビルド差・セッション上限等)は x264 へ退避
        print("[encoding] h264_nvenc unavailable; falling back to libx264", flush=True)
        encoder = "x264"
        stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"
    if encoder == "nvenc":
        stream.options = {
            "rc": "vbr",
            "cq": str(int(crf)),
            "preset": nvenc_preset,
            "tune": "hq",
            "b:v": "0",
        }
    else:
        stream.options = {"crf": str(int(crf)), "preset": preset}
    audio_stream = _prepare_audio_stream(container, audio_sample_rate)
    if _dbg:
        print(f"[ltx25] STAGE_DEBUG mp4.container_setup {_time.time() - _t:.3f}s", flush=True)
        _t = _time.time()
    for frame_array in frames:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    if _dbg:
        print(f"[ltx25] STAGE_DEBUG mp4.video_encode {_time.time() - _t:.3f}s", flush=True)
        _t = _time.time()
    _write_audio(container, audio_stream, audio, audio_sample_rate, av)
    container.close()
    if _dbg:
        print(f"[ltx25] STAGE_DEBUG mp4.audio_mux_close {_time.time() - _t:.3f}s", flush=True)
