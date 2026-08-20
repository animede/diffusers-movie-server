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
) -> None:
    """Encode frames (F, H, W, 3) — float in [0, 1] or uint8 — to H.264 mp4 with audio.

    Mirrors diffusers.utils.encode_video (yuv420p + aac mux) but with controllable
    x264 quality options.
    """
    import av
    from diffusers.utils.export_utils import _prepare_audio_stream, _write_audio

    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)

    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(int(crf)), "preset": preset}
    audio_stream = _prepare_audio_stream(container, audio_sample_rate)
    for frame_array in frames:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    _write_audio(container, audio_stream, audio, audio_sample_rate, av)
    container.close()
