#!/usr/bin/env python3
"""動画のモーション停止→ジャンプ(微ループ/巻き戻り感)検出器。

長尺単発生成の品質 QC 用(2026-09-10 の調査で作成・実運用検証済み)。
検出ロジック: フレーム間差分(グレースケール縮小)の5フレーム移動平均が
中央値×0.55 を下回る「停滞帯」の直後 4 フレーム以内に中央値×1.7 超の
「ジャンプ」が来る型を報告する。処理は ~0.5s/本。

背景(重要な実測知見):
- fps=16 指定は 4.0 秒(64f)周期のモーション揺らぎが構造的に出る(RoPE 時間座標が
  学習分布(24/25fps)外のため)。20/24fps では消える → リアルタイムは 20fps 推奨。
- 20/24fps でも長尺(25s 超)では seed 依存の停止イベントが散発しうる。本検出器を
  生成後 QC にして検出時は別 seed でリトライする運用が有効。

使い方: python detect_motion_stall.py <video.mp4> [fps]
戻り値: イベントゼロなら exit 0、検出ありなら exit 1(リトライ判定に使える)
"""
import subprocess
import sys

import numpy as np


def detect(path: str, fps: float = 16.0, W: int = 176, H: int = 104):
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", path,
         "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True)
    a = np.frombuffer(p.stdout, dtype=np.uint8)
    n = len(a) // (W * H)
    a = a[: n * W * H].reshape(n, H, W).astype(np.float32)
    d = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2))
    med = float(np.median(d))
    sm = np.convolve(d, np.ones(5) / 5, mode="same")
    events = []
    i = 0
    while i < len(sm):
        if sm[i] < med * 0.55:
            j = i
            while j < len(sm) and sm[j] < med * 0.9:
                j += 1
            if j < len(sm) and max(d[j:j + 4], default=0) > med * 1.7:
                events.append((i / fps, j / fps))
            i = j + 1
        else:
            i += 1
    return events


if __name__ == "__main__":
    path = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
    ev = detect(path, fps)
    for s, e in ev:
        print(f"stall-jump: t={s:.2f}-{e:.2f}s")
    print(f"{len(ev)} events")
    sys.exit(1 if ev else 0)
