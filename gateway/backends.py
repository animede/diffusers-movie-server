"""バックエンド定義(宣言的カタログ)。

各バックエンドの起動方法・ポート・ヘルス/busy 判定・env プリセットを1か所に集約する。
プリセットの env セットは backends/minimax-h3/README.md「VRAM級別クイックスタート」
「VRAM対応表と主な環境変数(早見表)」、および backends/ltx2_5/app/config.py の
実測構成(docs/phase0-*.md)に基づく。

注意:
- H3_TE_DEVICE=cuda:1 を含む README の起動例は GPU:1 を使うため、プリセットからは
  既定で外してある(必要なら overrides={"H3_TE_DEVICE": "cuda:1"} で明示指定する)。
- overrides はホワイトリスト方式(下記 override_allowed())。任意 env 注入は拒否する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Preset:
    name: str
    env: dict[str, str]
    description: str
    vram_hint: str  # 想定VRAM(実測ベースの目安)


@dataclass(frozen=True)
class BackendDef:
    name: str
    dir: Path                 # run.sh のあるディレクトリ
    run_script: str           # 起動コマンド(dir 相対)
    port: int
    health_path: str          # ヘルスチェック用 GET パス
    health_timeout_s: float   # 起動→ヘルスOKまでの待ちタイムアウト
    presets: dict[str, Preset]
    default_preset: str
    description: str = ""
    toggles: dict[str, dict[str, str]] = field(default_factory=dict)  # 例: turbo

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


# ---------------------------------------------------------------------------
# minimax-h3 (port 8631)
# ---------------------------------------------------------------------------

H3_PRESETS = {
    "96gb": Preset(
        name="96gb",
        env={},
        description="既定(env なし)。全モデル常駐、起動時プリロードで常駐 87.3GB。96GB 専有前提。"
                    "**ref2va 主体の用途では 96gb-int8 の方が 1.8倍速い(168.4s → 92.7s)**: "
                    "bf16 では transformer_ref(66GB)と TE-nf4(21GB)が同時に載らず、"
                    "リクエストごとに transformer_ref の再ロード(実測 34.7秒)が入るため。",
        vram_hint="常駐 87.3GB / t2va peak ~92GB",
    ),
    "96gb-int8": Preset(
        name="96gb-int8",
        # 2026-08-24 実測(RTX PRO 6000 96GB、768×448・8秒・turbo・seed 777・ref2va、
        # 定常状態 = transformer_ref ロード済みの2本目):
        #   96gb (bf16) : 合計 168.4s / denoise 20.5s / decode 5.4s / 固定費 142.6s / peak 87.9GB
        #   96gb-int8   : 合計  92.7s / denoise 20.9s / decode 5.5s / 固定費  66.3s / peak 73.8GB
        # denoise/decode はほぼ同じで、差は丸ごと固定費(142.6s → 66.3s)。bf16 は
        # transformer_ref(66GB)+ TE-nf4(21GB)が 96GB に同時に載らず、リクエストごとに
        # transformer_ref を 34.7秒かけて載せ替えていた。int8(34GB)なら
        # 34 + 21 + VAE 11 = 66GB で全部常駐でき、この載せ替えが消える。
        # 残る 66.3s の内訳: 参照エンコード ~55s(README も ref2va のボトルネックと記載)
        # + VAE の GPU⇔CPU 往復 11.6s(TE_QUANT=bnb-4bit のときだけ発生する設計)。
        # turbo は load 時トグル・リクエスト単位のどちらでも指定できる
        # (2026-08-24 に _validate_combination() の無条件拒否を comfy 形式限定へ修正した。
        #  既定の lightx2v 版は diffusers ネイティブで int8 と併用できる)。
        #
        # **int8 の代償(ref2va 以外の用途では要検討)**:
        #  - denoise 単体は bf16 の方が 5〜14% 速い(README: t2i 2.07s vs 2.40s /
        #    t2va 14.05s vs 14.81s)。ref2va で int8 が勝つのは載せ替えが消えるからで、
        #    載せ替えが元々起きない用途では逆転しうる。
        #  - **同一 seed でも bf16 とは出力が変わる**(README: PSNR 19dB の軌道分岐)。
        #    bf16 とのビット再現が要る対照実験では 96gb を使うこと。
        env={"H3_TRANSFORMER_QUANT": "int8"},
        description="96GB機の ref2va 最適(transformer int8)。96gb(bf16)比 1.8倍速の "
                    "合計 92.7s(denoise 20.9s / decode 5.5s / 固定費 66.3s)。"
                    "int8(34GB)なら transformer_ref と TE-nf4 が同時常駐でき、bf16 で "
                    "毎リクエスト発生していた transformer_ref の載せ替え(34.7s)が消える。"
                    "代償: denoise 単体は bf16 が 5〜14% 速く、同一 seed でも bf16 とは "
                    "出力が変わる(PSNR 19dB の軌道分岐)。ref2va 主体でないなら 96gb を。",
        vram_hint="ref2va peak 73.8GB(768×448・8秒。96gb bf16 は 87.9GB)",
    ),
    "48gb-lowvram": Preset(
        name="48gb-lowvram",
        # README「48GB級(推奨: 高速化フル)」から H3_TE_DEVICE=cuda:1 を除いたもの
        # (GPU:1 は使わない方針。必要なら overrides で明示)。
        # 注意: H3_KEEP_TRANSFORMER=1 は runner.py のガードにより
        # (H3_TE_DEVICE または H3_TE_PROJ)+ H3_VIDEO_VAE_FP16=1 が必須のため、
        # TE_DEVICE を外したこのプリセットには含められない(Phase 1 実機で確認)。
        # 高速化フルにしたい場合は overrides で H3_TE_DEVICE=cuda:1(または
        # H3_TE_PROJ)+ H3_KEEP_TRANSFORMER=1 を明示すること。
        env={
            "H3_LOWVRAM": "1",
            "H3_TE_PRUNE": "1",
            "H3_VIDEO_VAE_FP16": "1",
        },
        description="48GB級のフェーズ循環 + TE prune + fp16 デコード。README 推奨構成から "
                    "H3_TE_DEVICE=cuda:1 と(それが前提条件の)H3_KEEP_TRANSFORMER=1 を除外"
                    "(overrides で追加可)。",
        vram_hint="t2va peak ~38.9GB",
    ),
    "48gb-dual": Preset(
        name="48gb-dual",
        # README「48GB級(推奨: 高速化フル)」そのまま(2GPU分担)。
        # GPU0 = transformer/denoise、GPU1 = text_encoder(H3_TE_DEVICE=cuda:1)。
        env={
            "H3_LOWVRAM": "1",
            "H3_TE_PRUNE": "1",
            "H3_TE_DEVICE": "cuda:1",
            "H3_VIDEO_VAE_FP16": "1",
            "H3_KEEP_TRANSFORMER": "1",
        },
        description="2GPU分担(GPU0=transformer、GPU1=text_encoder)+ transformer常駐。"
                    "README 48GB級推奨構成。過去実測 t2i ~10s / t2va ~44s。"
                    "注意: 参照系(ref2va)は TE 側 GPU に約24GB 必要(24GBカードは境界)。",
        vram_hint="GPU0 ~35GB + GPU1 ~数GB〜20GB(TE)",
    ),
    "32gb-group": Preset(
        name="32gb-group",
        env={"H3_LOWVRAM": "group", "H3_TE_PRUNE": "1"},
        description="32GB級以下の block offload(H3_LOWVRAM=group)+ TE prune。",
        vram_hint="t2va peak ~28.7GB(prune 併用で ~17.7GB)",
    ),
    "16gb-proj": Preset(
        name="16gb-proj",
        env={
            "H3_LOWVRAM": "group",
            "H3_TE_PROJ": "NicoLab28/ClipProj-MiniMax-H3",
            "H3_VIDEO_VAE_FP16": "1",
            "H3_ATTN_BACKEND": "default",
        },
        description="16GB 単体向け(投影TE + fp16 デコード + SDPA)。README の 16GB 起動例どおり "
                    "H3_ATTN_BACKEND=default を含む(sm_120 で sage を使うなら overrides で上書き)。",
        vram_hint="peak ~11.4GB",
    ),
}

H3 = BackendDef(
    name="h3",
    dir=REPO_ROOT / "backends" / "minimax-h3",
    run_script="run.sh",
    port=8631,
    health_path="/api/status",
    health_timeout_s=600.0,  # 96gb 既定は起動時プリロードで数分かかる
    presets=H3_PRESETS,
    default_preset="96gb",
    description="MiniMax-H3 動画+音声生成(同期API、/api/status ポーリング)",
    toggles={"turbo": {"H3_TURBO_LORA": "1"}},
)

# ---------------------------------------------------------------------------
# ltx2_5 (port 8632)
# ---------------------------------------------------------------------------

LTX25_PRESETS = {
    "nf4": Preset(
        name="nf4",
        env={},
        description="既定(bnb 4bit transformer、遅延ロード)。",
        vram_hint="peak ~17.2GB",
    ),
    "fp8": Preset(
        name="fp8",
        # ⚠ OFFLOAD_MODE=none との併用は 48GB では不可(2026-08-22 実測、2回検証)。
        # ltx25 単体で 42〜45GB まで伸び、**512×288・3秒という最小構成でも OOM** する。
        # 同居プロセスのキャッシュを解放して余白を 2.4GB 増やしても再現したので、
        # 「他プロセスに圧迫されていたから」ではなく fp8 全常駐そのものが 48GB 級に
        # 対して大きすぎる。vram_hint の「常駐 ~18GB」は model オフロード時の値。
        # 速度が要るなら nf4-fast(全常駐でも ~28GB)を使う。
        env={"LTX25_TRANSFORMER_PRECISION": "fp8"},
        description="fp8 layerwise casting(bf16 相当品質)。48GB級向け(model オフロード)。",
        vram_hint="常駐 ~18GB / peak ~29GB(オフロードあり。none 併用は48GBで OOM)",
    ),
    "nf4-fast": Preset(
        name="nf4-fast",
        # nf4 は peak ~17.2GB と軽いので、48GB なら全常駐にできる。
        # 既定の model オフロードは毎ステップ transformer が CPU⇄GPU を往復して
        # 大幅に遅くなるため、速度優先ならこちら。
        env={"OFFLOAD_MODE": "none"},
        description="bnb 4bit transformer + オフロードなし。48GB級で速度優先。",
        vram_hint="peak ~17.2GB 相当(全常駐)",
    ),
    "bf16": Preset(
        name="bf16",
        env={"LTX25_TRANSFORMER_PRECISION": "bf16", "OFFLOAD_MODE": "none"},
        description="bf16 リリース重み全常駐(オフロードなし)。96GB級向け。",
        vram_hint="transformer ~38GB + TE/VAE",
    ),
    "nvfp4-fast": Preset(
        name="nvfp4-fast",
        # 公式 NVFP4 蒸留 transformer(Blackwell ネイティブ FP4)。sm_120 専用。
        # FP4 テンサーコア GEMM(torch._scaled_mm)で denoise の Linear が
        # bf16 比 ~1.8倍(活性化量子化コスト込み、GEMM 素は ~3.4倍)。
        # NVENC p4(2026-09-04): このプリセットはリアルタイム用途専用のため
        # mp4 encode も速度側へ倒す(低画素では画質差軽微)。プリセットに
        # 焼き込むことで「クライアントが override を付け忘れると p7 に戻る」
        # 状態を防ぐ(MV が使う nf4-fast 等は従来どおり既定 p7 = 最終出力品質)。
        env={"LTX25_TRANSFORMER_PRECISION": "nvfp4", "OFFLOAD_MODE": "none",
             "LTX25_NVENC_PRESET": "p4"},
        description="公式 NVFP4 蒸留 transformer + FP4 GEMM 全常駐 + NVENC p4。sm_120(Blackwell)専用・速度最優先。",
        vram_hint="transformer ~19GB + TE/VAE(全常駐)",
    ),
}

# app/config.py の Settings フィールド由来のキーのみ許可(pydantic-settings は
# フィールド名の大文字化を env 名として解決する)。
LTX25_ALLOWED_KEYS = {
    "MODEL_ID", "MODEL_REVISION", "QUANTIZED_MODEL_DIR", "HF_TOKEN",
    "OFFLOAD_MODE", "OUTPUT_DIR", "INPUT_DIR", "LORA_DIR",
    "MAX_UPLOAD_SIZE_MB", "MAX_QUEUE_SIZE", "HISTORY_DB",
    "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_TIMEOUT_SECONDS",
    "LTX25_PORT", "LTX25_TRANSFORMER_PRECISION", "LTX25_NVFP4_CKPT",
    "LTX25_VIDEO_ENCODER", "LTX25_VIDEO_CRF", "LTX25_DECODER",
}

LTX25 = BackendDef(
    name="ltx25",
    dir=REPO_ROOT / "backends" / "ltx2_5",
    run_script="run.sh",
    port=8632,
    health_path="/api/health",
    health_timeout_s=60.0,  # 遅延ロードのため起動は軽い
    presets=LTX25_PRESETS,
    default_preset="nf4",
    description="LTX-2.5 動画+音声生成(非同期ジョブ API、POST /api/jobs)",
)

BACKENDS: dict[str, BackendDef] = {"h3": H3, "ltx25": LTX25}


# ---------------------------------------------------------------------------
# overrides / 組み合わせバリデーション
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """400 に変換されるバリデーションエラー。"""


def override_allowed(backend: str, key: str) -> bool:
    if backend == "h3":
        return key.startswith("H3_")
    if backend == "ltx25":
        return key.startswith("LTX25_") or key in LTX25_ALLOWED_KEYS
    return False


_GPUS_RE = re.compile(r"^[0-9](,[0-9])*$")


def resolve_env(backend_name: str, preset_name: str | None,
                overrides: dict[str, str] | None,
                toggles: dict[str, bool] | None = None,
                gpus: str | None = None) -> tuple[dict[str, str], str]:
    """プリセット + トグル + overrides(+ gpus)を合成した env dict を返す。

    gpus: 実行GPUの選択(例 "0" / "1" / "0,1")。指定時は子プロセスの
    CUDA_VISIBLE_DEVICES に設定される。CUDA は可視GPUを 0 から再番号付けする
    ため、例えば gpus="1" のとき cuda:0 = 物理GPU1 になる点に注意。
    戻り値: (env, 実際に使ったプリセット名)。不正入力は ValidationError。
    """
    backend = BACKENDS.get(backend_name)
    if backend is None:
        raise ValidationError(
            f"未知のバックエンドです: {backend_name!r}(有効: {sorted(BACKENDS)})")

    preset_name = preset_name or backend.default_preset
    preset = backend.presets.get(preset_name)
    if preset is None:
        raise ValidationError(
            f"未知のプリセットです: {preset_name!r}(backend={backend_name}, "
            f"有効: {sorted(backend.presets)})")

    env = dict(preset.env)

    for toggle_name, enabled in (toggles or {}).items():
        if toggle_name not in backend.toggles:
            raise ValidationError(
                f"未知のトグルです: {toggle_name!r}(backend={backend_name}, "
                f"有効: {sorted(backend.toggles)})")
        if enabled:
            env.update(backend.toggles[toggle_name])

    for key, value in (overrides or {}).items():
        if not isinstance(value, str):
            value = str(value)
        if not override_allowed(backend_name, key):
            raise ValidationError(
                f"overrides で許可されていないキーです: {key!r}(backend={backend_name})")
        env[key] = value

    if gpus is not None and gpus != "":
        if not _GPUS_RE.match(gpus):
            raise ValidationError(
                f"gpus の形式が不正です: {gpus!r}(例: \"0\" / \"1\" / \"0,1\")")
        indices = gpus.split(",")
        if len(indices) != len(set(indices)):
            raise ValidationError(f"gpus に重複があります: {gpus!r}")
        env["CUDA_VISIBLE_DEVICES"] = gpus

    _validate_combination(backend_name, env)
    return env, preset_name


# turbo LoRA のうち **comfy 形式(融合QKV、Ostris 版)** だけが int8 と非互換。
# 機序: `apply_turbo_lora()` の `fuse_projections()` が to_q/to_k/to_v を `torch.cat` で
# 融合するが、torchao の Int8Tensor に `aten.cat` カーネルが無い。
# **既定の lightx2v 版(diffusers ネイティブ)は融合しない**ので int8 でも動く。
# バックエンド側の2つのガード(core/runner.py:946、core/settings.py の
# validate_instant_settings)はどちらも `turbo_lora_expected_format() == "comfy"` 条件付き。
# gateway だけが無条件に拒否しており実態と食い違っていた(2026-08-24 修正)。
# 実証: preset=96gb + H3_TRANSFORMER_QUANT=int8 + turbo で 92.7秒の生成に成功
# (transformer_quant="int8" / turbo_lora=true / total_elapsed_s=92.7 をログで確認)。
_H3_TURBO_COMFY_REPOS = ("larryvrh/MiniMax-H3-Turbo-Lora",)
_H3_TURBO_DEFAULT_REPO = "lightx2v/Minimax-h3-Turbo"  # core/runner.py:879 の既定と合わせる


def _h3_turbo_is_comfy_format(env: dict[str, str]) -> bool:
    """この env の turbo LoRA が comfy 形式(融合QKV)か。バックエンドの判定のミラー。"""
    repo = (env.get("H3_TURBO_LORA_REPO") or _H3_TURBO_DEFAULT_REPO).strip()
    return repo in _H3_TURBO_COMFY_REPOS


def _validate_combination(backend_name: str, env: dict[str, str]) -> None:
    if backend_name == "h3":
        turbo = env.get("H3_TURBO_LORA") == "1"
        if turbo and env.get("H3_TRANSFORMER_QUANT") == "int8" and _h3_turbo_is_comfy_format(env):
            raise ValidationError(
                "turbo LoRA(comfy 形式 = 融合QKV)と H3_TRANSFORMER_QUANT=int8 は"
                "併用できません(fuse_projections() の torch.cat が torchao Int8Tensor に"
                "対応していない)。既定の lightx2v 版(diffusers ネイティブ)なら int8 でも"
                f"動きます — H3_TURBO_LORA_REPO を外すか {_H3_TURBO_DEFAULT_REPO!r} にしてください")
        if turbo and env.get("H3_LOWVRAM") == "group":
            raise ValidationError(
                "turbo(H3_TURBO_LORA=1)と H3_LOWVRAM=group は併用できません"
                "(README 記載の既知の非互換)")
        if env.get("H3_KEEP_TRANSFORMER") == "1":
            # core/runner.py 側の起動時ガードを先取りして 400 にする(実機確認済み):
            # H3_KEEP_TRANSFORMER=1 は LOWVRAM != group かつ(TE_DEVICE または
            # TE_PROJ)かつ VIDEO_VAE_FP16=1 が必須。
            problems = []
            if env.get("H3_LOWVRAM") == "group":
                problems.append("H3_LOWVRAM=group と併用不可")
            if not env.get("H3_TE_DEVICE") and not env.get("H3_TE_PROJ"):
                problems.append("H3_TE_DEVICE か H3_TE_PROJ の指定が必要")
            if env.get("H3_VIDEO_VAE_FP16") != "1":
                problems.append("H3_VIDEO_VAE_FP16=1 が必要")
            if problems:
                raise ValidationError(
                    "H3_KEEP_TRANSFORMER=1 の前提条件を満たしていません: "
                    + " / ".join(problems))
        # gpus(CUDA_VISIBLE_DEVICES)指定時、H3_TE_DEVICE=cuda:N の N は
        # 「可視GPU内のインデックス」なので可視枚数の範囲内でなければならない
        # (例: gpus="1" は1枚可視 → cuda:1 は存在せず起動時に落ちる)。
        visible = env.get("CUDA_VISIBLE_DEVICES")
        te_device = env.get("H3_TE_DEVICE", "")
        m = re.match(r"^cuda:(\d+)$", te_device)
        if visible is not None and m is not None:
            if int(m.group(1)) >= len(visible.split(",")):
                raise ValidationError(
                    f"H3_TE_DEVICE={te_device} は gpus={visible!r}(可視 "
                    f"{len(visible.split(','))}枚)の範囲外です。CUDA は可視GPUを "
                    "0 から再番号付けするため、例えば 2GPU分担なら gpus=\"0,1\" "
                    "(または gpus 省略)にしてください")
    if backend_name == "ltx25":
        precision = env.get("LTX25_TRANSFORMER_PRECISION")
        if precision is not None and precision not in ("nf4", "fp8", "bf16", "nvfp4"):
            raise ValidationError(
                f"LTX25_TRANSFORMER_PRECISION の値が不正です: {precision!r}"
                "(有効: nf4 / fp8 / bf16 / nvfp4)")


def catalog() -> list[dict]:
    """GET /api/v1/backends 用のカタログ。"""
    items = []
    for backend in BACKENDS.values():
        items.append({
            "name": backend.name,
            "description": backend.description,
            "port": backend.port,
            "health_path": backend.health_path,
            "default_preset": backend.default_preset,
            "presets": [
                {
                    "name": p.name,
                    "description": p.description,
                    "vram_hint": p.vram_hint,
                    "env": p.env,
                }
                for p in backend.presets.values()
            ],
            "toggles": {k: v for k, v in backend.toggles.items()},
            "overrides_policy": (
                "H3_ プレフィックスのみ" if backend.name == "h3"
                else "LTX25_* / OFFLOAD_MODE / OUTPUT_DIR 等 config.py 由来キーのみ"
            ),
        })
    return items
