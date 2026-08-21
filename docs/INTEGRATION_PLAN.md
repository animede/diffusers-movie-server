# diffusers-movie-server 統合計画(2026-08-20)

minimax-h3(port 8611)と diffusers-ltx2_5(port 8000)の統合。

## 前提(1プロセス統合が不可能な理由)
| | minimax-h3 | diffusers-ltx2_5 |
|---|---|---|
| torch | 2.9.0+cu128(torchao 0.17 制約) | 2.11.0+cu130(NATTEN na3d 必須) |
| diffusers | PR #14355 コミット固定(f37ab93) | git 開発版 |
| transformers | 5.14.1 | git 開発版 |
| API様式 | 同期 + /api/status ポーリング | 非同期ジョブ(POST /api/jobs) |

→ プロセス分離必須。「ゲートウェイ + バックエンド2プロセス(各専用venv)」構成。
量子化・VRAM構成は両者とも起動時環境変数で決まるため、切替=env付きプロセス再起動で実現する。

## アーキテクチャ
- gateway/(port 8630、軽量venv): 統一API `/api/v1/*`、パススルー `/h3/*` `/ltx25/*`、
  プロセスマネージャ(同時アクティブ1バックエンド、切替は旧stop→新start の直列)
- backends/minimax-h3(内部 port 8631)/ backends/ltx2_5(内部 port 8632)
- GUI: タブ切替シェル(iframe で既存SPAをそのまま表示)+ バックエンド管理タブ

## 統一API(Phase 2)
- GET /api/v1/backends, /api/v1/status
- POST /api/v1/backend/load {backend, preset, overrides} / /backend/unload
- POST /api/v1/generate {backend, mode, params, asset_ids} → 統一ジョブ
  (h3 の同期APIはゲートウェイ側ワーカーでジョブ化)
- GET /api/v1/jobs/{id}, POST /api/v1/assets, /api/v1/outputs, /api/v1/prompt/enhance
- 統一mode: t2v/i2v/flf2v/ref2v/t2i/ref2i(+ ltx25固有: a2v/extend/retake/iclora/refine_image)
- プリセット: h3 = 96gb/96gb-resident/80gb-int8/48gb-lowvram/32gb-group/16gb-proj(+turbo等)、
  ltx25 = nf4-24gb/fp8-48gb/bf16-96gb(+decoder/offload)

## フェーズ
- Phase 0: スキャフォールド・コード移設・venv(当面は旧venvへのsymlink)・単独スモーク
- Phase 1: ゲートウェイ基盤(プロセス管理・排他・パススルー・load/unload/status)
- Phase 2: 統一API(ジョブモデル・モード変換・アセット)
- Phase 3: GUI タブシェル
- Phase 4: 検収・ドキュメント・移行メモ
- Phase 5(任意): in-process unload、統一ギャラリー、venv完全独立化

## Phase 0 の方針メモ
- venv は当面 **旧ディレクトリへの symlink**(h3 のピン留めdiffusers・ltx25 の
  shadowインストール済みtorch2.11 を壊さない最速・最安全経路)。完全独立の再構築は
  Phase 5 送り(手順は各 backends/*/VENV_REBUILD.md に記録)。
- 大容量物も symlink 共有: h3 models/(prequant 36GB)、ltx25 の
  LTX-2.5-Diffusers-bnb-4bit(27GB)・loras(1.6GB)。
- 旧フォルダ(/home/animede/minimax-h3、/home/animede/diffusers-ltx2_5)は検収完了まで無変更で残す。
- ポート: gateway 8630 / h3 8631 / ltx25 8632(既存 8600/8601/8602/8610/8611/8620 と非衝突)

## 完了記録(Phase 0〜4)

| Phase | 内容 | 完了日 | コミット |
|---|---|---|---|
| 0 | スキャフォールド初期化 | 2026-08-20 | 4550b19 |
| 0 | minimax-h3 / ltx2_5 バックエンド移設(symlink共有)・両スモーク合格 | 2026-08-20 | bd59d02 |
| 1 | ゲートウェイ基盤(プロセス管理・排他切替・パススルー) | 2026-08-21 | 912337e |
| 2 | 統一API(/api/v1/generate・統一ジョブモデル) | 2026-08-21 | 0125d6b |
| 3 | タブ切替GUI(headless Chrome 実機検証) | 2026-08-21 | 4179178 |
| 4 | 総合検収(96gb 実測含む)・README 本格化・MIGRATION.md | 2026-08-21 | (本コミット) |

Phase 4 検収結果: docs/phase4-acceptance.md(h3 96gb t2v peak 91.93GB、切替 9.1〜64.1s、
全回帰合格、実バグなし・コード無変更)。

| 5a | in-process unload による resident 切替(strategy パラメータ) | 2026-08-21 | ea14af4 |
| 5b | 統一ジョブの SQLite 永続化・統一ギャラリータブ | 2026-08-21 | 637b97f |
| 5c | venv 完全独立化・データ所有権の移転(旧位置は逆向き symlink) | 2026-08-21 | (本コミット) |

Phase 5a 実測: docs/phase5a-resident.md(h3→ltx25 切替 9.1s→0.5〜1.3s、
ltx25→h3 96gb 64.1s→51.7s、VRAM リークなし)。
Phase 5b: docs/phase5b-persistence-gallery.md(gateway 再起動でジョブ履歴復元、
running 同期、ギャラリータブ)。
Phase 5c: docs/phase5c-independence.md(新venvは新旧出力 MD5 完全一致(h3・ltx25 とも)。
models 36GB / 量子化済み 27GB / loras の実体は本リポジトリへ移転済み、旧ディレクトリは
コード+旧venvのみで symlink 経由で引き続き起動可)。**全フェーズ完了。**

| 6 | GPU割当指定(48gb-dual / gpus パラメータ / GUI) | 2026-08-21 | (本コミット) |

Phase 6: docs/phase6-gpu-assignment.md(h3 2GPU分担 t2i 12.2s、ltx25@GPU1 単独動作、
resident×dual はプロセス停止フォールバックの既知制限あり)。
