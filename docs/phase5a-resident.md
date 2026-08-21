# Phase 5a: in-process unload による resident 切替(2026-08-21)

切替のたびにプロセスを kill→再起動していた従来方式(9〜64s)に対し、**プロセスを
残したまま VRAM だけを解放/復帰する resident 戦略**を追加した。ホスト RAM 94GB の
本機では両バックエンドの常駐が許容されるため、往復切替が大幅に速くなる。

## 設計

### バックエンド側(追加のみ、既存生成経路は無変更)

| バックエンド | エンドポイント | 内容 |
|---|---|---|
| ltx25 | `POST /api/admin/unload` | `LTXGenerator.unload()`: `_pipe` / `_upsample_pipe` / `_temporal_upsample_pipe` / `_diffusion_decode_pipe` の参照を落とし `gc.collect()` + `torch.cuda.empty_cache()`。job が queued/running なら **409**。以後の生成は `load()` の遅延ロードで自動復帰(`/api/health` の `loaded` も false に戻る) |
| h3 | `POST /api/admin/unload` | `runner.unload_all()`(既存関数。transformer×2 / TE / VAE ペアを解放、各 `_free_*` が gc + empty_cache 済み)。`_generation_lock` 非ブロッキング取得で busy 中は **409**。解放後も `generate()` 系の冪等な `_ensure_*` で自動再ロードされる(実機確認済み) |
| h3 | `POST /api/admin/reload` | `runner.preload_all()` を再実行して定常常駐状態(96gb ならプリロード済み)へ戻す。resident 復帰時に gateway が呼ぶ。busy 中は 409 |

### gateway(procman.py)

- `ManagedProcess` を「アクティブ1つ」から **`_procs: dict[backend→process]` + `_active`**
  へ一般化。「VRAM を持てるのは1バックエンドのみ」の排他原則は `_active` が担う。
- `POST /api/v1/backend/load` に `strategy: "process"(既定・従来どおり) | "resident"`。
  - resident 切替: 旧アクティブへ `/api/admin/unload` → **nvidia-smi の per-process
    実測で VRAM 解放を確認**(閾値 2048MB、in-process unload 後も CUDA コンテキスト
    ~0.7GB が残るのが正常)→ 新バックエンドの既存プロセスを再有効化(h3 は
    `admin/reload`、ltx25 は遅延ロードに任せて何もしない)。
  - **env(プリセット/overrides)が前回起動時と異なる場合は resident 不可 → 自動で
    プロセス再起動へフォールバック**(`note` フィールドで通知)。procman は
    バックエンドごとの起動時 env セット(`env_extra`)を保持して比較する。
  - unload API 失敗・VRAM 未解放時も排他原則を守るためプロセス停止へフォールバック。
- `POST /api/v1/backend/unload` に `strategy`: `process`(既定)は**管理下の全プロセス
  停止**(parked 含む、完全クリーン)、`resident` はアクティブの VRAM 解放のみ。
- `GET /api/v1/status` に `backends`: 各バックエンドの `process_alive` /
  `weights_loaded` / `vram_mb`(nvidia-smi per-process)の2軸を追加。
- adopt: resident 運用では複数プロセス生存が正常のため全て adopt し、アクティブは
  各バックエンドの自己申告(h3 runner status / ltx25 health.loaded)から推定。
  自己申告なしで生存1つのみなら従来どおりアクティブ扱い(gateway 再起動退行の回避)。
- GUI(管理タブ): 切替戦略セレクト(既定 resident 推奨)と「プロセス/重み」2軸表示を追加。

## 実測(GPU:0 RTX PRO 6000 Blackwell 96GB、ベースライン ~2.85GB、2026-08-21)

### 切替時間比較(resident vs process)

| 切替 | process 戦略(Phase 4 実測) | resident 戦略(本実測) |
|---|---|---|
| h3 → ltx25(nf4) | 9.1s | **0.5〜1.3s**(h3 96gb の 83.8GB 解放込み) |
| ltx25 → h3(96gb、プリロード込み) | 64.1s | **51.7〜54.8s**(admin/reload = preload_all のみ。プロセス起動・import 分 ~10-12s を短縮) |
| ltx25 → h3(48gb-lowvram) | ~12.7s(本実測) | 10.8s(初回はプロセス無しのため通常起動) |
| ltx25 nf4 → fp8(env 変更、resident 指定) | - | 自動フォールバックで再起動(pid 変化を確認) |
| h3 48gb → 96gb(env 変更、resident 指定) | - | 自動フォールバックで再起動(65.7s、note 付き) |

- resident の恩恵が最大なのは **ltx25 側への切替**(h3 96gb の解放が in-process
  unload_all で数秒、ltx25 は遅延ロードのため復帰コストゼロ)。
- h3 96gb への「戻り」は preload_all(モデルロード ~50s)が支配的で、短縮は
  プロセス起動+import 分(~10-12s)に留まる。それでも h3 のモデルはページキャッシュ
  経由で温かく、プロセス kill を挟まない分安定して速い。

### VRAM 検証

| 状態 | per-process VRAM(nvidia-smi) |
|---|---|
| ltx25 nf4 生成後(model offload 構成) | ~1.5GB → **unload 後 684MB**(CUDA コンテキストのみ) |
| h3 96gb 常駐 | 83.8GB → **unload 後 686MB** |
| resident 往復2周後の全体 unload(process) | GPU:0 **2.86GB(ベースライン復帰)**、GPU:1 19MiB |

- 生成の実測: ltx25 t2i 60.1s/peak 17.2GB(初回)、resident 往復後 54.7s(遅延再
  ロード込み、退行なし)。h3 96gb t2i 54.2s/peak 87.7GB(Phase 4 と一致)。
  h3 48gb-lowvram t2i 65.8s/peak 35.0GB。
- busy 中の resident 切替 / resident unload はいずれも **409**、実行中ジョブは
  無傷で完走することを確認。
- ltx25 pytest(tests/test_api.py)8件合格(admin/unload の新テスト1件を追加)。

## 既知の制限・注意

1. **parked プロセスへのパススルーは従来どおり 502**(アクティブのみ転送)。parked
   バックエンドの内部ポート(8631/8632)を直接叩いて生成すると排他原則を迂回できて
   しまう(gateway 経由の運用が前提)。
2. resident の VRAM 解放確認閾値は 2048MB(`UNLOAD_VRAM_THRESHOLD_MB`)。unload 後の
   CUDA コンテキスト残量は実測 ~0.7GB だが、環境によっては閾値調整が必要になりうる。
3. h3 の resident 復帰(admin/reload)は起動時 env の構成のまま `preload_all()` を
   再実行するだけで、構成は変えられない(構成変更は自動フォールバックの再起動、
   または h3 自身の `/api/settings/apply`)。
4. adopt 時に複数プロセスが生存し、かつ複数が重みロード済みと自己申告した場合は
   先頭のみアクティブ扱い(排他原則違反状態のためログで手動確認を促す)。
