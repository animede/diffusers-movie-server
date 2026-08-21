# Phase 6: GPU 割当指定(48gb-dual プリセット / gpus パラメータ / GUI)2026-08-21

クライアント(API/GUI)から「GPU0=48GB級運用 + GPU1=24GB」のような GPU 割当を
指定できるようにした。実装は gateway 側のみ(バックエンドのコードは無変更)。
実装はサブエージェントが利用上限で中断したため親セッションが直接実施。

## 実装

1. **h3 プリセット `48gb-dual`**(gateway/backends.py): README 48GB級推奨構成そのまま
   (`H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1
   H3_KEEP_TRANSFORMER=1`)。GPU0=transformer/denoise、GPU1=text_encoder。
2. **`gpus` パラメータ**(`POST /api/v1/backend/load`): `"0"` / `"1"` / `"0,1"` 形式
   (正規表現検証・重複拒否)。子プロセスの `CUDA_VISIBLE_DEVICES` に設定。
   env セットの一部として記録されるため、resident 切替時の env 不一致フォールバック
   (プロセス再起動)も自然に機能する。`/api/v1/status` の process / backends に
   `gpus` フィールドを追加。
   - 組み合わせ検証: `H3_TE_DEVICE=cuda:N` が可視GPU枚数の範囲外になる指定
     (例: 48gb-dual + gpus="1")は 400。
3. **GUI**: 起動オーバーレイと管理タブに「実行GPU」セレクト(status の vram から
   遅延populate、「GPU1(24GB)」形式)。ltx25+24GB級単独選択時と h3+単一GPU選択時に
   注意文を表示。

## 実機検証(GPU:0=96GB、GPU:1=24GB)

| 項目 | 結果 |
|---|---|
| h3 `48gb-dual` load | started 12.1s(lowvram のため起動は軽い) |
| h3 t2i 512² 1本目(初回ロード込み) | completed **64.9s** / peak 40.2GB |
| 生成後の常駐 | **GPU0 36.8GB(transformer)+ GPU1 16.9GB(TE)** — 2GPU分担を実測確認 |
| h3 t2i 2本目(KEEP_TRANSFORMER) | completed **12.2s**(約5倍高速) |
| ltx25 `gpus="1"` load → t2i 512² | completed 85.0s(24GBカード単独)、**GPU0 はベースラインのまま** |
| status の gpus 表示 | ltx25: `"1"`、未指定は null |
| 異常系 | gpus="abc" 400 / gpus="1,1" 400 / 48gb-dual+gpus="1" 400(メッセージに再番号付けの説明) |
| GUI(headless Chrome + CDP) | セレクト4択(自動/GPU0(96GB)/GPU1(24GB)/全GPU明示)、48gb-dual 表示、ltx25@GPU1 警告文、h3@単一GPU 警告文、全て描画確認 |
| 後片付け | 全ポート閉鎖、GPU:0 2.7GB / GPU:1 19MiB 復帰 |

## 既知の制限

1. **resident 切替 × 48gb-dual は自動的にプロセス停止へフォールバックする**(実測)。
   h3 の `/api/admin/unload`(runner.unload_all)が GPU1 上の TE(~17.6GB)を解放
   しないため、gateway の VRAM 解放確認(閾値2048MB)がタイムアウトし、排他原則を
   守るための設計どおりのフォールバックが発動する(切替所要 ~97s)。安全側の挙動で
   実害はないが、dual 構成では resident の高速化が効かない。改善するには h3 側
   `unload_all` の TE デバイス対応が必要(未着手)。
2. `/api/v1/generate` の auto_load は gpus を受けない(backend/load で明示起動して
   から generate する運用)。
3. 将来候補(未着手): GPU が分かれている場合の2バックエンド同時アクティブ
   (排他原則の per-GPU 化)。現状は従来どおり同時1バックエンド。
