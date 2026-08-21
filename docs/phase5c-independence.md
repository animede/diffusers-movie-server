# Phase 5c: venv 完全独立化・データ所有権の移転(2026-08-21)

旧ディレクトリ(/home/animede/minimax-h3、/home/animede/diffusers-ltx2_5)への
依存を解消した。venv 再構築はサブエージェント、パリティ最終確認・データ移転・
最終検証は親セッションが実施(サブエージェントが利用上限で中断したため引き継ぎ)。

## 1. venv 再構築

| venv | 内容 | 検証 |
|---|---|---|
| backends/minimax-h3/venv | python3.12 + torch 2.9.0+cu128 / transformers 5.14.1 / torchao 0.17.0 / diffusers git@f37ab93(PR#14355 ピン)/ sageattention 2.2.0(third_party/wheels のビルド済み wheel から、ソースビルドなし) | pip freeze でピン一致確認 |
| backends/ltx2_5/.venv | python3.12 + torch 2.11.0+cu130 / diffusers git@11a82a15 / transformers git@f9b76f2d / kernels(NATTEN プリビルト)/ xformers 0.0.35(backends/ltx2_5/third_party/wheels に wheel を同梱) | 同上 |

- torchao の「Skipping import of cpp extensions … torch >= 2.11.0」警告は既知・無害
  (旧venvでも同様、VENV_REBUILD.md 参照)。
- 旧venvは旧ディレクトリに無変更で残置(旧サーバのフォールバック用)。

## 2. パリティ検証(新venv vs 旧venv、同一 seed/prompt)

| バックエンド | 条件 | 結果 |
|---|---|---|
| h3(48gb-lowvram、t2i 512²・seed=12345) | 旧venv出力 vs 新venv出力 | **MD5 完全一致**(0b886c6c…) |
| ltx25(nf4、t2i 512²・seed=42・"a red apple on a wooden table") | 旧venv出力(job 619d0dc6)vs 新venv出力(job 42429eff) | **MD5 完全一致**(97159f7f…) |

注: サブエージェントの当初比較はプロンプト差("photorealistic" 有無)で不一致に
見えたが、同一条件で再取得したところビット一致した(誤検出)。

## 3. データ所有権の移転(mv + 逆向き symlink)

両バックエンド停止状態で実施。同一FS内 mv のため瞬時・ディスク消費なし。

| 実体(新、本リポジトリ) | 旧位置(symlink化) |
|---|---|
| backends/minimax-h3/models(prequant 36GB) | /home/animede/minimax-h3/models → 新 |
| backends/ltx2_5/LTX-2.5-Diffusers-bnb-4bit(27GB) | /home/animede/diffusers-ltx2_5/同名 → 新 |
| backends/ltx2_5/loras | /home/animede/diffusers-ltx2_5/loras → 新 |

新旧どちらのパスからも `ls` で解決できることを確認済み。旧サーバは symlink 経由で
引き続き起動可能(実起動確認は未実施、パス解決の確認のみ)。

## 4. 最終検証(gateway 経由フル1周)

| ステップ | 実測 |
|---|---|
| ltx25 load(nf4)→ 統一API t2i | completed(従来同等) |
| resident 切替 → h3(48gb-lowvram) | 8.7s |
| h3 統一API t2i 512² | completed 65.2s / peak 34.96GB(Phase 4 実測と一致)。attn_backend=sage(新venvの sageattention wheel が有効) |
| unload(process)→ 全停止 | VRAM 2.88GB ベースライン復帰、全ポート閉鎖 |

## 5. 作業中の注意メモ

- `pkill -f 'uvicorn …8632'` のようなパターンは**発行元シェル自身のコマンドラインにも
  一致して自爆する**(実際に発生、mv 実行前だったため無害)。プロセス停止は PID 指定で
  行うこと。
- ディスク: venv 再構築で約 16GB 消費(空き 191GB → 175GB)。
- 残課題(任意): 旧サーバの symlink 経由での実起動確認、gallery のページ送りUI、
  旧ディレクトリの廃止判断(MIGRATION.md (a) 参照)。
