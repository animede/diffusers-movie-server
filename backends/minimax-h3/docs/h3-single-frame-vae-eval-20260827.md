# Single-Frame-VAE-500K 評価 + seed→構図引き継ぎ再検証(短尺スティル) — 2026-08-27

MV Phase 3(H3でシーン画像を生成し、キャラクター参照だけを動画生成へ渡し、seedを
使い回す設計)を裏付けるための評価タスク。コミュニティ製デコーダ単体
(`iamkaikai/MiniMax-H3-Single-Frame-VAE-500K`)の忠実度と、短尺スティル
(frames=22/5)での seed→構図引き継ぎ理論を実測で再検証した。

## 対象・環境

- backend: `backends/minimax-h3`(port 8631、gateway 8630、GPU0)。
- 本番構成: preset `96gb-int8` + overrides
  `H3_REF_PREFIX_CACHE_SINGLE=1` / `H3_VOCAL_LOCK=1` /
  `H3_TURBO_LORA_FILE=minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`。
- SF-VAE チェックポイント: 既にダウンロード済み
  (`~/.cache/huggingface/hub/models--iamkaikai--MiniMax-H3-Single-Frame-VAE-500K/
  snapshots/eada4e7b5f90f8d834e7691b6aee67efbe17bd7a/`)。decoder + post_quant_conv
  のみ(585テンソル、9GB、sha256検証込み)。ライセンス: MiniMax H3 Community License。
- キャラクター参照: `/home/animede/mv_studio_V3/data/ref_images/momoo_ad503ad4.png`
  (Momo、ぬいぐるみの子犬、サンタ帽+ヘッドホン)。
- 計測用スクリプト・生成画像は
  `/tmp/claude-1000/-home-animede-diffusers-server/8e173325-c225-4db4-8732-d755b1527652/
  scratchpad/sfvae_eval/` に保存(セッション終了で消える可能性あり、パスは本文中に記載)。

## 実装メモ: 潜在の取り出し方法

`generate_ref2va()`/`generate()` の decode 直前、`_cpu_norm_video_decode_step()`
(`core/runner.py` 内で `MiniMaxH3VideoDecodeStep.__call__` をサブクラスして
差し替えている箇所)の `block_state.latents` が、SF-VAE README の
`normalized-latents.safetensors` が期待する「正規化済み・denormalize前」の
generator latent とビット単位で一致するテンソル。

一時的に `H3_DEBUG_DUMP_LATENT=1`(既定 `"0"`、no-op)でこの `latents` を
safetensors へダンプする5行のフックを `_CpuNormVideoDecodeStep.__call__` の冒頭へ
追加し、本番と全く同じロード/排他/プレフィックスキャッシュ経路(8000行超の
`generate_ref2va()` を再実装せず、実サーバの `/api/ref2va` 経由で温まった状態のまま)
で潜在を採取した。**タスク終了時に `git checkout -- backends/minimax-h3/core/runner.py`
で完全に revert 済み**(`git status` はクリーン)。

デコード比較は `cuda:1`(RTX PRO 4000 Blackwell、本番は GPU0 単独使用のため常に
空き)で独立プロセスとして実行し、本番プロセスのVRAM/ロック状態には一切触れていない。

## Q1: 同一潜在での2デコーダ比較

対象2件(いずれも 768×768、frames=22、latent_index=3(T=7の中央)):
- **Momo(キャラクター)**: seed=111、ref2i(`/api/ref2va still=1`)。
- **シーン(jazz club interior)**: seed=222、t2i(`/api/t2i`)。

| 対象 | PSNR(official vs sfvae) | SSIM | Laplacian分散(official/sfvae) | decode時間(official/sfvae) |
|---|---:|---:|---:|---:|
| Momo 768² | 15.60 dB | 0.737 | 186.9 / 173.2 | 1.054s / 0.227s |
| Scene 768² | 18.98 dB | 0.639 | 363.4 / 388.0 | 1.113s / 0.228s |
| Scene 512²(参考) | 19.95 dB | 0.648 | 212.8 / 234.1 | 0.715s / 0.106s |

PSNR/SSIM は README のベンチマーク値(288件の既知ground-truth再構成、PSNR
~31dB)よりかなり低いが、これは想定通り: README自身が明記する通り
「generator latents have no ground-truth image, so decoder-to-decoder distances
are disagreement measurements rather than quality scores」であり、我々の対象は
まさにこのケース(H3自身が生成した潜在、既知の正解画像なし)。数値は「2つの
デコーダがどれだけ違う画素を出すか」であって画質スコアではない。

### 目視結果: 構図は完全に保たれる、ただしグリッド状アーティファクトあり

Momo 768² 比較(左official/右sfvae):
`/tmp/.../sfvae_eval/q1_momo_sidebyside.png`
Scene 768² 比較: `/tmp/.../sfvae_eval/q1_scene_sidebyside.png`

- **構図・ポーズ・衣装・被写体アイデンティティは official と sfvae で完全に一致**
  (Momoのサンタ帽・ヘッドホン・オーバーオール・体勢、シーンのステージ配置・
  カーテン・テーブル配置、いずれも人間が同じ判断をできるレベルで同一)。
- **sfvae 側にのみ、画像全体を覆う規則的なグリッド/チェッカーボード状の
  アーティファクトが見える**(`/tmp/.../sfvae_eval/sfvae_flat_patch_zoom.png` vs
  `official_flat_patch_zoom.png` で平坦領域を拡大すると明瞭。official は滑らかな
  グラデーション、sfvae は規則的な格子)。512² では格子がやや目立たなくなる
  (Laplacian分散の乖離幅も512²の方が小さい)。
- **原因切り分け**: 本番VAEは `use_tiling=True`(256pxタイル、常時有効)だが、
  README の直接デコード呼び出し(`vae.decoder(vae.post_quant_conv(z))`、タイリング
  非経由)でも同じグリッドが出ることを確認した
  (`decode_notile_test.py`、`q1_scene_tiletest_direct_decoder.png` /
  `_decodeclip_notile.png` / `_decodeclip_tiled.png` の3通りいずれも同一パターン)。
  → **タイル境界のブレンド起因ではなく、SF-VAE 500Kチェックポイント自体が持つ
  アップサンプリング系のグリッド性アーティファクト**と判断。README が明記する
  「1024段階は学習率ほぼゼロで通過しただけ、768段階(stage3、20,000枚)は
  正式に学習済み」という記述と整合し、LPIPS/DISTSがofficialに劣るという
  ベンチマーク上の弱点が視覚的に現れたもの。

### Q1 結論

**画面ローンチ用のスクリーニング(構図・衣装・ポーズ判断)に使うだけなら sfvae
デコードで十分実用に足る**: 人間レビュアーが official / sfvae のどちらを見ても
同じ採用・却下判断をするはず(グリッドは目を凝らせば分かるが、構図判断を誤らせる
レベルではない)。ただし **最終出力・LPIPS重視の用途には使わない**(README の
per-domain表の通り、"Broad photographs" ドメインが最も弱く、自然写真的な
このMVのシーン画像はまさにこのドメインに該当する)。

## Q2: 短尺スティルでの seed→構図引き継ぎ再検証(コアの問い)

プロンプト固定 + Momo参照固定、3 seed(111/222/333)で以下をマトリクス実行:
- still frames=22(`/api/ref2va still=1 frames=22`、768×768、turbo=true、4steps)
- still frames=5(同上 `frames=5`、768×768)
- ref2va video(`/api/ref2va`、768×448、seconds=5、turbo=true、4steps、同一seed)

いずれも `/api/ref2va` を直接叩いた(MV層は経由していない。バックエンドAPIは
`seed` を素通しで受け付けるため素直に指定できた)。frames=5 は
`H3_VAE_SMALLCLIP_FIX=1`(既定)のままで問題なく通った(明示設定は不要だった)。

### 結果テーブル

| seed | still22 vs still5(構図) | still22 vs video先頭フレーム(構図) |
|---|---|---|
| 111 | match | match |
| 222 | match | match |
| 333 | match | match |

3 seedとも `match`(no-matchやpartialは0件)。判定根拠(目視、全て同一プロンプト
「Momo、ベンチに座る秋の公園」使用):

- **still22 vs still5**: `/tmp/.../sfvae_eval/q2/still22_vs_still5_seed{111,222,333}.png`。
  3 seedともポーズ(体育座り気味に手を膝の間に置く座り姿勢)・ベンチ・背景の
  紅葉並木・光の向きが同一。フレーム数を変えるとクロップがわずかに変わる
  (5フレーム版の方がやや広角/タイト、seedにより向きが違う)が、同じショットの
  別トリミングという範囲に収まる。
- **still22 vs video(先頭+5フレーム目)**: `/tmp/.../sfvae_eval/q2/compare_seed{111,222,333}.png`
  (左から still22 / still5 / video frame0 / video frame5)。3 seedとも構図の骨格
  (座りポーズ・ベンチ・秋の並木)が動画側でも保たれている。動画側はカメラが
  スティルよりズームイン(seed 111/222)またはズームアウト(seed 333)する傾向が
  あるが、被写体の配置・向き・衣装・シーンは一致。5フレーム目の時点でも崩れは
  見られない(動画序盤で構図が破綻しない)。

frames=5 のスティルにも H3_VAE_SMALLCLIP_FIX 由来の境界アーティファクトは
見られなかった(公式デコーダ経由、`/tmp/.../sfvae_eval/q2/still_f5_seed111_zoom.png`)。

### SF-VAE でのスティル境界バグ回避テスト

frames=5 の潜在(latent shape `[1,24,2,48,48]`、T=2)を SF-VAE で直接デコード
(`vae.decoder(vae.post_quant_conv(z))`、`_decode`/`_decode_clip` の
チャンク境界計算を一切経由しない)したところ、latent_index=0/1 どちらも
クラッシュなく単一フレームを取り出せた
(`/tmp/.../sfvae_eval/q2/f5_seed111_sfvae.png`)。**smallclip境界バグ
(`num_chunks==0` で `torch.cat([])` が落ちる)は、そもそも `_decode()`
のチャンク処理を通らない SF-VAE の使い方では構造的に発生しない**ことを確認した
(Q1と同じグリッド状アーティファクトはこちらにも見えるが、境界由来の崩れ・
ゴーストフレームは無し)。frames=5 の実用上は現状の
`H3_VAE_SMALLCLIP_FIX=1`(公式デコーダのパッチ)で既に問題なく動いているため、
この回避策自体はQ3の速度目的でのみ意味を持つ(下記)。

**1seedでは断定しない**方針の通り: 3 seed全てで一致という結果は出たが、n=3の
サンプルであり、プロンプトも1本のみ試した点は留意(異なるプロンプト文体・
複数被写体構図等の頑健性は未検証)。

## Q3: 速度

768²、`/api/ref2va still=1`、turbo=true(4steps)、warm state(モデル常駐後)での
複数回計測(n=6ずつ、Q2の3 seed分 + 追加3回)。

| frames | denoise(平均) | decode(平均) | total_elapsed(平均) | total_elapsed range |
|---|---:|---:|---:|---:|
| 22(0.917s相当、既定) | 10.01s | 1.07s | 26.10s | 25.1〜29.2s |
| 5(0.208s相当、実験値) | 7.92s | 0.40s | 22.70s | 22.2〜23.2s |

frames=5 は denoise が約2.1s、decode が約0.7s 速い(合計約3.4s)。
`total_elapsed` の差(平均3.4s)は概ねこの2フェーズの差で説明できる
(encode/mux等の固定費はframe数非依存のため相殺)。

H3_PHASE_TIMING=1 で warm state の内訳を採取(frames=22、seed=555、
`generate_ref2va` breakdown ログ):

```
entry_lock=0.00s, setup_step=0.04s, text_encode(prefix cache HIT)=0.34s,
vae_to_gpu=2.01s, reference_encoder_step(vae encode条件潜在)=1.59s,
vae_to_cpu=3.51s, ensure_transformer_ref=0.00s(既に常駐)
-- このブロックの sum=7.54s
その後: denoise=9.71s, decode=1.05s
```

初回リクエスト(コールドスタート、`transformer_ref` 未ロード)は
`ensure_transformer_ref=16.56s` + プレフィックスキャッシュ MISS(2.2s) が乗り
`total_elapsed_s=47.44s` になった(本番の「初回は重い」という既知の挙動どおり、
本タスクの速度評価はwarm state基準)。

Q1のデコード単体計測(768²、1潜在フレーム)も合わせると:

| デコード方式 | 単体decode時間(768²、1潜在フレーム) |
|---|---:|
| 公式video-VAE(`_decode`経由、チャンク処理込み) | 1.05〜1.11s |
| SF-VAE(直接 `decoder()`呼び出し) | 0.227〜0.229s |

**SF-VAEはデコード単体で約4.6〜4.9倍速い**が、`decode_time_s`(1.07s、warm平均)
がtotal(26.10s)に占める割合は約4%に過ぎないため、**SF-VAE単独導入による
total_elapsed短縮効果は小さい**(理論上 ~0.85s、3%程度)。速度面での主な
レバーは引き続き frames=5(denoise短縮込みで合計約3.4s、13%減)であり、
SF-VAEはそれとは独立に「デコード方式の選択肢」として効く(frames=5と
SF-VAEを両方使えば理論上 denoise 7.92s + sfvae decode 0.23s ≈ 8.15s の
コア時間、frames=22公式の 10.01+1.07=11.08s から約26%減)。

HANDOFFが挙げる「21秒プレビュー」的な用途では、SF-VAE単体よりも
frames=5化の方が効果が大きく、SF-VAEは「粗いプレビューを一段と軽くする」
追加の最適化として位置づけるのが妥当。

## 統合可否の提言

1. **SF-VAEをスティルモードのデフォルトデコーダにするのは尚早(No-Go)**。
   Q1で確認した規則的なグリッドアーティファクトは、720p級シーン画像として
   最終出力に使うにはノイズが目立ちすぎる(README自身も natural photographs
   ドメインでの弱さを認めている)。
2. **ただし「構図プレビュー専用」の用途に限れば有望(Go、フラグ付き)**。
   Q1で構図・衣装・ポーズの判断は official と sfvae で一致することを確認した。
   MV Phase 3 の「seedプレビューを大量に出して人間が良い構図を選ぶ」用途では、
   グリッドノイズは実害が小さい一方、decode時間を1/4〜1/5に削減できる。
   導入するなら既存の `H3_VAE_SMALLCLIP_FIX` と同じ思想の opt-in フラグ
   (例 `H3_STILL_DECODER=sfvae`、既定は現状維持の `official`)で追加し、
   スティルモード(`/api/t2i`・`/api/ref2va still=1`)の decode ステップだけ
   `vae.decoder(vae.post_quant_conv(z))` の直接呼び出しに差し替える設計が
   最小差分になる(`_cpu_norm_video_decode_step()` と同様、`MiniMaxH3VideoDecodeStep`
   のサブクラス differ で対応可能。動画モード・`t2i_batch`/`ref_batch` の
   位相並べ替え経路は対象外のままでよい)。
3. **フォローアップ実装タスク(未着手)**:
   - SF-VAEの重み(9GB)をどこに常駐させるか(公式VAEと同時ロードは追加VRAM
     ~1-2GB程度で軽微だが、`state`管理・排他ロード設計への組み込みが必要)。
   - グリッドアーティファクトの強度が解像度依存(768²で顕著、512²でやや軽微)
     である点を踏まえ、スティル既定解像度でのアーティファクト許容度を
     ユーザー(プレビュー利用側)に確認する。
   - frames=5 + SF-VAE の組み合わせ(理論値 denoise 7.92s + sfvae decode 0.23s)
     を実際に配線してエンドツーエンドのtotal_elapsed短縮を計測する
     (本タスクでは decode_compare.py によるオフライン比較のみ、配線はしていない)。
   - Q2は3 seed×1プロンプトのみの検証。プロンプト・構図の多様性を広げた
     頑健性確認が望ましい。

## 復元確認

- `git status`: クリーン(`backends/minimax-h3/core/runner.py` revert済み、
  他の変更なし)。
- gateway `backend/load` で本番構成
  (`preset=96gb-int8` + `H3_REF_PREFIX_CACHE_SINGLE=1` + `H3_VOCAL_LOCK=1` +
  `H3_TURBO_LORA_FILE=minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`、
  `gpus=0`、debugフラグ・phase timingフラグなし)へ再ロード済み。
  `/api/v1/status` で env_extra が完全一致することを確認。
- ヘルスチェック: `busy: false`、`transformer_quant: int8` 確認済み
  (追加生成は本番トラフィックへの余計な負荷を避けるため実施せず、idle確認のみ)。
- GPU: GPU0 は本番常駐分(~56.7GB)のみ、GPU1(評価で使用)は 18MiB
  (アイドルベースライン相当)まで復帰。評価用の独立プロセス
  (`decode_compare.py` 等)は全て実行完了後に自然終了しており、残存プロセスなし。

## 参照ファイル(パス、レビュー用)

- 評価スクリプト: `.../scratchpad/sfvae_eval/decode_compare.py`,
  `decode_notile_test.py`
- Q1 画像: `q1_momo_sidebyside.png`, `q1_scene_sidebyside.png`,
  `sfvae_flat_patch_zoom.png` / `official_flat_patch_zoom.png`(グリッド比較),
  `q1_scene_tiletest_*.png`(タイリング切り分け)
- Q2 画像: `q2/compare_seed{111,222,333}.png`(still22|still5|video frame0|frame5),
  `q2/still22_vs_still5_seed{111,222,333}.png`, `q2/f5_seed111_sfvae.png`
  (SF-VAEでのsmallclip回避テスト)
- 生JSON応答: `q1_momo_still.json`, `q1_scene_still.json`,
  `q2/still_f{22,5}_seed{111,222,333}.json`, `q2/video_seed{111,222,333}.json`,
  `q3_*.json`
