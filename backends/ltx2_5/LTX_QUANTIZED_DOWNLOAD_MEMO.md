# LTX-2.5を量子化しながら取得する再開メモ

参考記事: [【Diffusers】LTX-2.5 Distilledを量子化して保存する](https://touch-sp.hatenablog.com/entry/2026/08/14/212914)

## 目的

`Lightricks/LTX-2.5-Diffusers` の非量子化版を丸ごと保存してから変換すると、変換中に
「元モデル一式 + 量子化後モデル」の容量が必要になる。それを避け、重いコンポーネントを
1個ずつ取得時にbitsandbytes NF4へ量子化して保存する。

## 採用する流れ

モデルとリビジョンは現在のアプリ設定に固定する。

```text
REPO_ID=Lightricks/LTX-2.5-Diffusers
REVISION=69009ff070135c693ad1ad1ef2cc149c227963da
出力先=LTX-2.5-Diffusers-bnb-4bit
```

1. Hugging Faceでモデルの利用条件に同意し、Read tokenを用意する。
2. `bitsandbytes`、開発版`diffusers`、開発版`transformers`を準備する。
3. `snapshot_download()`で、巨大な`text_encoder/**`と`transformer/**`を除く
   パイプライン構成ファイル・VAE・vocoder等だけを出力先へ取得する。
4. 専用の一時Hubキャッシュを作る。
5. `text_encoder`だけをHubから`load_in_4bit=True`、`nf4`、計算dtype
   `torch.bfloat16`で読み込む。この読み込み中に量子化される。
6. `text_encoder_bnb_4bit/`へ`save_pretrained()`し、保存物を再ロードできることを確認する。
7. text encoderをメモリから解放し、その一時Hubキャッシュを削除する。
8. 新しい一時Hubキャッシュで`transformer`だけを同様に取得・量子化する。
9. `transformer_bnb_4bit/`へ保存して再ロード確認後、一時Hubキャッシュを削除する。
10. パイプライン読み込み時は、保存した2つの量子化コンポーネントを明示的に渡す。

重要なのは、`text_encoder`と`transformer`の非量子化重みを同じHubキャッシュに
貯めたままにしないこと。各コンポーネントの量子化済み保存と再ロード確認が終わってから、
そのコンポーネント専用の一時キャッシュだけを削除する。

## 量子化設定

記事と同じ設定を使う。

```python
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16
```

- `text_encoder`: `transformers.BitsAndBytesConfig` と
  `Gemma4UnifiedForConditionalGeneration`
- `transformer`: `diffusers.BitsAndBytesConfig` と
  `LTX2VideoTransformer3DModel`
- 保存: 各モデルの `save_pretrained()`

量子化済みモデルの再利用時は、記事の例と同様に各`AutoModel.from_pretrained()`で
量子化済みディレクトリを読み、`LTX2Pipeline.from_pretrained()`へ
`text_encoder=`と`transformer=`として渡す。再ロード時に量子化設定をもう一度指定する
必要はない。

## ディスク容量上のポイント

- この方法でも、処理中の1コンポーネント分については非量子化ダウンロードキャッシュが
  一時的に必要。ネットワークから各shardを直接4bitファイルとして保存する方式ではない。
- ただし、非量子化モデル一式を永続保存せず、2つの巨大コンポーネントを同時にキャッシュ
  しないため、ピーク容量と最終使用量を抑えられる。
- 一時キャッシュは専用パスに限定する。通常の`~/.cache/huggingface`全体は削除しない。
- 中断時は量子化済み出力を消さず、未完了コンポーネントの専用一時キャッシュから再開する。
- `save_pretrained()`直後に元キャッシュを消さず、量子化済みディレクトリからの再ロード成功を
  必ず先に確認する。

## 次回再開時に実装するもの

`scripts/download_quantize_ltx25.py`を作り、次を自動化する。

- 固定リビジョンとtokenを使用
- 重い2ディレクトリを除いたベースファイルの取得
- text encoderとtransformerの順次NF4量子化
- コンポーネントごとに独立した一時キャッシュ
- 保存後の再ロード検証
- 検証成功時だけ対応する一時キャッシュを削除
- 完了済みコンポーネントを検出して再実行時にスキップ

既存の`app/generator.py`も、完成した
`LTX-2.5-Diffusers-bnb-4bit/text_encoder_bnb_4bit`と
`transformer_bnb_4bit`を読むように変更する必要がある。
