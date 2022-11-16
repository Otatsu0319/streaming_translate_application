# streaming translate application

PC上で流れる音声をキャプチャしリアルタイム翻訳を実行するアプリ
ユースケース : youtube Live のリアルタイム自動翻訳

## 処理の流れ

1. 音声分離

   認識精度向上に必要な部分 今は実装していない
   話者分離 : ConvTasNet, sepformer, VoiceFilter
   ボーカル抽出 : Spleeter, Demucs

2. 音声認識

   whisperを用いた音声認識\
   多くの言語に対応し、英語には専用モデルが存在\
   言語判定(自由言語対応)を利用することも言語指定(高速)をすることもできる\
   英語へは直接翻訳できるらしいが精度は知らない(高速化オプションとして用意？)

3. 翻訳

   nllb-200を用いた翻訳\
   こちらも多言語に対応 精度の確保にはbeam数を増やす必要がある

## 再生音の取得

- ステレオミキサー等を利用した録音

  開発環境にはVoicemeeterが入っているのでこれを使う
  python-sounddeviceで簡単に利用可能

- 拡張機能を利用した録音

  chrome.tabCaptureを使うとできるらしい

## 開発メモ

既存のclient -> serverに翻訳コードを追記するだけでそれらしきものは動いた
バッファをどうするかは重要になりそう 今は結構レイテンシが高い
またGPUに処理がオフロードできていないため他の作業に支障をきたしそう

## 謝辞

サーバーコードの公開に感謝します\
https://github.com/TadaoYamaoka/StreamingWhisper/blob/master/WhisperServer/WhisperServer.py
