# streaming translate application

PC上で流れる音声をキャプチャしリアルタイム翻訳を実行するアプリ
ユースケース : youtube Live のリアルタイム自動翻訳

## 処理の流れ

1. 音声分離

   認識精度向上に必要な部分 Demucsを利用\
   音声分離専用の軽量なモデルを用意してもよいかもしれない

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

TensorRTを活用すると速度が向上しそう

TODO:

- [ ] 新たな音声分離モデルの学習
      → バッファサイズが長すぎる(今だと約10秒？ → どのポイントの翻訳かわかりにくい)
- [ ] Whisperの精度検証と高速化の検討
      現状は1フレーズ1~4秒 もう少しストリーミング推論できるようにしたさがある TensorRTでさらにチューニングできるかもしれない
- [ ] 翻訳モデルの高速化と精度向上
      fugumtよりgemma-2b-jpn-itの方が性能が高そう
      → 出力の切り取り+TensorRTによる高速化を行う

## trtllm-build commands

```bash
checkpoint_dir=whisper_large_v3_fp16
output_dir=/workspace/streaming_translate_application/models/whisper_trt_engine
trtllm-build  --checkpoint_dir ${checkpoint_dir}/encoder \
              --output_dir ${output_dir}/encoder \
              --max_input_len 3000 --max_seq_len 3000 \
              --max_batch_size 8 \
              --moe_plugin disable \
              --gemm_plugin disable \
              --bert_attention_plugin float16 \
              --log_level warning \
              --profiling_verbosity detailed \
              --kv_cache_type paged

trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --max_encoder_input_len 3000 \
              --max_beam_width 1 \
              --max_batch_size 8 \
              --max_seq_len 114 \
              --max_input_len 14 \
              --kv_cache_type paged \
              --profiling_verbosity detailed \
              --moe_plugin disable \
              --gemm_plugin float16 \
              --log_level warning \
              --logits_dtype float16 \
              --bert_attention_plugin float16 \
              --gpt_attention_plugin float16
```

```bash
variant=2-2b-jpn # 27b
git clone git@hf.co:google/gemma-2-2b-jpn-it
mkdir range3/cc100-ja
wget https://huggingface.co/datasets/range3/cc100-ja/resolve/main/train_0.parquet

CKPT_PATH=gemma-2-2b-jpn-it/
UNIFIED_CKPT_PATH=gemma-2-2b-it_trt/bf16-fp8
ENGINE_PATH=/workspace/streaming_translate_application/models/gemma-2-2b-jpn-it_bf16
VOCAB_FILE_PATH=/workspace/TensorRT-LLM/examples/gemma/gemma-2-2b-jpn-it/tokenizer.model

python3 convert_checkpoint.py --ckpt-type hf \
            --model-dir ${CKPT_PATH} \
            --dtype bfloat16 \
            --world-size 1 \
            --output-model-dir ${UNIFIED_CKPT_PATH}
# -> これでsafetensorとconfigを準備する これがないとquantizeで死ぬ

# [このissueに従い修正(v0.15.0.dev2024101500)](https://github.com/NVIDIA/TensorRT-LLM/issues/2327)

python3 ../quantization/quantize.py \
            --model_dir ${CKPT_PATH} \
            --dtype bfloat16 \
            --qformat fp8 \
            --kv_cache_dtype fp8 \
            --output_dir ${UNIFIED_CKPT_PATH} \
            --tp_size 1 \
            --calib_dataset range3/cc100-ja

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --kv_cache_type paged \
             --profiling_verbosity detailed \
             --gemm_plugin fp8 \
             --output_dir ${ENGINE_PATH}
```
