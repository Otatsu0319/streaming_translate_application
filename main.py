import logging
import queue
from collections import deque
from threading import Thread

import librosa
import numpy as np
import torch
from faster_whisper import WhisperModel
from llama_cpp import Llama
from silero_vad import load_silero_vad

import streamer
import voice_separator

SAMPLING_RATE = 16000
MODEL_LABEL = "large-v3"  # or "distil-large-v3"


PROMPT_JP2EN = "Translate this from Japanese to English:\nJapanese: {0}\nEnglish:"
PROMPT_EN2JP = "Translate this from English to Japanese:\nEnglish: {0}\nJapanese:"

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    llm = Llama.from_pretrained(
        "mmnga/webbigdata-ALMA-7B-Ja-V2-gguf",
        "*q8_0.gguf",
        local_dir="/workspace/models",
        cache_dir="/workspace/models",
        n_gpu_layers=-1,
        verbose=False,
    )

    silero_model = load_silero_vad(onnx=True)
    probs = deque(maxlen=3)
    vad_threshold = 0.5
    is_speech = False
    buffer = []
    # It is better to tune this parameter for each dataset separately,
    # but "lazy" 0.5 is pretty good for most datasets.

    # Run on GPU with FP16
    whisper_model = WhisperModel(MODEL_LABEL, device="cuda", compute_type="float16", download_root="../models")
    speech_queue = queue.Queue()
    transcribe_log_probability_threshold = -0.5

    running = True

    vs = voice_separator.VoiceSeparator()

    receiver = streamer.VBANStreamingReceiver(
        "127.0.0.1",
        "Stream1",
        6981,
        current_sample_rate=vs.model_samplerate,
        current_channels=vs.model_audio_channels,
        chunk_size=vs.chunk_size - vs.overlap_frames,
    )
    # receiver = streamer.WavStreamReceiver(
    #     "./audio_samples/sample_elira.wav",
    #     current_sample_rate=vs.model_samplerate,
    #     current_channels=vs.model_audio_channels,
    #     chunk_size=vs.chunk_size - vs.overlap_frames,
    # )
    sound_queue = queue.Queue()

    def transcribe_thread():
        while running:
            speech = speech_queue.get()
            speech = np.array(speech)

            # TODO: 精度を高める int16 -> float32の変換が死んでる？ フラグメントを保存しつつ確認していく
            segments, _ = whisper_model.transcribe(
                speech, language="ja", log_prob_threshold=transcribe_log_probability_threshold
            )
            for segment in segments:
                print("[id:%d, p:%.02f] %s" % (segment.id, segment.avg_logprob, segment.text))
                output = llm(PROMPT_JP2EN.format(segment.text), max_tokens=128, stop=["\n"])
                print("translated: ", output['choices'][0]['text'])

    def sound_receiv_thread():
        for chunk in receiver.recv_generator():
            sound_queue.put(chunk.T)

    print("stand-by cmpl.")

    th = Thread(target=transcribe_thread)
    th.start()
    sd_th = Thread(target=sound_receiv_thread)
    sd_th.start()

    print("start streaming...")

    chunk_buffer = []

    for chunk in vs.extract_streamer(sound_queue):
        chunk = librosa.to_mono(chunk)
        chunk = librosa.resample(chunk, orig_sr=vs.model_samplerate, target_sr=SAMPLING_RATE)

        chunk_buffer.extend(chunk)
        for i in range(0, len(chunk_buffer) - 512, 512):
            chunk = torch.Tensor(chunk_buffer[i : i + 512]).float()

            speech_prob = silero_model(chunk, SAMPLING_RATE).item()
            # print(speech_prob)
            probs.append(speech_prob)

            if not is_speech and speech_prob > vad_threshold:
                is_speech = True

            if is_speech:
                buffer.extend(chunk.tolist())

            if is_speech and (sum(probs) / len(probs)) < vad_threshold:
                is_speech = False
                speech_queue.put(buffer)
                buffer = []

        chunk_buffer = chunk_buffer[i + 512 :]

    th.join()
