import logging
import time

import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

import streamer
from whisper_online import FasterWhisperASR, OnlineASRProcessor

SAMPLING_RATE = 16000
MODEL_LABEL = "distil-large-v3"  # or "large-v3"

if __name__ == "__main__":
    model = load_silero_vad(onnx=True)
    logging.basicConfig(level=logging.ERROR)

    # receiver = streamer.VBANStreamingReceiver("127.0.0.1", "Stream1", 6981)
    receiver = streamer.WavStreamReceiver("test.wav")

    # asr = FasterWhisperASR("en", MODEL_LABEL, cache_dir="../models")

    # # Run on GPU with FP16
    # # model = WhisperModel(MODEL_LABEL, device="cuda", compute_type="float16", download_root="../models")
    # online = OnlineASRProcessor(asr)

    # for data in receiver.recv_generator():
    #     online.insert_audio_chunk(data)
    #     o = online.process_iter()
    #     print(o[2])

    # just probabilities
    speech_probs = []
    for chunk in receiver.recv_generator():
        if len(chunk) < 512:
            break
        chunk = torch.from_numpy(chunk).float()
        speech_prob = model(chunk, SAMPLING_RATE).item()
        speech_probs.append(speech_prob)
        # threshold = 0.5
        # It is better to tune this parameter for each dataset separately,
        # but "lazy" 0.5 is pretty good for most datasets.

    print(speech_probs[:25])  # first 10 chunks predicts
