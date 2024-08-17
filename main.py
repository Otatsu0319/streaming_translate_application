import logging
import queue
from collections import deque
from threading import Thread

import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

import streamer

SAMPLING_RATE = 16000
MODEL_LABEL = "large-v3"  # or "distil-large-v3"

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

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

    def transcribe_thread():
        while running:
            speech = speech_queue.get()
            speech = np.array(speech)

            segments, _ = whisper_model.transcribe(speech, language="en")
            for segment in segments:
                if segment.avg_logprob > transcribe_log_probability_threshold:
                    print("[%.2fs -> %.2fs] %s p:%s" % (segment.start, segment.end, segment.text, segment.avg_logprob))

    th = Thread(target=transcribe_thread)
    th.start()

    # receiver = streamer.VBANStreamingReceiver("127.0.0.1", "Stream1", 6981)
    receiver = streamer.WavStreamReceiver("./audio_samples/sample_elira.wav")

    for chunk in receiver.recv_generator():
        chunk = torch.from_numpy(chunk).float()
        if len(chunk) < 512:
            chunk = torch.cat([chunk, torch.zeros(512 - len(chunk))])

        speech_prob = silero_model(chunk, SAMPLING_RATE).item()

        probs.append(speech_prob)

        if not is_speech and speech_prob > vad_threshold:
            is_speech = True

        if is_speech:
            buffer.extend(chunk.tolist())

        if is_speech and (sum(probs) / len(probs)) < vad_threshold:
            is_speech = False
            speech_queue.put(buffer)
            buffer = []

    th.join()
