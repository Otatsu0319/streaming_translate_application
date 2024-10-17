#!/usr/bin/python
import logging
import queue

# import time
from threading import Thread

import librosa
import numpy as np
import torch
from silero_vad import load_silero_vad

import streamer
import voice_separator
import voice_transcriber

SAMPLING_RATE = 16000


CHUNK_SIZE = 512  # silero_vad model's chunk size


class StreamingTranslateApp:
    def __init__(self):
        logging.basicConfig(level=logging.ERROR)

        self.vs = voice_separator.VoiceSeparator()

        self.receiver = streamer.VBANStreamingReceiver(
            "127.0.0.1",
            "Stream1",
            6981,
            current_sample_rate=self.vs.model_samplerate,
            current_channels=self.vs.model_audio_channels,
            chunk_size=self.vs.chunk_size - self.vs.overlap_frames,
        )
        self.sound_queue = queue.Queue()
        self.speech_queue = queue.Queue()

        self.transcriber = voice_transcriber.VoiceTranscriber(self.speech_queue)

        self.silero_model = load_silero_vad(onnx=True)
        self.vad_threshold = 0.5
        # It is better to tune this parameter for each dataset separately,
        # but "lazy" 0.5 is pretty good for most datasets.

        onvoice_timeout = 0.5  # sec
        self.onvoice_timeout_chunk_count = (
            int(onvoice_timeout * SAMPLING_RATE) // CHUNK_SIZE + 0
            if int(onvoice_timeout * SAMPLING_RATE) % CHUNK_SIZE == 0
            else 1
        )

        self.buffer = []
        self.chunk_buffer = np.zeros(0)

        self.tr_th = Thread(target=self.transcriber.transcribe_thread)
        self.sd_th = Thread(target=self.sound_receiv_thread)
        print("stand-by cmpl.")

    def sound_receiv_thread(self):
        for chunk in self.receiver.recv_generator():
            self.sound_queue.put(chunk.T)

    def streaming_thread(self):
        self.tr_th.start()
        self.sd_th.start()

        timeout_counter = 0

        print("start streaming...")

        for chunk in self.vs.extract_streamer(self.sound_queue):
            chunk = librosa.to_mono(chunk)
            chunk = librosa.resample(chunk, orig_sr=self.vs.model_samplerate, target_sr=SAMPLING_RATE)

            self.chunk_buffer = np.append(self.chunk_buffer, chunk)

            for i in range(0, len(self.chunk_buffer) - CHUNK_SIZE, CHUNK_SIZE):
                chunk = torch.Tensor(self.chunk_buffer[i : i + CHUNK_SIZE]).float()

                speech_prob = self.silero_model(chunk, SAMPLING_RATE).item()

                if self.vad_threshold < speech_prob:
                    timeout_counter = 0
                else:
                    timeout_counter += 1

                if self.vad_threshold < speech_prob and timeout_counter <= self.onvoice_timeout_chunk_count:
                    self.buffer.extend(chunk.tolist())

                if (
                    speech_prob < self.vad_threshold
                    and self.onvoice_timeout_chunk_count < timeout_counter
                    and 0 < len(self.buffer)
                ):
                    self.transcriber.speech_queue.put(self.buffer)
                    self.buffer = []

            self.chunk_buffer = self.chunk_buffer[i + CHUNK_SIZE :]

        self.tr_th.join()
        self.sd_th.join()

    def end(self):
        print("shutting down...")
        self.sound_queue.put(None)
        self.speech_queue.put(None)
        self.receiver._running = False


if __name__ == "__main__":
    stapp = StreamingTranslateApp()
    th = Thread(target=stapp.streaming_thread)
    th.start()

    input("Press Enter to stop...\n")
    stapp.end()
