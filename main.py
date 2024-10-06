#!/usr/bin/python
import logging
import queue
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


def main():
    logging.basicConfig(level=logging.ERROR)

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

    transcriber = voice_transcriber.VoiceTranscriber()

    def sound_receiv_thread():
        for chunk in receiver.recv_generator():
            sound_queue.put(chunk.T)

    silero_model = load_silero_vad(onnx=True)
    vad_threshold = 0.5
    # It is better to tune this parameter for each dataset separately,
    # but "lazy" 0.5 is pretty good for most datasets.
    onvoice_timeout = 0.5  # sec
    onvoice_timeout_chunk_count = (
        int(onvoice_timeout * SAMPLING_RATE) // CHUNK_SIZE + 0
        if int(onvoice_timeout * SAMPLING_RATE) % CHUNK_SIZE == 0
        else 1
    )
    buffer = []

    print("stand-by cmpl.")

    th = Thread(target=transcriber.transcribe_thread)
    th.start()
    sd_th = Thread(target=sound_receiv_thread)
    sd_th.start()

    print("start streaming...")

    chunk_buffer = np.zeros(0)

    timeout_counter = 0

    for chunk in vs.extract_streamer(sound_queue):
        chunk = librosa.to_mono(chunk)
        chunk = librosa.resample(chunk, orig_sr=vs.model_samplerate, target_sr=SAMPLING_RATE)

        chunk_buffer = np.append(chunk_buffer, chunk)
        i = 0
        for i in range(0, len(chunk_buffer) - CHUNK_SIZE, CHUNK_SIZE):
            chunk = torch.Tensor(chunk_buffer[i : i + CHUNK_SIZE]).float()

            speech_prob = silero_model(chunk, SAMPLING_RATE).item()
            # print(speech_prob)

            if vad_threshold < speech_prob:
                timeout_counter = 0
            else:
                timeout_counter += 1

            if vad_threshold < speech_prob and timeout_counter <= onvoice_timeout_chunk_count:
                buffer.extend(chunk.tolist())

            if speech_prob < vad_threshold and onvoice_timeout_chunk_count < timeout_counter and 0 < len(buffer):
                transcriber.speech_queue.put(buffer)
                buffer = []

        chunk_buffer = chunk_buffer[i + CHUNK_SIZE :]

    th.join()


if __name__ == "__main__":
    main()
