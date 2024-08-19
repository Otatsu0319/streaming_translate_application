import queue

import librosa
import numpy as np
import soundfile as sf
import torch
from demucs import apply
from demucs.pretrained import get_model
from torchaudio.transforms import Fade

torch.hub.set_dir("../models")


class VoiceSeparator:
    def __init__(self):
        self.model = get_model("htdemucs")
        # torch.jit.load("demucs_models/demucs_ts.pt")

        self.model_samplerate = self.model.samplerate
        self.model_audio_channels = self.model.audio_channels
        self.model_sources = self.model.sources
        self.model_segment = 10.0  # model.max_allowed_segment  # 7.8s

        self.overlap = 0.5  # sec
        self.overlap_frames = int(self.model_samplerate * self.overlap)
        self.chunk_size = int(self.model_samplerate * self.model_segment)  # 7.8s

        self.fade = Fade(fade_in_len=self.overlap_frames, fade_out_len=self.overlap_frames, fade_shape="half_sine")

        self.buffer = np.zeros((self.model_audio_channels, (self.overlap_frames)), dtype=np.float32)

        print("overlap frames:", self.overlap_frames)
        print("chunk size:", self.chunk_size)
        print("buffer size:", self.buffer.shape)

    def _check_chunk_dim(self, chunk):
        if chunk.shape[0] == self.model_audio_channels:
            pass
        elif chunk.shape[0] != self.model_audio_channels and self.model_audio_channels == 1:
            chunk = chunk.mean(0, keepdim=True)
        elif chunk.shape[0] != self.model_audio_channels and self.model_audio_channels == 2:
            chunk = torch.cat([chunk, chunk], dim=0)
        else:
            raise ValueError(f"Expected {self.model_audio_channels} channels, got {chunk.shape[0]}")
        return chunk

    def extract_voice(self, chunk: np.ndarray):
        chunk = torch.Tensor(chunk)
        chunk = self._check_chunk_dim(chunk)

        ref = chunk.mean(0)
        chunk -= ref.mean()
        chunk /= ref.std() + 1e-6

        sources = apply.apply_model(
            self.model,
            chunk[None],
            device="cuda",
            shifts=1,
            overlap=self.overlap,
            progress=False,
            num_workers=0,
            segment=None,
            split=True,
        )[0]

        sources *= ref.std() + 1e-6
        sources += ref.mean()

        sources = self.fade(sources)
        sources = list(sources)

        stem = "vocals"
        forcus_stem: torch.Tensor = sources.pop(self.model_sources.index(stem))
        forcus_stem = forcus_stem.cpu().numpy()

        # other_stem = torch.zeros_like(sources[0])
        # for i in sources:
        #     other_stem += i
        # other_stem = other_stem.cpu().numpy()

        # return forcus_stem, other_stem
        return forcus_stem

    def extract_streamer(self, chunk_queue: queue.Queue):
        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                break

            extract_chank = self.extract_voice(chunk)

            extract_chank[:, : self.overlap_frames] += self.buffer
            self.buffer = extract_chank[:, -self.overlap_frames :]

            yield extract_chank[:, : -self.overlap_frames]


if __name__ == "__main__":
    vs = VoiceSeparator()

    wav, sr = sf.read("/workspace/streaming_translate_application/audio_samples/test.wav", always_2d=True)
    wav = wav.T
    wav = librosa.resample(wav, orig_sr=sr, target_sr=vs.model_samplerate).astype("float32")
    data_length = wav.shape[1]
    start = 0
    chunk_queueq = queue.Queue()

    print(wav.shape)
    while start < data_length - vs.chunk_size:
        chunk = wav[:, start : start + vs.chunk_size].copy()
        chunk_queueq.put(chunk)
        start += vs.chunk_size - vs.overlap_frames
    chunk_queueq.put(None)

    forcus_stem_out = np.array([])

    for i, chunk in enumerate(vs.extract_streamer(chunk_queueq)):
        forcus_stem = librosa.to_mono(chunk)
        forcus_stem_out = np.append(forcus_stem_out, forcus_stem)

    sf.write(
        "/workspace/streaming_translate_application/results/htdemucs_vocal.wav",
        forcus_stem_out,
        samplerate=vs.model_samplerate,
    )
