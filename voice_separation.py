from fractions import Fraction

import librosa
import soundfile as sf
import torch
from demucs import apply
from demucs.pretrained import get_model

torch.hub.set_dir("../models")
model = get_model("htdemucs")
# torch.jit.load("demucs_models/demucs_ts.pt")

model_samplerate = 44100
model_audio_channels = 2
model_sources = ['drums', 'bass', 'other', 'vocals']
model_segment = Fraction(39, 5)


wav, sr = sf.read("/workspace/streaming_translate_application/audio_samples/test.wav", always_2d=True)
wav = wav.T
wav = librosa.resample(wav, orig_sr=sr, target_sr=model_samplerate).astype("float32")
wav = torch.from_numpy(wav)

if wav.shape[0] == model_audio_channels:
    pass
elif wav.shape[0] != model_audio_channels and model_audio_channels == 1:
    wav = wav.mean(0, keepdim=True)
elif wav.shape[0] != model_audio_channels and model_audio_channels == 2:
    wav = torch.cat([wav, wav], dim=0)
else:
    raise ValueError(f"Expected {model_audio_channels} channels, got {wav.shape[0]}")

ref = wav.mean(0)
wav -= ref.mean()
wav /= ref.std()


sources = apply.apply_model(
    model,
    wav[None],
    device="cuda",
    shifts=1,
    overlap=0.25,
    progress=True,
    num_workers=0,
    segment=None,
)[0]


sources *= ref.std()
sources += ref.mean()


ext = "wav"
kwargs = {
    "samplerate": model_samplerate,
    "clip": "rescale",
    "as_float": False,
    "bits_per_sample": 16,  # if need : 24
}
sources = list(sources)
stem = "vocals"
forcus_stem = sources.pop(model_sources.index(stem))
forcus_stem = forcus_stem.cpu().numpy().T
sf.write(
    "/workspace/streaming_translate_application/results/htdemucs_vocal.wav",
    forcus_stem,
    samplerate=model_samplerate,
)

# Warning : after poping the stem, selected stem is no longer in the list 'sources'
other_stem = torch.zeros_like(sources[0])
for i in sources:
    other_stem += i
other_stem = other_stem.cpu().numpy().T
sf.write(
    "/workspace/streaming_translate_application/results/htdemucs_no_vocals.wav",
    other_stem,
    samplerate=model_samplerate,
)
