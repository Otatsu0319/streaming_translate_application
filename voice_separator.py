import librosa
import numpy as np
import soundfile as sf
import torch
from demucs import apply
from demucs.pretrained import get_model
from torchaudio.transforms import Fade

torch.hub.set_dir("../models")
model = get_model("htdemucs")
# torch.jit.load("demucs_models/demucs_ts.pt")

model_samplerate = model.samplerate
model_audio_channels = model.audio_channels
model_sources = model.sources
model_segment = 10.0  # model.max_allowed_segment  # 7.8s


wav, sr = sf.read("/workspace/streaming_translate_application/audio_samples/test.wav", always_2d=True)

wav = wav.T
wav = librosa.resample(wav, orig_sr=sr, target_sr=model_samplerate).astype("float32")
wav = torch.Tensor(wav)
data_length = wav.shape[1]

if wav.shape[0] == model_audio_channels:
    pass
elif wav.shape[0] != model_audio_channels and model_audio_channels == 1:
    wav = wav.mean(0, keepdim=True)
elif wav.shape[0] != model_audio_channels and model_audio_channels == 2:
    wav = torch.cat([wav, wav], dim=0)
else:
    raise ValueError(f"Expected {model_audio_channels} channels, got {wav.shape[0]}")

overlap = 0.5  # sec
overlap_frames = int(model_samplerate * overlap)
chunk_size = int(model_samplerate * model_segment)  # 7.8s

fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape='half_sine')


forcus_stem_out = np.array([])
other_stem_out = np.array([])

start = 0
end = chunk_size + overlap_frames


print(wav.shape)
while start < data_length - (chunk_size + 2 * overlap_frames):
    chunk = wav[:, start:end]
    chunk = torch.empty_like(chunk).copy_(chunk)
    ref = chunk.mean(0)
    chunk -= ref.mean()
    chunk /= ref.std() + 1e-6

    sources = apply.apply_model(
        model,
        chunk[None],
        device="cuda",
        shifts=1,
        overlap=overlap,
        progress=False,
        num_workers=0,
        segment=None,
        split=True,
    )[0]
    sources = torch.stack([chunk, chunk, chunk, chunk])

    sources *= ref.std() + 1e-8
    sources += ref.mean()

    sources = fade(sources)
    sources = list(sources)

    stem = "vocals"
    forcus_stem = sources.pop(model_sources.index(stem))
    forcus_stem = forcus_stem.cpu().numpy()

    # Warning : after poping the stem, selected stem is no longer in the list 'sources'
    other_stem = torch.zeros_like(sources[0])
    for i in sources:
        other_stem += i
    other_stem = other_stem.cpu().numpy()

    forcus_stem = librosa.to_mono(forcus_stem)
    other_stem = librosa.to_mono(other_stem)

    stem_zeros = np.zeros(chunk_size + overlap_frames)
    print(
        forcus_stem_out.shape,
        stem_zeros.shape,
        forcus_stem.shape,
        start,
        end,
        data_length,
    )
    forcus_stem_out = np.concatenate([forcus_stem_out, stem_zeros])
    other_stem_out = np.concatenate([other_stem_out, stem_zeros])
    forcus_stem_out[start:end] += forcus_stem
    other_stem_out[start:end] += other_stem

    if start == 0:
        fade.fade_in_len = overlap_frames
        start += chunk_size
    else:
        start += chunk_size + overlap_frames

    end += chunk_size + overlap_frames
    if end >= data_length:
        fade.fade_out_len = 0


sf.write(
    "/workspace/streaming_translate_application/results/htdemucs_vocal.wav",
    forcus_stem_out,
    samplerate=model_samplerate,
)


sf.write(
    "/workspace/streaming_translate_application/results/htdemucs_no_vocals.wav",
    other_stem_out,
    samplerate=model_samplerate,
)
