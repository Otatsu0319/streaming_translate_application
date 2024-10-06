# 音声分離を行うdemucsモデルをパッケージをインポートせずに使えるように、jitコンパイルを行う
# Jit compile so that you can use the demucs model for speech separation without importing the package
#
# 使用モデル(using models): hybrid_transformer (Demucs v4)

import inspect
import warnings

import torch
from demucs import states

URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"


def get_model(url=URL):
    pkg = torch.hub.load_state_dict_from_url(url, model_dir="./demucs_models", map_location='cpu', check_hash=True)

    klass = pkg["klass"]
    args = pkg["args"]
    kwargs = pkg["kwargs"]
    state = pkg["state"]

    sig = inspect.signature(klass)

    for key in list(kwargs):
        if key not in sig.parameters:
            warnings.warn("Dropping inexistant parameter " + key)
            del kwargs[key]
    model = klass(*args, **kwargs)
    states.set_state(model, state)

    return model


if __name__ == "__main__":
    model = get_model()
    model.to("cuda")
    model.eval()

    segment = model.segment
    segment_length: int = int(model.samplerate * segment)
    example_tensor = torch.rand(
        (1, model.audio_channels, segment_length), device="cuda"
    )  # (1, 2, 343980) # Batch, Channels, Times(audio signal)

    with torch.no_grad():
        torch.jit.trace(model, example_inputs=example_tensor, check_trace=False).save("./demucs_models/demucs_ts.pt")

    # get model info (need to inference)
    print(f"model_samplerate={model.samplerate}")
    print(f"model_audio_channels={model.audio_channels}")
    print(f"model_souces={model.sources}")
    print(f"model_segment={model.segment}")
