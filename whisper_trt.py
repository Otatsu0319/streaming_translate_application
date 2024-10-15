# Modified from https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/whisper/run.py
# original license & copyright:
# # SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: Apache-2.0
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
import argparse
import json
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import soundfile
import tensorrt_llm
import torch
import torch.nn.functional as F
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.bindings import GptJsonConfig
from tensorrt_llm.runtime import ModelRunnerCpp
from whisper import tokenizer
from whisper.audio import HOP_LENGTH, N_FFT, N_SAMPLES, SAMPLE_RATE  # , CHUNK_LENGTH
from whisper.normalizers import EnglishTextNormalizer

tensorrt_llm.logger.set_level('warning')


def load_audio_wav_format(wav_path):
    # make sure audio in .wav format
    assert wav_path.endswith('.wav'), f"Only support .wav format, but got {wav_path}"
    waveform, sample_rate = soundfile.read(wav_path)
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
    # assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    return waveform, sample_rate


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    return_duration: bool = False,
    mel_filters_dir: str = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80 or 128, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio, _ = load_audio_wav_format(audio)

        assert isinstance(audio, np.ndarray), f"Unsupported audio type: {type(audio)}"
        duration = audio.shape[-1] / SAMPLE_RATE
        audio = pad_or_trim(audio, N_SAMPLES)
        audio = audio.astype(np.float32)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    mel_128 = librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)
    filters = torch.from_numpy(mel_128).to(device)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    if return_duration:
        return log_spec, duration
    else:
        return log_spec


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='warning')
    parser.add_argument('--engine_dir', type=str, default='whisper_large_v3')
    parser.add_argument('--results_dir', type=str, default='tmp')
    parser.add_argument('--assets_dir', type=str, default='./assets')
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument('--name', type=str, default="librispeech_dummy_benchmark")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16'])
    parser.add_argument('--accuracy_check', action='store_true', help="only for CI test")
    parser.add_argument('--use_py_session', action='store_true', help="use python session or cpp session")
    return parser.parse_args()


def read_config(component, engine_dir):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config


class WhisperTRTLLM(object):
    def __init__(self, engine_dir, debug_mode=False, use_py_session=False):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)
        encoder_config = read_config('encoder', engine_dir)
        decoder_config = read_config('decoder', engine_dir)
        self.n_mels = encoder_config['n_mels']
        self.num_languages = encoder_config['num_languages']
        is_multilingual = decoder_config['vocab_size'] >= 51865
        if is_multilingual:
            tokenizer_name = "multilingual"
        else:
            tokenizer_name = "gpt2"
        self.tokenizer = tokenizer.get_encoding(tokenizer_name, self.num_languages)
        self.eot_id = self.tokenizer.encode("<|endoftext|>", allowed_special=self.tokenizer.special_tokens_set)[0]

        json_config = GptJsonConfig.parse_file(engine_dir / 'decoder' / 'config.json')
        assert json_config.model_config.supports_inflight_batching
        runner_kwargs = dict(
            engine_dir=engine_dir,
            is_enc_dec=True,
            max_batch_size=16,
            max_input_len=3000,
            max_output_len=96,
            max_beam_width=4,
            debug_mode=debug_mode,
            kv_cache_free_gpu_memory_fraction=0.9,
        )
        self.model_runner_cpp = ModelRunnerCpp.from_dir(**runner_kwargs)
        self.use_py_session = use_py_session

    def process_batch(
        self,
        mel,
        mel_input_lengths,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        num_beams=1,
        max_new_tokens=96,
    ):
        prompt_id = self.tokenizer.encode(text_prefix, allowed_special=self.tokenizer.special_tokens_set)
        prompt_id = torch.tensor(prompt_id)
        batch_size = mel.shape[0]
        decoder_input_ids = prompt_id.repeat(batch_size, 1)

        with torch.no_grad():
            outputs = self.model_runner_cpp.generate(
                batch_input_ids=decoder_input_ids,
                encoder_input_features=mel.transpose(1, 2),
                encoder_output_lengths=mel_input_lengths // 2,
                max_new_tokens=max_new_tokens,
                end_id=self.eot_id,
                pad_id=self.eot_id,
                num_beams=num_beams,
                output_sequence_lengths=True,
                return_dict=True,
            )
            torch.cuda.synchronize()
            output_ids = outputs['output_ids'].cpu().numpy().tolist()
        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.decode(output_ids[i][0]).strip()
            texts.append(text)
        return texts


def decode_wav_file(
    input_file_path,
    model,
    text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    dtype='float16',
    batch_size=1,
    num_beams=1,
    normalizer=None,
    mel_filters_dir=None,
):
    mel, total_duration = log_mel_spectrogram(
        input_file_path, model.n_mels, device='cuda', return_duration=True, mel_filters_dir=mel_filters_dir
    )
    mel = mel.type(str_dtype_to_torch(dtype))
    mel = mel.unsqueeze(0)
    # repeat the mel spectrogram to match the batch size
    mel = mel.repeat(batch_size, 1, 1)
    # TODO: use the actual input_lengths rather than padded input_lengths
    feature_input_lengths = torch.full((mel.shape[0],), mel.shape[2], dtype=torch.int32, device=mel.device)
    predictions = model.process_batch(mel, feature_input_lengths, text_prefix, num_beams)
    prediction = predictions[0]

    # remove all special tokens in the prediction
    prediction = re.sub(r'<\|.*?\|>', '', prediction)
    if normalizer:
        prediction = normalizer(prediction)
    print(f"prediction: {prediction}")
    results = [(0, [""], prediction.split())]
    return results, total_duration


if __name__ == '__main__':
    args = parse_arguments()
    model = WhisperTRTLLM("/mnt/wsl/workspace/TensorRT-LLM/examples/whisper/whisper_large_v3_trt", use_py_session=True)
    normalizer = EnglishTextNormalizer()

    start_time = time.time()
    results, total_duration = decode_wav_file(
        "/mnt/wsl/workspace/streaming_translate_application/results/htdemucs_vocal_16k.wav",
        model,
        dtype=args.dtype,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        mel_filters_dir=args.assets_dir,
        normalizer=normalizer,
    )
    elapsed = time.time() - start_time
    results = sorted(results)

    print(results)

    rtf = elapsed / total_duration
    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration/3600:.2f} hours)\n"
    s += f"processing time: {elapsed:.3f} seconds " f"({elapsed/3600:.2f} hours)\n"
    s += f"batch size: {args.batch_size}\n"
    s += f"num_beams: {args.num_beams}\n"
    print(s)

    del model
