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
from subprocess import CalledProcessError, run
from typing import Optional, Union

import librosa
import numpy as np
import soundfile
import tensorrt_llm
import tensorrt_llm.logger as logger
import torch
import torch.nn.functional as F
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.bindings import GptJsonConfig, KVCacheType
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo
from whisper import tokenizer
from whisper.normalizers import EnglishTextNormalizer

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

from whisper.audio import HOP_LENGTH, N_FFT, N_SAMPLES, SAMPLE_RATE  # , CHUNK_LENGTH


# MARK: whisper_utils.py
def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le", "-ac",
        "1", "-acodec", "pcm_s16le", "-ar",
        str(sr), "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


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
            if audio.endswith('.wav'):
                audio, _ = load_audio_wav_format(audio)
            else:
                audio = load_audio(audio)
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


# MARK: run.py
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


def remove_tensor_padding(input_tensor, input_tensor_lengths=None, pad_value=0):
    if input_tensor.dim() == 2:
        # Text tensor case: batch, seq_len
        assert torch.all(input_tensor[:, 0] != pad_value), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    elif input_tensor.dim() == 3:
        # Audio tensor case: batch, seq_len, feature_len
        assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"
        batch_size, seq_len, feature_len = input_tensor.shape

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(batch_size):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length, :])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)

    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions")

    return output_tensor


def read_config(component, engine_dir):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config


class WhisperEncoding:

    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)
        config = read_config('encoder', engine_dir)
        self.n_mels = config['n_mels']
        self.dtype = config['dtype']
        self.num_languages = config['num_languages']
        self.encoder_config = config

    def get_session(self, engine_dir):
        serialize_path = engine_dir / 'encoder' / 'rank0.engine'
        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())
        return session

    def get_audio_features(self, mel, mel_input_lengths, encoder_downsampling_factor=2):
        if self.encoder_config['plugin_config']['remove_input_padding']:
            # mel B,D,T -> B,T,D -> BxT, D
            mel = mel.transpose(1, 2)
            mel = remove_tensor_padding(mel, mel_input_lengths)

        inputs = OrderedDict()
        inputs['input_features'] = mel
        inputs['input_lengths'] = mel_input_lengths

        output_list = [
            TensorInfo('input_features', str_dtype_to_trt(self.dtype), mel.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'), mel_input_lengths.shape),
        ]

        output_info = (self.session).infer_shapes(output_list)

        logger.debug(f'output info {output_info}')
        outputs = {
            t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device='cuda') for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
        encoder_output = outputs['encoder_output']
        encoder_output_lengths = mel_input_lengths // encoder_downsampling_factor

        return encoder_output, encoder_output_lengths


class WhisperDecoding:

    def __init__(self, engine_dir, runtime_mapping, debug_mode=False):

        self.decoder_config = read_config('decoder', engine_dir)
        self.decoder_generation_session = self.get_session(engine_dir, runtime_mapping, debug_mode)

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        serialize_path = engine_dir / 'decoder' / 'rank0.engine'
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config['max_batch_size'],
            max_beam_width=self.decoder_config['max_beam_width'],
            num_heads=self.decoder_config['num_attention_heads'],
            num_kv_heads=self.decoder_config['num_attention_heads'],
            hidden_size=self.decoder_config['hidden_size'],
            vocab_size=self.decoder_config['vocab_size'],
            cross_attention=True,
            num_layers=self.decoder_config['num_hidden_layers'],
            gpt_attention_plugin=self.decoder_config['plugin_config']['gpt_attention_plugin'],
            remove_input_padding=self.decoder_config['plugin_config']['remove_input_padding'],
            kv_cache_type=(
                KVCacheType.PAGED
                if self.decoder_config['plugin_config']['paged_kv_cache'] is True
                else KVCacheType.CONTINUOUS
            ),
            has_position_embedding=self.decoder_config['has_position_embedding'],
            dtype=self.decoder_config['dtype'],
            has_token_type_embedding=False,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config, decoder_engine_buffer, runtime_mapping, debug_mode=debug_mode
        )

        return decoder_generation_session

    def generate(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_max_input_length,
        encoder_input_lengths,
        eot_id,
        max_new_tokens=40,
        num_beams=1,
    ):
        batch_size = decoder_input_ids.shape[0]
        decoder_input_lengths = torch.tensor(
            [decoder_input_ids.shape[-1] for _ in range(decoder_input_ids.shape[0])], dtype=torch.int32, device='cuda'
        )
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = torch.ones([batch_size, 1, encoder_max_input_length]).int().cuda()

        # generation config
        sampling_config = SamplingConfig(end_id=eot_id, pad_id=eot_id, num_beams=num_beams)
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_max_input_length,
        )

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        if self.decoder_config['plugin_config']['remove_input_padding']:
            # 50256 is the index of <pad> for all whisper models' decoder
            WHISPER_PAD_TOKEN_ID = 50256
            decoder_input_ids = remove_tensor_padding(decoder_input_ids, pad_value=WHISPER_PAD_TOKEN_ID)
            if encoder_outputs.dim() == 3:
                encoder_output_lens = torch.full(
                    (encoder_outputs.shape[0],), encoder_outputs.shape[1], dtype=torch.int32, device='cuda'
                )

                encoder_outputs = remove_tensor_padding(encoder_outputs, encoder_output_lens)
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids


class WhisperTRTLLM(object):

    def __init__(self, engine_dir, debug_mode=False, assets_dir=None, use_py_session=False):
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
            # assert (
            #     Path(assets_dir) / "multilingual.tiktoken"
            # ).exists(), "multilingual.tiktoken file is not existed in assets_dir"
        else:
            tokenizer_name = "gpt2"
            # assert (Path(assets_dir) / "gpt2.tiktoken").exists(), "gpt2.tiktoken file is not existed in assets_dir"
        # self.tokenizer = get_tokenizer(
        #     name=tokenizer_name,
        #     num_languages=self.num_languages,
        #     tokenizer_dir=assets_dir,
        # )
        self.tokenizer = tokenizer.get_encoding(tokenizer_name, self.num_languages)
        self.eot_id = self.tokenizer.encode("<|endoftext|>", allowed_special=self.tokenizer.special_tokens_set)[0]
        if use_py_session:
            self.encoder = WhisperEncoding(engine_dir)
            self.decoder = WhisperDecoding(engine_dir, runtime_mapping, debug_mode=debug_mode)
        else:
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
        if self.use_py_session:
            encoder_output, encoder_output_lengths = self.encoder.get_audio_features(mel, mel_input_lengths)
            encoder_max_input_length = torch.max(encoder_output_lengths).item()
            output_ids = self.decoder.generate(
                decoder_input_ids,
                encoder_output,
                encoder_max_input_length,
                encoder_output_lengths,
                self.eot_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        else:
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


def collate_wrapper(batch):
    speeches, durations, labels, ids = [], [], [], []
    for item in batch:
        speech = item["audio"]["array"]
        duration = speech.shape[-1]
        speech = pad_or_trim(speech, N_SAMPLES)
        speech = speech.astype(np.float32)
        speech = torch.from_numpy(speech)
        speeches.append(speech)
        durations.append(duration)
        labels.append(item["text"])
        ids.append(item["id"])
    return speeches, durations, labels, ids


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
