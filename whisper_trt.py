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
import os
import re
import time
from collections import OrderedDict
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch
import torch.nn.functional as F
from whisper.audio import HOP_LENGTH, N_FFT, N_SAMPLES  # , SAMPLE_RATE, CHUNK_LENGTH

# from whisper.normalizers import EnglishTextNormalizer

os.environ['TLLM_LOG_LEVEL'] = 'ERROR'

if os.environ['TLLM_LOG_LEVEL'] in ["INTERNAL_ERROR", "ERROR", "WARNING", "INFO", "VERBOSE", "DEBUG"]:
    from tensorrt_llm import logger
    from tensorrt_llm._utils import str_dtype_to_torch
    from tensorrt_llm.bindings import GptJsonConfig
    from tensorrt_llm.runtime import ModelRunnerCpp
    from whisper import tokenizer


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


class WhisperTRTLLM(object):
    def __init__(self, engine_dir, debug_mode=False, device="cuda"):
        engine_dir = Path(engine_dir)
        encoder_config = self.read_config('encoder', engine_dir)
        decoder_config = self.read_config('decoder', engine_dir)
        self.n_mels = encoder_config['n_mels']
        self.num_languages = encoder_config['num_languages']
        self.batch_size = 1

        is_multilingual = decoder_config['vocab_size'] >= 51865
        if is_multilingual:
            tokenizer_name = "multilingual"
        else:
            tokenizer_name = "gpt2"
        self.tokenizer = tokenizer.get_encoding(tokenizer_name, self.num_languages)
        self.eot_id = self.tokenizer.encode("<|endoftext|>", allowed_special=self.tokenizer.special_tokens_set)[0]

        # self.normalizer = EnglishTextNormalizer() # 大文字小文字、記号等が正規化され比較しやすい形になる (=読みづらい)

        self.device = device
        self.mel_filter = torch.from_numpy(
            librosa.filters.mel(sr=16000, n_fft=400, n_mels=self.n_mels, dtype=np.float64)
        ).to(self.device)
        self.window = torch.hann_window(N_FFT).to(self.device)

        json_config = GptJsonConfig.parse_file(engine_dir / 'decoder' / 'config.json')
        assert json_config.model_config.supports_inflight_batching
        runner_kwargs = dict(
            engine_dir=engine_dir,
            is_enc_dec=True,
            max_batch_size=8,
            max_input_len=3000,
            max_output_len=96,
            max_beam_width=1,
            debug_mode=debug_mode,
            # kv_cache_free_gpu_memory_fraction=0.1,
            kv_cache_free_gpu_memory_fraction=0.1,
            cross_kv_cache_fraction=0.5,
        )
        self.model_runner_cpp = ModelRunnerCpp.from_dir(**runner_kwargs)

    def read_config(self, component, engine_dir):
        config_path = engine_dir / component / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_config = OrderedDict()
        model_config.update(config['pretrained_config'])
        model_config.update(config['build_config'])
        return model_config

    # @torch.jit.script
    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the log-Mel spectrogram

        Parameters
        ----------
        audio: torch.Tensor, shape = (audio_length,)
            NumPy array containing the audio waveform in 16 kHz

        padding: int
            Number of zero samples to pad to the right

        device: Optional[Union[str, torch.device]]
            If given, the audio tensor is moved to this device before STFT

        Returns
        -------
        torch.Tensor, shape = (80 or 128, n_frames)
            A Tensor that contains the Mel spectrogram
        """

        if audio.shape[-1] > N_SAMPLES:
            audio = audio.index_select(dim=-1, index=torch.arange(N_SAMPLES, device=audio.device))
            logger.warning(f"Audio length exceeds; input length: {audio.shape[-1]}, truncating to {N_SAMPLES}")
        elif audio.shape[-1] < N_SAMPLES:
            pad_widths = [(0, 0)] * audio.ndim
            pad_widths[-1] = (0, N_SAMPLES - audio.shape[-1])
            audio = F.pad(audio, [pad for sizes in pad_widths[::-1] for pad in sizes])

        stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=self.window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filter @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    def transcribe(self, audio: np.ndarray, language="en", log_prob_threshold=-float("inf")):
        audio = torch.from_numpy(audio).to(self.device)
        mel = self.log_mel_spectrogram(audio)
        mel = mel.type(str_dtype_to_torch("float16"))
        mel = mel.unsqueeze(0)
        mel = mel.repeat(self.batch_size, 1, 1)
        feature_input_lengths = torch.full((mel.shape[0],), mel.shape[2], dtype=torch.int32, device=mel.device)

        mel_input_lengths = feature_input_lengths
        text_prefix = f"<|startoftranscript|><|{language}|><|transcribe|><|notimestamps|>"
        num_beams = 1
        max_new_tokens = 96

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
                output_cum_log_probs=True,
                output_log_probs=True,
                return_dict=True,
            )
            torch.cuda.synchronize()
            output_ids = outputs['output_ids'].cpu().numpy().tolist()
            cum_log_probs = outputs['cum_log_probs'].cpu().numpy().tolist()

        predictions = []
        probs = []
        for i in range(len(output_ids)):
            if cum_log_probs[i][0] < log_prob_threshold:
                continue
            text = self.tokenizer.decode(output_ids[i][0]).strip()
            text = re.sub(r'<\|.*?\|>', '', text)
            # if self.normalizer:
            #     text = self.normalizer(text)
            predictions.append(text)
            probs.append(cum_log_probs[i][0])
        return predictions, probs


if __name__ == '__main__':
    args = parse_arguments()
    model = WhisperTRTLLM("/mnt/wsl/workspace/streaming_translate_application/models/whisper_trt_engine")
    wav, sr = load_audio_wav_format("/mnt/wsl/workspace/streaming_translate_application/results/htdemucs_vocal_16k.wav")
    start_time = time.time()
    results, probs = model.transcribe(wav, language="en", log_prob_threshold=-float("inf"))
    elapsed = time.time() - start_time

    print(results)
    total_duration = len(wav) / sr
    rtf = elapsed / total_duration
    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration / 3600:.2f} hours)\n"
    s += f"processing time: {elapsed:.3f} seconds " f"({elapsed / 3600:.2f} hours)\n"
    s += f"batch size: {args.batch_size}\n"
    s += f"num_beams: {args.num_beams}\n"
    print(s)

    del model
