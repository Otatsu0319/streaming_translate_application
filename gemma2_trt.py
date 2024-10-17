import os

import torch
import transformers

os.environ['TLLM_LOG_LEVEL'] = 'ERROR'

if os.environ['TLLM_LOG_LEVEL'] in ["INTERNAL_ERROR", "ERROR", "WARNING", "INFO", "VERBOSE", "DEBUG"]:
    from tensorrt_llm.runtime import ModelRunnerCpp


class Gemma2:
    def __init__(self, engine_dir, mode="en2ja"):
        runner_kwargs = dict(
            engine_dir=engine_dir,
            is_enc_dec=False,
            max_batch_size=8,
            max_input_len=3000,
            max_output_len=96,
            max_beam_width=1,
            debug_mode=False,
            kv_cache_free_gpu_memory_fraction=0.1,
        )
        self.model_runner_cpp = ModelRunnerCpp.from_dir(**runner_kwargs)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-2-2b-jpn-it")

    def __call__(self, text, temperature=0.7):
        messages = [
            {"role": "user", "content": text},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to("cuda")
        batch_input_ids = [inputs["input_ids"][0]]

        with torch.no_grad():
            outputs = self.model_runner_cpp.generate(
                batch_input_ids=batch_input_ids,
                max_new_tokens=96,
                temperature=temperature,
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.pad_token_id,
                return_dict=True,
            )
            torch.cuda.synchronize()
        output_ids = outputs["output_ids"][:, 0, inputs['input_ids'].shape[1] :]
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
