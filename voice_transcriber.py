import multiprocessing as mp
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# from llama_cpp import Llama
from gemma2_trt import Gemma2

# from faster_whisper import WhisperModel
from whisper_trt import WhisperTRTLLM

# import torch_tensorrt


MODEL_LABEL = "large-v3"  # or "distil-large-v3"

PROMPT_JP2EN = "Translate this from Japanese to English:\nJapanese: {0}"
# PROMPT_EN2JP = "Translate this from English to Japanese:\nEnglish: {0}"
PROMPT_EN2JP = "次の発言を英語から日本語に翻訳してください:\n\n{0}"
"/workspace/streaming_translate_application/models/gemma-2-2b-jpn-it"

mp.set_start_method('spawn', force=True)


def translate_process(queue):
    # translator = Gemma2("/workspace/streaming_translate_application/models/gemma-2-2b-jpn-it_bf16")
    # translator = pipeline('translation', model='staka/fugumt-en-ja', device="cuda")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-jpn-it", cache_dir="./models")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-jpn-it", device_map="auto", torch_dtype=torch.bfloat16, cache_dir="./models"
    )

    while True:
        segment = queue.get()
        if segment is None:
            break
        # print("[id:%d, p:%.02f] %s" % (segment.id, segment.avg_logprob, segment.text))
        st = time.time()
        # messages = [{"role": "user", "content": PROMPT_EN2JP.format(segment)}]
        # inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # inputs += "**日本語訳**:\n\n"
        inputs = "\n".join(
            [
                "<bos><start_of_turn>user",
                "次の発言を英語から日本語に翻訳してください:",
                "",
                f"{segment}<end_of_turn>",
                "<start_of_turn>model",
                "**日本語訳**:\n\n",
            ]
        )
        # print(inputs)
        inputs = tokenizer.encode(inputs, return_tensors="pt").to(model.device)

        outputs = model.generate(inputs, max_new_tokens=256, stop_strings="\n", tokenizer=tokenizer)
        generated_text = tokenizer.batch_decode(outputs[:, inputs.shape[1] :], skip_special_tokens=True)[0]

        # output = translator(segment)
        # print("translated: ", output['choices'][0]['text'])
        print(f"time: {time.time() - st} translated: {generated_text.strip()}")


class VoiceTranscriber:
    def __init__(self, speech_queue, translate=True):
        self.speech_queue = speech_queue
        # Run on GPU with FP16
        # self.whisper_model = WhisperModel(
        #     MODEL_LABEL,
        #     device="cuda",
        #     compute_type="float16",
        #     download_root="../models",
        # )
        self.whisper_model = WhisperTRTLLM("/workspace/streaming_translate_application/models/whisper_trt_engine")
        self.transcribe_log_probability_threshold = -30  # -0.4

        self.translate = translate
        self.text_queue = mp.Queue()
        if self.translate:
            self.p = mp.Process(target=translate_process, args=(self.text_queue,))
            self.p.start()

    def transcribe_thread(self):
        while True:
            speech = self.speech_queue.get()
            if speech is None:
                self.text_queue.put(None)
                if self.translate:
                    self.p.join()
                break
            speech = np.array(speech)
            st = time.time()
            segments, probs = self.whisper_model.transcribe(
                speech, language="ja", log_prob_threshold=self.transcribe_log_probability_threshold
            )
            print("transcribe time: ", time.time() - st)
            for i in range(len(segments)):

                if self.translate:
                    self.text_queue.put(segments[i])
                else:
                    print("[id:%d, p:%.02f] %s" % (i, probs[i], segments[i]))


if __name__ == "__main__":

    text_queue = mp.Queue()
    p = mp.Process(target=translate_process, args=(text_queue,))
    p.start()

    text = open(os.path.join(os.path.dirname(__file__), "assets/sample_en.txt"), "r").read().split("\n")

    for i in range(len(text)):
        text_queue.put(text[i])
    text_queue.put(None)
    p.join()
