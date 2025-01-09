import multiprocessing as mp
import os
import time

import numpy as np
import rich
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from faster_whisper import WhisperModel
from whisper_trt import WhisperTRTLLM

# from llama_cpp import Llama
# from gemma2_trt import Gemma2


# import torch_tensorrt


MODEL_LABEL = "large-v3"  # or "distil-large-v3"

PROMPT_JP2EN = "Translate this from Japanese to English:\nJapanese: {0}"
PROMPT_EN2JP = "次の発言を英語から日本語に翻訳してください:\n\n{0}"

mp.set_start_method('spawn', force=True)


def translate_process(queue):
    # translator = Gemma2("/workspace/streaming_translate_application/models/gemma-2-2b-jpn-it_bf16")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-jpn-it", cache_dir="./models")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-jpn-it", device_map="auto", torch_dtype=torch.bfloat16, cache_dir="./models"
    )

    while True:
        packet = queue.get()
        if packet is None:
            break
        seg_id, prob, segment = packet
        # print("[id:%d, p:%.02f] %s" % (segment.id, segment.avg_logprob, segment.text))
        # st = time.time()
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
        # print(f"time: {time.time() - st} translated: {generated_text.strip()}")
        rich.print(f"id: {seg_id:08}, prob: {prob:.02f}, transcribe: [cyan]{segment}[/cyan]")
        rich.print(f"                          translated: [green]{generated_text.strip()}[/green]")


class VoiceTranscriber:
    def __init__(self, speech_queue, text_queue=None):
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

        if text_queue is not None:
            self.translate = True
            self.text_queue = text_queue

    def transcribe_thread(self):
        while True:
            speech = self.speech_queue.get()
            if speech is None:
                break
            speech = np.array(speech)
            st = time.time()
            segments, probs = self.whisper_model.transcribe(
                speech, language="en", log_prob_threshold=self.transcribe_log_probability_threshold
            )
            print("transcribe time: ", time.time() - st)
            for i in range(len(segments)):

                if self.translate:
                    packet = (i, probs[i], segments[i])
                    self.text_queue.put(packet)
                else:
                    print("[id:%d, p:%.02f] %s" % (i, probs[i], segments[i]))


if __name__ == "__main__":
    sound_queue = mp.Queue()
    text_queue = mp.Queue()

    p = mp.Process(target=translate_process, args=(text_queue,))
    p.start()

    vt = VoiceTranscriber(sound_queue, text_queue=text_queue)
    vt = VoiceTranscriber(sound_queue)
    from threading import Thread

    tr_th = Thread(target=vt.transcribe_thread)
    tr_th.start()

    text = open(os.path.join(os.path.dirname(__file__), "assets/sample_en.txt"), "r").read().split("\n")

    for i in range(len(text)):
        packet = (i, 0.8, text[i])
        text_queue.put(packet)
    text_queue.put(None)
    sound_queue.put(None)

    p.join()
    tr_th.join()
