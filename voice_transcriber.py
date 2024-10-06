import multiprocessing as mp

import numpy as np
from faster_whisper import WhisperModel
from llama_cpp import Llama

MODEL_LABEL = "large-v3"  # or "distil-large-v3"

PROMPT_JP2EN = "Translate this from Japanese to English:\nJapanese: {0}\nEnglish:"
PROMPT_EN2JP = "Translate this from English to Japanese:\nEnglish: {0}\nJapanese:"


mp.set_start_method('spawn', force=True)


def translate_process(queue):
    translator = Llama.from_pretrained(
        "mmnga/webbigdata-ALMA-7B-Ja-V2-gguf",
        "*q4_0.gguf",
        local_dir="/workspace/models",
        cache_dir="/workspace/models",
        n_gpu_layers=-1,
        verbose=False,
    )  # VRAM 6.7GB?

    while True:
        segment = queue.get()
        if segment is None:
            break
        print("[id:%d, p:%.02f] %s" % (segment.id, segment.avg_logprob, segment.text))
        output = translator(PROMPT_EN2JP.format(segment.text), max_tokens=128, stop=["\n"])
        print("translated: ", output['choices'][0]['text'])


class VoiceTranscriber:
    def __init__(self, speech_queue, translate=True):
        self.speech_queue = speech_queue
        # Run on GPU with FP16
        self.whisper_model = WhisperModel(MODEL_LABEL, device="cuda", compute_type="float16", download_root="../models")
        self.transcribe_log_probability_threshold = -0.4

        self.translate = translate
        self.text_queue = mp.Queue()
        if translate:
            self.p = mp.Process(target=translate_process, args=(self.text_queue,))
            self.p.start()

    def transcribe_thread(self):
        while True:
            speech = self.speech_queue.get()
            if speech is None:
                self.text_queue.put(None)
                self.p.join()
                break
            speech = np.array(speech)

            segments, _ = self.whisper_model.transcribe(
                speech, language="en", log_prob_threshold=self.transcribe_log_probability_threshold
            )
            for segment in segments:

                if self.translate:
                    self.text_queue.put(segment)
                else:
                    print("[id:%d, p:%.02f] %s" % (segment.id, segment.avg_logprob, segment.text))
