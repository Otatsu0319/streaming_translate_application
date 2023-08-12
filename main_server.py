# code copy from https://github.com/TadaoYamaoka/StreamingWhisper/blob/master/WhisperServer/WhisperServer.py
import whisper
import socket
import threading
import queue
import numpy as np
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time

SAMPLE_RATE = 16000
INTERVAL = 10
BUFFER_SIZE = 4096

parser = argparse.ArgumentParser()
parser.add_argument('--address', default='127.0.0.1')
parser.add_argument('--port', type=int, default=50000)
parser.add_argument('--model', default='base')
args = parser.parse_args()

print('Loading model...')
model_whisper = whisper.load_model(args.model, download_root="/workspace/whisper/")
print('Done')

s = socket.socket(socket.AF_INET)
s.bind((args.address, args.port))
s.listen()

q = queue.Queue()
b = np.ones(100) / 100

options = whisper.DecodingOptions()

# model_path = "facebook/nllb-200-distilled-600M"
model_path = "facebook/nllb-200-distilled-1.3B"
# model_path = "facebook/nllb-200-3.3B"

tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="eng_Latn", use_fast=True)
model_nllb200 = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def recognize():
    while True:
        audio = q.get()
        if (audio ** 2).max() > 0.001:
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model_whisper.device)

            # detect the spoken language
            _, probs = model_whisper.detect_language(mel)

            # decode the audio
            result = whisper.decode(model_whisper, mel, options)

            # print the recognized text
            print(f'{max(probs, key=probs.get)}: {result.text}')
            recognize_text = result.text
            inputs = tokenizer(recognize_text, return_tensors="pt")
            
            translated_tokens = model_nllb200.generate(
                **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["jpn_Jpan"], max_length=150,
                num_beams=5,
            )
            print("transrate: ", tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])


th_recognize = threading.Thread(target=recognize, daemon=True)
th_recognize.start()

# start listening
while True:
    try:
        print('Listening...')
        cilent, address = s.accept()
        print(f'Connected from {address}')

        audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
        n = 0
        while True:
            while n < SAMPLE_RATE * INTERVAL:
                cilent.settimeout(INTERVAL)
                try:
                    data = cilent.recv(BUFFER_SIZE)
                    # print(time.time(), len(data), type(data), data)
                    
                except socket.timeout:
                    break
                finally:
                    cilent.settimeout(None)
                data = np.frombuffer(data, dtype=np.int16)
                # print(len(data), data)
                # cilent.close()
                # exit()
                audio[n:n+len(data)] = data.astype(np.float32) / 32768
                n += len(data)
            if n > 0:
                # find silent periods
                m = n * 4 // 5
                vol = np.convolve(audio[m:n] ** 2, b, 'same')
                m += vol.argmin()
                if m > 0:
                    q.put(audio[:m])

                audio_prev = audio
                audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
                audio[:n-m] = audio_prev[m:n]
                n = n-m
    except Exception as e:
        print(e)
    finally:
        cilent.close()