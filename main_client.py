import socket
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
BUFFER_SIZE = 4096//2 # int16 = 2 bytes

device_list = sd.query_devices()
# sd.default.device = [5, 13]
# sd.default.device = [5, 10]
sd.default.device = [19, 27]

print(device_list)

sd.default.samplerate = SAMPLE_RATE #普通41000Hzか48000Hz
sd.default.channels = 1
sd.default.dtype = 'int16'


if __name__ == "__main__":
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 50000))
    print("server connected")
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    def callback(indata, frames, time, status):
        size = s.send(indata)

    with sd.InputStream(blocksize=BUFFER_SIZE, callback=callback):
        print('#' * 40)
        print('press Return to quit')
        print('#' * 40)
        input()
        
    s.close()
    print("server disconneected")

