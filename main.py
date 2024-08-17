import logging

import streamer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # receiver = streamer.VBANStreamingReceiver("127.0.0.1", "Stream1", 6981)
    receiver = streamer.WavStreamReceiver("test.wav")

    for data in receiver.recv_generator():
        print(data[:5], len(data))
