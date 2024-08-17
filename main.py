import logging

import streamer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    vban = streamer.VBANStreamingReceiver("127.0.0.1", "Stream1", 6981)
    data = vban.recv_once()

    for data in vban.recv_generator():
        print(data[:5], len(data))
