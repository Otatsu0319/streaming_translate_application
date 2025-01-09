import logging
import queue
import socket

import librosa
import numpy as np
import pyvban
import pyvban.subprotocols
import soundfile as sf
from pyvban.subprotocols import audio as pyvban_audio


class StreamReceiver:
    def __init__(
        self,
        current_sample_rate: int,
        current_channels: int,
        chunk_size: int,
        logger: logging.Logger = None,
    ):
        self._logger = logger if logger else logging.getLogger("StreamReceiver")

        self.current_sample_rate = current_sample_rate
        self.current_channels = current_channels
        self.chunk_size = chunk_size

        self._running = True

    def recv_generator(self):
        raise NotImplementedError()


class VBANStreamingReceiver(StreamReceiver):
    def __init__(
        self,
        sender_ip: str,
        stream_name: str,
        port: int,
        current_sample_rate: int = 16000,
        current_channels: int = 1,
        chunk_size: int = 512,
        logger: logging.Logger = None,
    ):
        self._logger = logger if logger else logging.getLogger(f"VBAN_Receiver_{sender_ip}_{port}_{stream_name}")
        super().__init__(current_sample_rate, current_channels, chunk_size, logger=self._logger)

        self._sender_ip = sender_ip
        self._stream_name = stream_name

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((sender_ip, port))

        self._buff = []
        self._first_check = False

    def _check_pyaudio(self, header: pyvban.subprotocols.audio.VBANAudioHeader):
        if pyvban_audio.const.VBANSampleRatesEnum2SR[header.sample_rate] != self.current_sample_rate:
            raise NotImplementedError("This sample rate is not supported")
        if header.channels != 1 and not self._first_check:
            raise NotImplementedError("This channels is not supported. set VBAudio channels to 1")
        if header.samples_per_frame > self.chunk_size:
            raise NotImplementedError(
                f"This chunk_size is not supported. Please specify a value greater than {header.samples_per_frame}"
            )

    def recv_once(self):
        try:
            data, addr = self._socket.recvfrom(pyvban.const.VBAN_PROTOCOL_MAX_SIZE)
            packet = pyvban.packet.VBANPacket(data)
            if packet.header:
                self._logger.info(
                    f"RVBAN {packet.header.sample_rate}Hz {packet.header.samples_per_frame}samp "
                    f"{packet.header.channels}chan Format:{packet.header.format} Codec:{packet.header.codec} "
                    f"Name:{packet.header.stream_name} Frame:{packet.header.frame_counter}"
                )

                if packet.header.sub_protocol != pyvban.const.VBANProtocols.VBAN_PROTOCOL_AUDIO:
                    self._logger.debug(f"Received non audio packet {packet}")
                    return
                if packet.header.stream_name != self._stream_name:
                    self._logger.debug(
                        f"Unexpected stream name \"{packet.header.stream_name}\" != \"{self._stream_name}\""
                    )
                    return
                if addr[0] != self._sender_ip:
                    self._logger.debug(f"Unexpected sender \"{addr[0]}\" != \"{self._sender_ip}\"")
                    return

                self._check_pyaudio(packet.header)

                return packet.data

        except Exception as e:
            self._logger.error(f"An exception occurred: {e}")
            raise e

    def recv_generator(self):
        while self._running:
            try:
                data = self.recv_once()

                data = np.frombuffer(data, dtype="int16")
                data = data / 32768.0
                self._buff.extend(data.tolist())

            except Exception as e:
                logging.error(f"An exception occurred: {e}")
                continue

            if len(self._buff) >= self.chunk_size:
                data = self._buff[: self.chunk_size]
                self._buff = self._buff[self.chunk_size :]

                if self.current_channels == 2:
                    yield np.array([data, data]).T
                else:
                    yield np.array(data)


class VBANStreamingSender:
    def __init__(
        self,
        receiver_ip: str,
        stream_name: str,
        port: int,
        data_queue: queue.Queue,
        sample_rate: int = 16000,
        channels: int = 1,
        logger: logging.Logger = None,
    ):
        self._logger = logger if logger else logging.getLogger(f"VBAN_Sender_{receiver_ip}_{port}_{stream_name}")

        self._receiver_ip = receiver_ip
        self._port = port
        self._stream_name = stream_name
        self._sample_rate = sample_rate
        self._vban_sample_rate = pyvban_audio.const.VBANSampleRatesSR2Enum[sample_rate]
        self._channels = channels

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._samples_per_frame = 206
        self._frame_counter = 0
        self._buff = []

        self.data_queue = data_queue

    def send_once(self, data: np.ndarray):
        try:

            header = pyvban.subprotocols.audio.VBANAudioHeader(
                sample_rate=self._vban_sample_rate,
                channels=self._channels,
                samples_per_frame=self._samples_per_frame,
                format=pyvban_audio.const.VBANBitResolution.VBAN_BITFMT_16_INT,
                codec=pyvban_audio.const.VBANCodec.VBAN_CODEC_PCM,
                stream_name=self._stream_name,
                frame_counter=self._frame_counter,
            )

            if len(data) != self._samples_per_frame:
                raise ValueError("Data size is not correct")

            packet = header.to_bytes() + data.tobytes()

            if len(packet) > pyvban.const.VBAN_PROTOCOL_MAX_SIZE:
                raise ValueError("Packet size is too large")

            self._frame_counter += 1
            self._socket.sendto(packet, (self._receiver_ip, self._port))
            self._logger.debug(f"Sent {len(data)} samples")

        except Exception as e:
            self._logger.error(f"An exception occurred: {e}")
            raise e

    def send_thread(self):
        while True:
            data = self.data_queue.get()
            if data is None:
                break
            data = data * 32768.0
            self._buff.extend(data.tolist())
            for _ in range(len(self._buff) // self._samples_per_frame - 1):
                data = self._buff[: self._samples_per_frame]
                self._buff = self._buff[self._samples_per_frame :]
                self.send_once(np.array(data, dtype="int16"))


class WavStreamReceiver(StreamReceiver):
    def __init__(
        self,
        filename: str,
        current_sample_rate: int = 16000,
        current_channels: int = 1,
        chunk_size: int = 512,
        logger: logging.Logger = None,
    ):
        self._logger = logger if logger else logging.getLogger(f"WAV_Receiver_{filename}")
        super().__init__(current_sample_rate, current_channels, chunk_size, logger=self._logger)

        self.data, sr = sf.read(filename, always_2d=True)
        if sr != self.current_sample_rate:
            self._logger.debug("This sample rate is not supported. Resampling...")
        if self.current_channels == 1:
            self._logger.debug("This channels is not supported. Convert to mono...")
            self.data = librosa.to_mono(self.data.T)
        self.data = librosa.resample(self.data, orig_sr=sr, target_sr=self.current_sample_rate)

    def recv_generator(self):
        i = 0
        while self._running:
            data = self.data[i : i + self.chunk_size]
            i += self.chunk_size
            if len(data) == 0:
                break
            yield data


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    receiver = VBANStreamingReceiver(
        "127.0.0.1", "Stream1", 6990, current_sample_rate=44100, current_channels=1, chunk_size=512
    )

    send_queue = queue.Queue()
    sender = VBANStreamingSender("192.168.1.206", "Stream2", 6980, data_queue=send_queue, sample_rate=44100, channels=1)
    import threading

    # while True:
    #     chunk = receiver.recv_once()
    #     print(len(chunk))
    #     data = np.frombuffer(chunk, dtype="int16")
    #     # data = data / 32768.0
    #     print(len(data))
    #     sender.send_once(data)
    # i = 0

    # i += 1
    # if i == 10:
    #     break
    # send_queue.put(None)

    # sender.send_thread()

    th = threading.Thread(target=sender.send_thread)
    th.start()
    for chunk in receiver.recv_generator():
        send_queue.put(chunk)
