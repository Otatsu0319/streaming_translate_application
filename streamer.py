import logging
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
        self._socket.bind(("0.0.0.0", port))

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
            except Exception as e:
                logging.error(f"An exception occurred: {e}")
                continue

            data = np.frombuffer(data, dtype="int16")
            data = data / 32768.0
            self._buff.extend(data.tolist())

            if len(self._buff) >= self.chunk_size:
                data = self._buff[: self.chunk_size]
                self._buff = self._buff[self.chunk_size :]

                if self.current_channels == 2:
                    yield np.array([data, data]).T
                else:
                    yield np.array(data)


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
