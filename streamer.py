import logging
import socket

import numpy as np
import pyvban
import pyvban.subprotocols
from pyvban.subprotocols import audio as pyvban_audio


class VBANStreamingReceiver:
    def __init__(self, sender_ip: str, stream_name: str, port: int, logger: logging.Logger = None):
        self._logger = logger if logger else logging.getLogger(f"VBAN_Receiver_{sender_ip}_{port}_{stream_name}")

        self._sender_ip = sender_ip
        self._stream_name = stream_name

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("0.0.0.0", port))

        # TODO: make this dynamic
        self.current_sample_rate = 48000
        self.current_channles = 1
        self.chunk_size = 2048

        self._running = True
        self._buff = []

    def _check_pyaudio(self, header):
        if pyvban_audio.const.VBANSampleRatesEnum2SR[header.sample_rate] != self.current_sample_rate:
            raise NotImplementedError("This sample rate is not supported")
        if header.channels != self.current_channles:
            raise NotImplementedError(f"This channels is not supported. set channels to {self.current_channles}")
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

    def recv_generator(self):
        while self._running:
            try:
                data = self.recv_once()
            except Exception as e:
                logging.error(f"An exception occurred: {e}")
                continue

            self._buff.extend(np.frombuffer(data, dtype="int16").tolist())

            if len(self._buff) >= self.chunk_size:
                data = self._buff[: self.chunk_size]
                self._buff = self._buff[self.chunk_size :]
                yield data

        yield self._buff
