import pyvban
from pyvban.subprotocols import audio as pyvban_audio
import socket
import logging

import pyvban.subprotocols


class VBAN_Recv:
    def __init__(self, sender_ip: str, stream_name: str, port: int):
        self._logger = logging.getLogger(f"VBAN_Receiver_{sender_ip}_{port}_{stream_name}")
        self._logger.info("Hellow world")

        self._sender_ip = sender_ip
        self._stream_name = stream_name

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("0.0.0.0", port))

        self._running = True
        
        self._stream = None # TODO: make

    def _check_pyaudio(self, header):
        if pyvban_audio.const.VBANSampleRatesEnum2SR[header.sample_rate] != self._current_pyaudio_config["rate"] or header.channels != self._current_pyaudio_config["channels"]:
            self._logger.info("Re-Configuring PyAudio")
            self._current_pyaudio_config["rate"] = pyvban_audio.const.VBANSampleRatesEnum2SR[header.sample_rate]
            self._current_pyaudio_config["channels"] = header.channels
            self._stream.close()
            self._stream = self._p.open(
                format=self._p.get_format_from_width(2),
                channels=self._current_pyaudio_config["channels"],
                rate=self._current_pyaudio_config["rate"],
                output=True,
                output_device_index=self._device_index
            )

    def run_once(self):
        try:
            data, addr = self._socket.recvfrom(pyvban.const.VBAN_PROTOCOL_MAX_SIZE)
            packet = pyvban.packet.VBANPacket(data)
            if packet.header:
                if packet.header.sub_protocol != pyvban.const.VBANProtocols.VBAN_PROTOCOL_AUDIO:
                    self._logger.debug(f"Received non audio packet {packet}")
                    return
                if packet.header.stream_name != self._stream_name:
                    self._logger.debug(f"Unexpected stream name \"{packet.header.stream_name}\" != \"{self._stream_name}\"")
                    return
                if addr[0] != self._sender_ip:
                    self._logger.debug(f"Unexpected sender \"{addr[0]}\" != \"{self._sender_ip}\"")
                    return

                # self._check_pyaudio(packet.header)

                return packet.data
                # self._stream.write(packet.data)
        except Exception as e:
            self._logger.error(f"An exception occurred: {e}")

    def run(self):
        self._running = True
        while self._running:
            self.run_once()
        self.stop()

    def stop(self):
        self._running = False
        self._stream.close()
        self._stream = None

if __name__ == "__main__":
    vban = VBAN_Recv("127.0.0.1", "Stream1", 6981)
    print(vban.run_once())