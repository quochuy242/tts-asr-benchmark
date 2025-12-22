import sys
from pathlib import Path
import soundfile as sf
import sherpa_onnx
import sounddevice as sd
from datetime import datetime as dt


class ASRModel:
    def __init__(self, use_int8: bool = False):
        if use_int8:
            self.model_path = Path(
                "./sherpa-onnx/sherpa-onnx-zipformer-vi-int8-2025-04-20"
            )
            self.encoder_path = self.model_path / "encoder-epoch-12-avg-8.int8.onnx"
            self.joiner_path = self.model_path / "joiner-epoch-12-avg-8.int8.onnx"
        else:
            self.model_path = Path("./sherpa-onnx/sherpa-onnx-zipformer-vi-2025-04-20")
            self.encoder_path = self.model_path / "encoder-epoch-12-avg-8.onnx"
            self.joiner_path = self.model_path / "joiner-epoch-12-avg-8.onnx"

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(self.encoder_path),
            decoder=str(self.model_path / "decoder-epoch-12-avg-8.onnx"),
            joiner=str(self.joiner_path),
            tokens=str(self.model_path / "tokens.txt"),
            num_threads=12,
            sample_rate=16000,
        )

    def transcribe_from_file(self, wav_path: Path | str) -> str:
        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        audio = audio[:, 0]  # only use the first channel

        # audio is a 1-D float32 numpy array normalized to the range [-1, 1]
        # sample_rate does not need to be 16000 Hz

        start_t = dt.now()

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self.recognizer.decode_stream(stream)

        end_t = dt.now()
        elapsed_seconds = (end_t - start_t).total_seconds()
        duration = audio.shape[-1] / sample_rate
        rtf = elapsed_seconds / duration

        print(wav_path)
        print("Text:", stream.result.text)
        print(f"Audio duration:\t{duration:.3f} s")
        print(f"Elapsed:\t{elapsed_seconds:.3f} s")
        print(f"RTF = {elapsed_seconds:.3f}/{duration:.3f} = {rtf:.3f}")
        return stream.result.text

    def transcribe_from_microphone(self):
        devices = sd.query_devices()

        if len(devices) == 0:
            print("No microphone devices found")
            sys.exit(0)

        print(f"[INFO]: Use default device: {devices[0]['name']}")

        print("[INFO]: Started! Please speak")

        # Start recording
        sampling_rate = 16000
        last_result = ""
        stream = self.recognizer.create_stream()
        with (
            sd.InputStream(
                channels=1,  # Mono
                dtype="float32",  # 32-bit float
                samplerate=sampling_rate,  # 16 kHz, default of sherpa-onnx, if it's not 16 kHz, it will be resampled
            ) as s
        ):
            samples_per_read = int(0.1 * sampling_rate)  # 0.1 second = 100 ms
            while True:
                samples, _ = s.read(samples_per_read)  # a blocking read
                samples = samples.reshape(-1)
                stream.accept_waveform(sampling_rate, samples)
                while self.recognizer.is_ready(stream):
                    self.recognizer.decode_stream(stream)
                result = self.recognizer.get_result(stream)
                if last_result != result:
                    last_result = result
                    print("\r" + result, end="", flush=True)

        return last_result
