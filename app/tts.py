import sherpa_onnx
import soundfile as sf
from pathlib import Path



class TTSModel:
    def __init__(self):
        self.model_path = Path("./sherpa-onnx/vits-piper-vi_VN-vais1000-medium")
        self.tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(self.model_path / "vi_VN-vais1000-medium.onnx"),
                    tokens=str(self.model_path / "tokens.txt"),
                    data_dir=str(self.model_path / "espeak-ng-data"),
                )
            )
        )

        if not self.tts_config.validate():
            raise ValueError("Please check your config")

        self.tts = sherpa_onnx.OfflineTts(self.tts_config)

    def synthesize(
        self, text: str, sid: int = 0, speed: float = 1.0, save_path: str = None
    ):
        audio = self.tts.generate(text, sid, speed)
        if save_path:
            sf.write(
                save_path, audio.samples, samplerate=audio.sample_rate, subtype="PCM_16"
            )
            print(f"[INFO] Audio saved to {save_path}")
        print(f"[INFO] Audio duration in seconds: {len(audio) / audio.sample_rate}")
        print(f"[INFO] Audio sample rate: {audio.sample_rate}")
        print(f"[INFO] The text is: {text}")
        return audio
