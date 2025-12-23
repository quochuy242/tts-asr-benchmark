import sherpa_onnx
import soundfile as sf

config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model="./sherpa-onnx/vits-piper-vi_VN-vais1000-medium/vi_VN-vais1000-medium.onnx",
            lexicon="",
            data_dir="./sherpa-onnx/vits-piper-vi_VN-vais1000-medium/espeak-ng-data",
            tokens="./sherpa-onnx/vits-piper-vi_VN-vais1000-medium/tokens.txt",
        ),
        num_threads=12,
    ),
)

if not config.validate():
    raise ValueError("Please check your config")

tts = sherpa_onnx.OfflineTts(config)
audio = tts.generate(text="nói hệ điều hành Windows, WSL, Ubuntu, Arch mình setup đúng cho bạn.", sid=0, speed=1.0)

sf.write("test.mp3", audio.samples, samplerate=audio.sample_rate)
