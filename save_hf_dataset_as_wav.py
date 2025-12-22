import os
import csv
import soundfile as sf
import librosa
from datasets import load_dataset


def save_hf_dataset_as_wav(
    output_dir: str = "data",
    audio_col: str = "audio",
    text_col: str = "transcription",
    target_sr: int = 16000,
    **dataset_info,
):
    dataset = load_dataset(
        dataset_info["id"], split=dataset_info.get("split", "train"), streaming=False
    )
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    meta_path = os.path.join(output_dir, "metadata.csv")

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "wav_path", "text", "duration"])

        for i, sample in enumerate(dataset):
            utt_id = f"utt_{i:06d}"

            audio = sample[audio_col]
            waveform = audio["array"]
            sr = audio["sampling_rate"]

            wav_path = os.path.join(audio_dir, f"{utt_id}.wav")

            if sr != target_sr:
                waveform = librosa.resample(
                    waveform.astype(float), orig_sr=sr, target_sr=target_sr
                )
                sr = target_sr

            duration = len(waveform) / sr
            text = sample.get(text_col, "")
            sf.write(wav_path, waveform, sr)
            writer.writerow([utt_id, wav_path, text, f"{duration:.3f}"])

    print(f"Saved {len(dataset)} samples to {output_dir}")


if __name__ == "__main__":
    save_hf_dataset_as_wav(id="doof-ferb/infore1_25hours", split="train")
