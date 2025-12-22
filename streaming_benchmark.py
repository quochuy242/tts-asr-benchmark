import sys
import time
from typing import Optional

import torch
import torchaudio
from datasets import Dataset, IterableDataset, load_dataset
from jiwer import wer
from ViStreamASR import StreamingASR


# List of metrics
def compute_wer(reference: str, prediction: str) -> float:
    return wer(reference, prediction)


# Modify the StreamingASR class to include benchmarking functionality
# Define symbols that work across platforms
symbols = {
    "tool": "🔧"
    if sys.stdout.encoding and "utf" in sys.stdout.encoding.lower()
    else "[CONFIG]",
    "check": "✅"
    if sys.stdout.encoding and "utf" in sys.stdout.encoding.lower()
    else "[OK]",
    "ruler": "📏"
    if sys.stdout.encoding and "utf" in sys.stdout.encoding.lower()
    else "[SIZE]",
    "folder": "📁"
    if sys.stdout.encoding and "utf" in sys.stdout.encoding.lower()
    else "[FILE]",
    "wave": "🎵"
    if sys.stdout.encoding and "utf" in sys.stdout.encoding.lower()
    else "[AUDIO]",
}


class BenchmarkingStreamingASR(StreamingASR):
    def _preprocess_hf_dataset_sample(self, sample):
        try:
            # Extract audio array and sampling rate from the dataset sample
            audio_array = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]

            # Load with torchaudio
            waveform, original_sr = torchaudio.load(audio_array)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                if self.debug:
                    print(f"{symbols['tool']} [StreamingASR] Converted stereo to mono.")
            # Resample if necessary
            if original_sr != sampling_rate:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=original_sr, new_freq=sampling_rate
                )(waveform)
                if self.debug:
                    print(
                        f"{symbols['tool']} [StreamingASR] Resampled from {original_sr}Hz to {sampling_rate}Hz."
                    )
            # Normalize
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val

            # Return processed sample
            return {
                "waveform": waveform.squeeze().numpy(),
                "sampling_rate": sampling_rate,
                "duration": len(waveform.squeeze()) / sampling_rate,
                "transcription": sample.get("transcription", ""),
            }

        except Exception as e:
            if self.debug:
                print(
                    f"{symbols['tool']} [StreamingASR] Error preprocessing sample: {e}"
                )
            return None

    def stream_from_dataset(
        self,
        dataset: IterableDataset | Dataset,
        chunk_size_ms: Optional[int] = None,
    ):
        """
        Stream ASR over dataset and yield ONE result per utterance.
        """

        self._ensure_engine_initialized()

        chunk_size = chunk_size_ms or self.chunk_size_ms
        chunk_size_samples = int(16_000 * chunk_size / 1000)

        utterance_id = 0
        total_audio_duration = 0.0
        total_process_time = 0.0

        for sample in dataset:
            utterance_id += 1

            processed = self._preprocess_hf_dataset_sample(sample)
            if processed is None:
                continue

            waveform = processed["waveform"]
            duration = processed["duration"]
            reference = processed.get("transcription", "").strip()

            total_audio_duration += duration

            if self.debug:
                print(
                    f"\n{symbols['folder']} [StreamingASR] "
                    f"Utterance {utterance_id} | {duration:.2f}s"
                )

            # Reset engine PER utterance
            self.engine.reset_state()

            final_segments = []

            total_chunks = (
                len(waveform) + chunk_size_samples - 1
            ) // chunk_size_samples

            start_time = time.time()

            for i in range(total_chunks):
                start = i * chunk_size_samples
                end = min(start + chunk_size_samples, len(waveform))
                chunk = waveform[start:end]

                is_last = i == total_chunks - 1

                result = self.engine.process_audio(chunk, is_last=is_last)

                if result.get("new_final_text"):
                    final_segments.append(result["new_final_text"])

            elapsed = time.time() - start_time
            total_process_time += elapsed

            hypothesis = " ".join(final_segments).strip()
            utterance_wer = wer(reference, hypothesis)

            yield {
                "utterance_id": utterance_id,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": utterance_wer,
                "duration": duration,
                "rtf": elapsed / duration if duration > 0 else 0.0,
            }

        if self.debug and total_audio_duration > 0:
            print(f"\n{symbols['check']} [StreamingASR] Dataset complete")
            print(f"{symbols['ruler']} Total audio: {total_audio_duration:.2f}s")
            print(f"{symbols['ruler']} Total time: {total_process_time:.2f}s")
            print(
                f"{symbols['check']} ⚡ Global RTF: "
                f"{total_process_time / total_audio_duration:.2f}x"
            )


def plot_benchmark_results(results, output_dir="benchmark_output"):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    utterance_ids = [r["utterance_id"] for r in results]
    wers = [r["wer"] for r in results]
    rtfs = [r["rtf"] for r in results]
    durations = [r["duration"] for r in results]

    # ---- WER per utterance ----
    plt.figure()
    plt.plot(utterance_ids, wers, marker="o")
    plt.xlabel("Utterance ID")
    plt.ylabel("WER")
    plt.title("WER per Utterance")
    plt.grid(True)
    plt.savefig(f"{output_dir}/wer_per_utterance.png")
    plt.close()

    # ---- RTF per utterance ----
    plt.figure()
    plt.plot(utterance_ids, rtfs, marker="o")
    plt.xlabel("Utterance ID")
    plt.ylabel("RTF")
    plt.title("RTF per Utterance")
    plt.grid(True)
    plt.savefig(f"{output_dir}/rtf_per_utterance.png")
    plt.close()

    # ---- WER histogram ----
    plt.figure()
    plt.hist(wers, bins=20)
    plt.xlabel("WER")
    plt.ylabel("Count")
    plt.title("WER Distribution")
    plt.grid(True)
    plt.savefig(f"{output_dir}/wer_histogram.png")
    plt.close()

    # ---- Duration vs RTF ----
    plt.figure()
    plt.scatter(durations, rtfs)
    plt.xlabel("Audio Duration (s)")
    plt.ylabel("RTF")
    plt.title("Duration vs RTF")
    plt.grid(True)
    plt.savefig(f"{output_dir}/duration_vs_rtf.png")
    plt.close()


if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset("doof-ferb/infore1_25hours", split="train")
    test_ds = dataset.shuffle(seed=42)
    # test_ds = test_ds.select(range(200))  # Take a small subset for testing

    # Initialize benchmarking ASR
    asr = BenchmarkingStreamingASR()
    results = list(asr.stream_from_dataset(test_ds))

    # Plot benchmark results
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        plot_benchmark_results(results)
