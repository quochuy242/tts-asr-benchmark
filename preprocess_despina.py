# This script uses Zipfomer Transducer Vi Int8 to transcribe the audio file
import os
import shutil
import time
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import sherpa_onnx as sherpa
import soundfile as sf
from jiwer import wer
from tqdm import tqdm


def create_metadata(
    wav_paths: List[Path | str],
    text_paths: List[Path | str],
    save_path: Optional[Path | str],
) -> pd.DataFrame:
    metadata = pd.DataFrame(
        columns=[
            "utt_id",
            "wav_path",
            "transcription",
            "duration",
            "num_words",
            "sample_rate",
        ]
    )

    for wav, text in tqdm(zip(wav_paths, text_paths), desc="Creating metadata"):
        # convert to Path
        if isinstance(wav, str):
            wav = Path(wav)
        if isinstance(text, str):
            text = Path(text)

        # extract metadata
        utt_id = wav.stem
        wav_path = str(wav.resolve())
        transcription = text.read_text(encoding="utf-8")
        num_words = len(transcription.split())

        with sf.SoundFile(wav_path) as f:
            duration = len(f) / f.samplerate
            sr = f.samplerate

        metadata = metadata._append(
            {
                "utt_id": utt_id,
                "wav_path": wav_path,
                "transcription": transcription,
                "duration": duration,
                "num_words": num_words,
                "sample_rate": sr,
            },
            ignore_index=True,
        )

    # print information
    print(f"[INFO] Number of utterances: {len(metadata)}")
    print(f"[INFO] Avgerage number of words: {metadata['num_words'].mean()}")
    print(f"[INFO] Avgerage duration: {metadata['duration'].mean()}")
    print(f"[INFO] Avgerage sample rate: {metadata['sample_rate'].mean()}")

    # save metadata
    if save_path:
        metadata.to_csv(save_path, index=False)
        print(f"[INFO] Metadata saved to {save_path}")

    return metadata


def recognize(
    model: sherpa.OfflineRecognizer, metadata: pd.DataFrame, batch_size: int = 32
) -> List[str]:
    results = []
    total_duration = metadata["duration"].sum()
    start_t = time.time()

    for i in range(0, len(metadata), batch_size):
        batch = metadata.iloc[i : i + batch_size]
        streams = []

        for _, row in batch.iterrows():
            audio, sr = sf.read(row["wav_path"], dtype="float32")
            s = model.create_stream()
            s.accept_waveform(sr, audio)
            streams.append(s)

        model.decode_streams(streams)
        results.extend(s.result.text for s in streams)

    del streams  # interupt OOM

    elapsed = time.time() - start_t
    print(f"[INFO] RTF: {elapsed / total_duration:.3f}")
    return results


def check_wer(
    model: sherpa.OfflineRecognizer,
    metadata: pd.DataFrame,
    threshold: float,
    batch_size: int = 32,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Double-check WER of original dataset by recognized results by another ASR model

    Args:
        model (sherpa.OfflineRecognizer): Model object
        metadata (pd.DataFrame)
        threshold (float)
        batch_size (int, optional). Defaults to 32.
        inplace (bool, optional). Defaults to False.

    Returns:
        pd.DataFrame: Unqualified utterances
    """
    references = metadata["transcription"].tolist()

    # decode
    hypotheses = recognize(model, metadata)

    # per-utterance WER
    list_wer = [wer(r, h) for r, h in zip(references, hypotheses)]
    print(f"[INFO] Max WER: {max(list_wer)}")
    print(f"[INFO] Average WER: {sum(list_wer) / len(list_wer)}")

    if inplace:
        print(
            f'[WARNING] "inplace" is set to True. The original metadata will be modified. The length of metadata is {len(metadata)}'
        )
        metadata = metadata.copy()
        metadata["wer"] = list_wer
        metadata = metadata[metadata["wer"] < float(threshold)]
        print(f"[INFO] Number of unqualified utterances: {len(metadata)}")
        return metadata
    else:
        temp = pd.DataFrame(
            {
                "wav_path": metadata["wav_path"],
                "text_path": metadata["text_path"],
                "wer": list_wer,
            }
        )
        unqualified = temp[temp["wer"] < float(threshold)]
        print(f"[INFO] Number of unqualified utterances: {len(unqualified)}")
        return unqualified


def load_model(
    model_path: Path | str, num_threads: int = 1
) -> sherpa.OfflineRecognizer:
    model_path = Path(model_path) if isinstance(model_path, str) else model_path
    model = sherpa.OfflineRecognizer.from_transducer(
        encoder=str(model_path / "encoder-epoch-12-avg-8.int8.onnx"),
        decoder=str(model_path / "decoder-epoch-12-avg-8.onnx"),
        joiner=str(model_path / "joiner-epoch-12-avg-8.int8.onnx"),
        tokens=str(model_path / "tokens.txt"),
        num_threads=min(num_threads, os.cpu_count() - 1),
    )
    return model


def move_to_unqualified(path: Path | str, metadata: pd.DataFrame):
    path = Path(path)
    for wav_path, text_path in metadata[["wav_path", "text_path"]].values:
        wav_path = Path(wav_path)
        text_path = Path(text_path)
        shutil.move(wav_path, path / wav_path.name)
        shutil.move(text_path, path / text_path.name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads to use",
    )

    parser.add_argument(
        "--unqualified-path",
        type=str,
        required=True,
        help="Path to unqualified directory",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Threshold for WER, if WER > threshold, move to unqualified directory",
    )

    parser.add_argument(
        "--metadata-save-path",
        type=str,
        default="./data/despina.csv",
        help="Path to save metadata",
    )

    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Modify metadata inplace",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for decoding",
    )

    args = parser.parse_args()

    save_path = Path(args.metadata_save_path)
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    unqualified_path = Path(args.unqualified_path)

    if save_path.exists():
        print(f"[WARNING] Metadata file {save_path} already exists")
        metadata = pd.read_csv(save_path)
    else:
        if not data_dir.exists():
            print(f"[ERROR] Data directory {data_dir} does not exist")
            sys.exit(1)
        wav_paths = list(data_dir.glob("*.wav"))
        text_paths = list(data_dir.glob("*.txt"))
        metadata = create_metadata(wav_paths, text_paths, save_path=save_path)

    # Load model
    if not model_dir.exists():
        print(f"[ERROR] Model directory {model_dir} does not exist")
        sys.exit(1)
    model = load_model(model_dir, num_threads=args.num_threads)

    # Move wav and text files to unqualified directory if WER > threshold
    if not unqualified_path.exists():
        print(f"[WARNING] Unqualified directory {unqualified_path} does not exist")
        unqualified_path.parent.mkdir(parents=True, exist_ok=True)
    unqualified_df: pd.DataFrame = check_wer(
        model, metadata, threshold=args.threshold, inplace=False
    )
    move_to_unqualified(unqualified_path, unqualified_df)

    print("[INFO] Done")
