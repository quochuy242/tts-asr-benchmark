import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import re
    from pprint import pprint
    from pathlib import Path
    from typing import Dict, Tuple, Any

    import marimo as mo
    import pandas as pd
    return Any, Dict, Path, mo, pd, pprint, re


@app.cell
def _(mo):
    mo.md(r"""
    ## Load all result files from models' benchmarking
    """)
    return


@app.cell
def _(Path, pd, pprint):
    # Vosk
    OUT_DIR = Path("./benchmark_output/")
    VOSK_DIR = OUT_DIR / "vosk"
    vosk_df = pd.read_csv(VOSK_DIR / "vi_with_server.csv")
    print(vosk_df.head(), "\n")

    vosk_stats = {
        "model": "vosk-model-vn-0.4",
        "server": True,
        "elapsed_time": vosk_df["elapsed"].sum(),
        "rtf": vosk_df["rtf"].mean().round(decimals=4),
    }
    pprint(vosk_stats)
    return OUT_DIR, vosk_df, vosk_stats


@app.cell
def _(Dict, OUT_DIR, Path, pd, pprint, re):
    # Sherpa-onnx
    SHERPA_DIR = OUT_DIR / "sherpa-onnx"
    sherpa_vi_df = pd.read_csv(SHERPA_DIR / "vi_hypothesis.csv")
    sherpa_vi_df["utt"] = sherpa_vi_df["wav"].apply(lambda x: x.split("/")[-1])
    sherpa_vi_df = sherpa_vi_df.drop(columns=["wav"])

    sherpa_vi_int8_df = pd.read_csv(SHERPA_DIR / "vi_int8_hypothesis.csv")
    sherpa_vi_int8_df["utt"] = sherpa_vi_int8_df["wav"].apply(
        lambda x: x.split("/")[-1]
    )
    sherpa_vi_int8_df = sherpa_vi_int8_df.drop(columns=["wav"])


    def parse_sherpa_stats(path: Path | str) -> Dict[str, float]:
        stats = dict()
        with open(path) as f:
            for line in f:
                line = line.strip()

                # Num-threads
                if line.startswith("num threads:"):
                    stats["num_threads"] = int(line.split(":")[1].strip())

                # Decoding method
                elif line.startswith("decoding method:"):
                    stats["decoding_method"] = line.split(":")[1].strip()

                # Elapsed (s)
                elif line.startswith("Elapsed seconds:"):
                    stats["elapsed_seconds"] = float(
                        re.search(r"([\d.]+)", line).group(1)
                    )

                # RTF & duration (s)
                elif line.startswith("Real time factor"):
                    m = re.search(r"([\d.]+)\s*/\s*([\d.]+)\s*=\s*([\d.]+)", line)
                    if m:
                        stats["audio_duration"] = float(m.group(2))
                        stats["rtf"] = float(m.group(3))
        return stats


    sherpa_vi_stats = {
        "model": "zipformer-vi-2025-04-20",
        "precision": "fp32",
        **parse_sherpa_stats(SHERPA_DIR / "vi_stats.txt"),
    }
    sherpa_vi_int8_stats = {
        "model": "zipformer-vi-2025-04-20",
        "precision": "int8",
        **parse_sherpa_stats(SHERPA_DIR / "vi_int8_stats.txt"),
    }

    print(sherpa_vi_df.head())
    print(sherpa_vi_int8_df.head(), "\n")
    pprint(sherpa_vi_stats)
    pprint(sherpa_vi_int8_stats)
    return (
        sherpa_vi_df,
        sherpa_vi_int8_df,
        sherpa_vi_int8_stats,
        sherpa_vi_stats,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Calc WER
    """)
    return


@app.cell
def _(pd):
    # Load the reference
    ref_df = pd.read_csv("./data/metadata.csv")
    ref_df = ref_df.drop(columns=["wav_path", "duration"])
    ref_df = ref_df.rename(columns={"utt_id": "utt"})
    print(ref_df.head())
    return (ref_df,)


@app.cell
def _(Any, Dict, pd):
    import jiwer
    from jiwer import (
        cer,
        wer,
        Compose,
        RemoveMultipleSpaces,
        Strip,
        ToLowerCase,
        ReduceToListOfListOfWords,
    )


    def compute_error(refs: pd.Series, hypos: pd.Series, stats: Dict[str, Any]):
        """
        Compute Word Error Rate (WER) between reference and hypothesis texts.

        Args:
            refs  (pd.Series): ground-truth transcripts
            hypos (pd.Series): ASR outputs

        Returns:
            float: WER (0.0 ~ 1.0)
        """
        assert len(refs) == len(hypos), "refs and hypos must have the same length"

        # Text normalization (VERY important)
        transform = Compose(
            [
                ToLowerCase(),
                Strip(),
                RemoveMultipleSpaces(),
                ReduceToListOfListOfWords(),
            ]
        )

        refs = refs.fillna("").astype(str).tolist()
        hypos = hypos.fillna("").astype(str).tolist()

        outputs = jiwer.process_words(
            reference=refs,
            hypothesis=hypos,
            reference_transform=transform,
            hypothesis_transform=transform,
        )

        wer = outputs.wer
        wil = outputs.wil
        mer = outputs.mer

        cer = jiwer.cer(
            reference=refs,
            hypothesis=hypos,
            reference_transform=transform,
            hypothesis_transform=transform,
        )
        stats.update(
            {
                "wer": wer,
                "wil": wil,
                "mer": mer,
                "cer": cer,
            }
        )

        return stats, wer, wil, mer, cer
    return (compute_error,)


@app.cell
def _(
    compute_error,
    pprint,
    ref_df,
    sherpa_vi_df,
    sherpa_vi_int8_df,
    sherpa_vi_int8_stats,
    sherpa_vi_stats,
    vosk_df,
    vosk_stats,
):
    [
        compute_error(hypos=df["text"], refs=ref_df["text"], stats=stat)
        for df, stat in zip(
            [vosk_df, sherpa_vi_df, sherpa_vi_int8_df],
            [vosk_stats, sherpa_vi_stats, sherpa_vi_int8_stats],
        )
    ]

    pprint(vosk_stats)
    pprint(sherpa_vi_stats)
    pprint(sherpa_vi_int8_stats)
    return


@app.cell
def _(sherpa_vi_int8_stats, sherpa_vi_stats, vosk_stats):
    import matplotlib.pyplot as plt

    stats = [vosk_stats, sherpa_vi_stats, sherpa_vi_int8_stats]
    metrics = ["rtf", "wer", "wil", "mer", "cer"]
    models = ["vosk", "sherpa_fp32", "sherpa_int8"]

    scores = {metric: [s[metric] for s in stats] for metric in metrics}
        
    x = range(len(models))
    width = 0.15

    plt.figure(figsize=(12, 6))

    plt.bar([i - 2*width for i in x], scores['rtf'], width, label="RTF ↓")
    plt.bar([i - width for i in x], scores['wer'], width, label="WER ↓")
    plt.bar(x, scores['wil'], width, label="WIL ↓")
    plt.bar([i + width for i in x], scores['mer'], width, label="MER ↓")
    plt.bar([i + 2*width for i in x], scores['cer'], width, label="CER ↓")

    plt.xticks(x, models)
    plt.ylabel("Metric value")
    plt.title("ASR Benchmark Comparison (↓ lower is better)")
    plt.legend(title="Metrics")

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
