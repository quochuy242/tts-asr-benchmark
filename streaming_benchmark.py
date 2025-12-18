from typing import Dict, List, Any, Tuple

from datasets import load_dataset
from jiwer import wer
from torch.utils.data import DataLoader

from .vi_stream_asr.streaming import StreamingASR


def load_test_dataset(hf_id: str, split: str | None = None) -> DataLoader:
    ds = load_dataset(hf_id, split=split)
    ds.set_format(type="torch", columns=["audio", "transcription"])
    dataloader = DataLoader(ds, batch_size=1)
    return dataloader


# List of metrics
def compute_wer(reference: str, prediction: str) -> float:
    return wer(reference, prediction)


# The main prediction function
def predict(
    model_name: str, dataloader: DataLoader
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if model_name == "ViStreamASR":
        model = StreamingASR()
        return list(
            model.stream_from_dataloader(dataloader, benchmark=True)
        ), dataloader.dataset["transcription"]
    else:
        raise ValueError(f"Model {model_name} not supported.")


# The main benchmarking function
def benchmark(model_name: str, hf_id: str, split: str | None = None):
    dataloader = load_test_dataset(hf_id, split)
    predictions, reference_texts = predict(model_name, dataloader)

    rtfs = []
    predicted_texts = []
    for out in predictions:
        if out.get("benchmark_stat"):
            rtfs.append(out["benchmark_stat"]["rtf"])
        elif out["final"]:
            predicted_texts.append(out["text"])
