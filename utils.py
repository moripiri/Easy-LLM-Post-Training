from pathlib import Path

import torch
from huggingface_hub import snapshot_download


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable in this Python environment.")
    if requested == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError("MPS was requested, but this PyTorch build has no MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested, but it is unavailable. Run with a PyTorch build "
                "that can access the Mac's Metal device."
            )

    return torch.device(requested)


def get_device():
    return select_device()


def resolve_model_path(model_path: str) -> str:
    if Path(model_path).is_dir():
        return model_path
    return snapshot_download(model_path)
