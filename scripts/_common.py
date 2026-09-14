import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def device_or_fallback(name: str) -> str:
    import torch
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU")
        return "cpu"
    return name
