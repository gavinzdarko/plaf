"""Download and locally save GPT-2 for PLAF experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "gpt2"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

    sample = tokenizer("PLAF privacy leakage audit", return_tensors="pt")
    with torch.no_grad():
        logits = model(**sample).logits

    print(f"Saved GPT-2 tokenizer and weights to {MODEL_DIR}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
