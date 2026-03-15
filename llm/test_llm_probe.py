"""End-to-end test of LLM membership inference probe.

Loads GPT-2, runs perplexity-based membership inference on known-training
texts vs. novel texts, and prints results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from llm.llm_probe import LLMTarget, LLMMembershipProbe


def load_texts(path: Path) -> list[str]:
    """Load texts from a JSON file (expects a list of strings or list of dicts with 'text' key)."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        if len(data) == 0:
            return []
        if isinstance(data[0], str):
            return data
        if isinstance(data[0], dict) and "text" in data[0]:
            return [d["text"] for d in data]

    raise ValueError(f"Unexpected JSON format in {path}")


def main() -> None:
    model_path = PROJECT_ROOT / "models" / "gpt2"
    training_path = PROJECT_ROOT / "data" / "llm_known_training.json"

    # Look for novel texts file (try common naming patterns)
    novel_candidates = [
        PROJECT_ROOT / "data" / "llm_novel_text.json",
        PROJECT_ROOT / "data" / "llm_novel.json",
        PROJECT_ROOT / "data" / "llm_novel_texts.json",
        PROJECT_ROOT / "data" / "llm_known_nontraining.json",
        PROJECT_ROOT / "data" / "llm_nontraining.json",
        PROJECT_ROOT / "data" / "llm_test.json",
    ]
    novel_path = None
    for candidate in novel_candidates:
        if candidate.exists():
            novel_path = candidate
            break

    if novel_path is None:
        # List what's available for debugging
        json_files = list((PROJECT_ROOT / "data").glob("llm_*.json"))
        if json_files:
            # Use whichever JSON file isn't the training one
            for f in json_files:
                if f != training_path:
                    novel_path = f
                    break
        if novel_path is None:
            print("ERROR: Could not find novel texts JSON file.")
            print(f"  Available JSON files: {json_files}")
            sys.exit(1)

    print(f"Training texts: {training_path}")
    print(f"Novel texts:    {novel_path}")
    print(f"Model:          {model_path}")
    print()

    # Load data
    training_texts = load_texts(training_path)
    novel_texts = load_texts(novel_path)
    print(f"Loaded {len(training_texts)} training texts, {len(novel_texts)} novel texts")

    # Load model
    print("Loading GPT-2...")
    target = LLMTarget(str(model_path))
    print("Model loaded.\n")

    # Run probe
    print("=" * 60)
    print("LLM MEMBERSHIP INFERENCE ATTACK")
    print("=" * 60)

    probe = LLMMembershipProbe(target)
    result = probe.probe(training_texts, novel_texts)

    ppl_train = np.array(result.confidence_scores_members)
    ppl_novel = np.array(result.confidence_scores_nonmembers)

    # Filter inf for stats
    ppl_train_finite = ppl_train[np.isfinite(ppl_train)]
    ppl_novel_finite = ppl_novel[np.isfinite(ppl_novel)]

    print(f"  AUC:                    {result.auc:.4f}")
    print(f"  TPR @ 5% FPR:           {result.tpr_at_5fpr:.4f}")
    print(f"  Detected members:       {result.n_detected_members}")
    print()
    print(f"  Mean perplexity (train): {ppl_train_finite.mean():.2f} ± {ppl_train_finite.std():.2f}")
    print(f"  Mean perplexity (novel): {ppl_novel_finite.mean():.2f} ± {ppl_novel_finite.std():.2f}")
    print(f"  Median ppl (train):      {np.median(ppl_train_finite):.2f}")
    print(f"  Median ppl (novel):      {np.median(ppl_novel_finite):.2f}")

    # Query-budget analysis
    print("\n  Query-budget curve:")
    budget = probe.run_query_budget_analysis(training_texts, novel_texts)
    for n_q, auc in sorted(budget.items()):
        print(f"    {n_q:>5} queries → AUC {auc:.4f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
