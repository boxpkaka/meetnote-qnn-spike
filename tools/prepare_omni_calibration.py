#!/usr/bin/env python3
"""Materialize the pinned raw Wikitext samples used by MNN OmniQuant."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SAMPLE_COUNT = 128
SHUFFLE_SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(cache_root: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to replace existing calibration data: {output}")
    os.environ["HF_HOME"] = str(cache_root.resolve())
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
        revision=DATASET_REVISION,
    ).shuffle(seed=SHUFFLE_SEED)
    samples: list[str] = []
    for row in dataset:
        text = row["text"]
        if isinstance(text, str) and text.strip():
            samples.append(text)
            if len(samples) == SAMPLE_COUNT:
                break
    if len(samples) != SAMPLE_COUNT:
        raise ValueError(f"expected {SAMPLE_COUNT} non-empty samples, got {len(samples)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for text in samples:
                handle.write(json.dumps(text, ensure_ascii=False) + "\n")
        os.replace(temporary_name, output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return {
        "format": "meetnote.omni_calibration.v1",
        "dataset": "Salesforce/wikitext",
        "configuration": "wikitext-2-raw-v1",
        "revision": DATASET_REVISION,
        "split": "train",
        "shuffle_seed": SHUFFLE_SEED,
        "samples": SAMPLE_COUNT,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.cache_root, args.output), indent=2))


if __name__ == "__main__":
    main()
