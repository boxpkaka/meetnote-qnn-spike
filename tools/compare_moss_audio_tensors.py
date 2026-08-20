#!/usr/bin/env python3
"""Compare fixed MNN CPU and SM8850 QNN audio tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def statistics(reference: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {actual.shape}")
    left = reference.astype(np.float64, copy=False)
    right = actual.astype(np.float64, copy=False)
    difference = np.abs(left - right)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return {
        "elements": int(left.size),
        "all_finite": bool(np.isfinite(right).all()),
        "cosine": float(np.dot(left, right) / denominator),
        "mean_abs_error": float(difference.mean()),
        "max_abs_error": float(difference.max()),
    }


def read(path: Path, elements: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size != elements:
        raise ValueError(f"{path} has {values.size} elements, expected {elements}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnn-front", type=Path, required=True)
    parser.add_argument("--mnn-back", type=Path, required=True)
    parser.add_argument("--qnn-front", type=Path, required=True)
    parser.add_argument("--qnn-back-isolated", type=Path)
    parser.add_argument("--qnn-back-chain", type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=0.995)
    args = parser.parse_args()

    mnn_front = read(args.mnn_front, 1500 * 1024)
    mnn_back = read(args.mnn_back, 375 * 1024)
    report: dict[str, object] = {
        "format": "meetnote.moss_sm8850_audio_alignment.v1",
        "front": statistics(mnn_front, read(args.qnn_front, mnn_front.size)),
        "end_to_end": statistics(mnn_back, read(args.qnn_back_chain, mnn_back.size)),
    }
    if args.qnn_back_isolated is not None:
        report["back_isolated"] = statistics(
            mnn_back, read(args.qnn_back_isolated, mnn_back.size)
        )
    report["minimum_cosine"] = args.min_cosine
    report["status"] = "valid" if all(
        record["all_finite"] and record["cosine"] >= args.min_cosine
        for record in report.values()
        if isinstance(record, dict) and "cosine" in record
    ) else "invalid"
    print(json.dumps(report, indent=2))
    if report["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
