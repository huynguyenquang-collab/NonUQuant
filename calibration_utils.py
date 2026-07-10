"""
Calibration data loading utilities for RBVTQuant.

This is a local copy so RBVTQuant remains self-contained and does not depend on
NCCQuant at runtime.
"""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset


def get_c4_calibration_data(tokenizer, n_samples=128, seqlen=2048, seed=42, return_tensors=False, cache_dir="./calibration_cache"):
    print(f"\n[C4 Calibration Data - Optimized]")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seqlen} tokens")
    print(f"  Method: Random slicing with fast filtering")
    print(f"  Seed: {seed}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    cache_file = cache_path / f"c4_calib_n{n_samples}_len{seqlen}_seed{seed}_tensors{return_tensors}.pkl"
    if cache_file.exists():
        print(f"\n  Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"\n  No cache found, downloading from C4...")
    random.seed(seed)

    url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
    traindata = load_dataset(
        "json",
        data_files={"train": url},
        split="train",
        streaming=True,
    )

    dataset = []
    skipped = 0
    char_threshold = seqlen * 3

    print(f"\n  Streaming C4 with fast filtering...")
    for data in traindata:
        text = data["text"]
        if len(text) < char_threshold:
            skipped += 1
            continue

        trainenc = tokenizer(text, return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            skipped += 1
            continue

        max_start = trainenc.input_ids.shape[1] - seqlen
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + seqlen
        inp = trainenc.input_ids[:, start_idx:end_idx]

        if return_tensors:
            dataset.append(inp)
        else:
            dataset.append(tokenizer.decode(inp[0], skip_special_tokens=True))

        if len(dataset) % 32 == 0:
            print(f"    Collected {len(dataset)}/{n_samples} samples (skipped {skipped} short docs)...")
        if len(dataset) == n_samples:
            break

    print(f"\n  Collected {len(dataset)} samples from C4")
    print(f"  Skipped {skipped} documents (too short)")
    print(f"  Saving to cache: {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def get_wikitext2_calibration_data(tokenizer, n_samples=128, seqlen=2048, seed=42, split="train", cache_dir="./calibration_cache"):
    print(f"\n[WikiText-2 Calibration Data]")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seqlen} tokens")
    print(f"  Split: {split}")
    print(f"  Seed: {seed}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    cache_file = cache_path / f"wikitext2_calib_n{n_samples}_len{seqlen}_seed{seed}_split{split}.pkl"
    if cache_file.exists():
        print(f"\n  Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"\n  No cache found, downloading from WikiText-2...")
    random.seed(seed)

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 0]

    print(f"  Total non-empty texts: {len(texts)}")
    print(f"  Tokenizing and concatenating...")
    all_tokens = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        all_tokens.append(tokens)

    all_tokens = torch.cat(all_tokens, dim=0)
    print(f"  Total tokens: {len(all_tokens)}")

    num_chunks = len(all_tokens) // seqlen
    print(f"  Available {seqlen}-token chunks: {num_chunks}")
    if num_chunks < n_samples:
        print(f"  Warning: Only {num_chunks} chunks available, requested {n_samples}")
        n_samples = num_chunks

    chunk_indices = random.sample(range(num_chunks), n_samples)
    calibration_texts = []
    for idx in chunk_indices:
        start = idx * seqlen
        end = start + seqlen
        chunk_tokens = all_tokens[start:end]
        calibration_texts.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))

    print(f"  Collected {len(calibration_texts)} samples from WikiText-2")
    print(f"  Saving to cache: {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(calibration_texts, f)
    return calibration_texts


def _normalize_token_cache(tokens, seqlen: int) -> list[torch.Tensor]:
    if isinstance(tokens, torch.Tensor):
        if tokens.ndim == 2:
            return [tokens[i : i + 1, :seqlen].long() for i in range(tokens.shape[0])]
        if tokens.ndim == 3 and tokens.shape[1] == 1:
            return [tokens[i, :, :seqlen].long() for i in range(tokens.shape[0])]
        raise ValueError(f"Unsupported token tensor shape: {tuple(tokens.shape)}")
    if isinstance(tokens, (list, tuple)):
        out = []
        for item in tokens:
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"Unsupported token item type: {type(item).__name__}")
            item = item.detach().cpu().long()
            if item.ndim == 1:
                item = item.unsqueeze(0)
            if item.ndim != 2 or item.shape[0] != 1:
                raise ValueError(f"Unsupported token item shape: {tuple(item.shape)}")
            out.append(item[:, :seqlen])
        return out
    raise TypeError(f"Unsupported token cache type: {type(tokens).__name__}")


def get_redpajama_calibration_data(
    tokenizer,
    n_samples=1024,
    seqlen=4096,
    seed=0,
    cache_dir="./calibration_cache",
):
    """Load GuidedQuant-style RedPajama calibration as decoded text.

    If CALIB_TOKENS_PATH points to a GuidedQuant token cache, that exact cache is
    used. Otherwise this samples the public mirror by the same random document +
    random span protocol used by the BV-SQ/LNQ utilities.
    """

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    explicit_cache = os.environ.get("CALIB_TOKENS_PATH", "")
    if explicit_cache and Path(explicit_cache).exists():
        print(f"\n[RedPajama Calibration Data] Loading token cache: {explicit_cache}")
        try:
            tokens = torch.load(explicit_cache, map_location="cpu", weights_only=False)
        except TypeError:
            tokens = torch.load(explicit_cache, map_location="cpu")
        rows = _normalize_token_cache(tokens, seqlen)[:n_samples]
        return [tokenizer.decode(row[0], skip_special_tokens=True) for row in rows]

    cache_file = cache_path / f"redpajama_calib_n{n_samples}_len{seqlen}_seed{seed}.pkl"
    if cache_file.exists():
        print(f"\n  Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    dataset_name = os.environ.get("REDPAJAMA_DATASET", "ZengXiangyu/RedPajama-Data-1T-Sample")
    config = os.environ.get("REDPAJAMA_CONFIG", "")
    split = os.environ.get("REDPAJAMA_SPLIT", "train")
    print(
        "\n[RedPajama Calibration Data]\n"
        f"  Dataset: {dataset_name}\n"
        f"  Config: {config or '<none>'}\n"
        f"  Split: {split}\n"
        f"  Samples: {n_samples}\n"
        f"  Sequence length: {seqlen}\n"
        f"  Seed: {seed}"
    )
    args = [dataset_name]
    if config:
        args.append(config)
    raw = load_dataset(*args, split=split, trust_remote_code=True)
    rng = random.Random(seed)
    texts = []
    seen = set()
    while len(texts) < n_samples:
        idx = rng.randint(0, len(raw) - 1)
        if idx in seen and len(seen) < len(raw):
            continue
        seen.add(idx)
        item = raw[idx]
        text = item.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        enc = tokenizer(text, return_tensors="pt").input_ids
        if enc.shape[1] < seqlen:
            continue
        start = rng.randint(0, enc.shape[1] - seqlen)
        texts.append(tokenizer.decode(enc[0, start : start + seqlen], skip_special_tokens=True))

    with open(cache_file, "wb") as f:
        pickle.dump(texts, f)
    return texts


def load_calibration_data(dataset_name, tokenizer, n_samples=128, seqlen=2048, seed=42, cache_dir="./calibration_cache"):
    dataset_name = dataset_name.lower()
    if dataset_name == "c4":
        return get_c4_calibration_data(tokenizer, n_samples, seqlen, seed, cache_dir=cache_dir)
    if dataset_name in ["wikitext2", "wikitext"]:
        return get_wikitext2_calibration_data(tokenizer, n_samples, seqlen, seed, split="train", cache_dir=cache_dir)
    if dataset_name == "redpajama":
        return get_redpajama_calibration_data(tokenizer, n_samples, seqlen, seed, cache_dir=cache_dir)
    raise ValueError(f"Unknown dataset: {dataset_name}. Use 'c4', 'wikitext2', or 'redpajama'")
