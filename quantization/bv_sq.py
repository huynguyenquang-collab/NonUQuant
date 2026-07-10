import argparse
import gc
import heapq
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm

from quantization.rbvt_squeezellm import collect_activation_stats


_WORKER_MU = None
_WORKER_H = None
_WORKER_K = None
_WORKER_SOLVER = None
_WORKER_BIAS_LAMBDA = None
_WORKER_MIN_SIZE = None
_WORKER_EPS = None


def atomic_pickle_dump(payload, path):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "wb") as handle:
            pickle.dump(payload, handle)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def atomic_torch_save(payload, path):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_range(value):
    if value is None:
        return None
    start, end = [int(x) for x in value.split(",")]
    return range(start, end)


def sort_row(w_row, mu, h):
    order = np.argsort(w_row, kind="mergesort")
    x = np.asarray(w_row, dtype=np.float64)[order]
    mu_sorted = np.asarray(mu, dtype=np.float64)[order]
    h_sorted = np.asarray(h, dtype=np.float64)[order]
    return x, mu_sorted, h_sorted, order


def build_prefix_sums(x, mu, h):
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    return {
        "PM": np.concatenate(([0.0], np.cumsum(mu))),
        "PU": np.concatenate(([0.0], np.cumsum(mu * x))),
        "PA": np.concatenate(([0.0], np.cumsum(h))),
        "PB": np.concatenate(([0.0], np.cumsum(h * x))),
        "PC": np.concatenate(([0.0], np.cumsum(h * x * x))),
        "x": x,
    }


def interval_stats(prefix, l, r):
    M = prefix["PM"][r + 1] - prefix["PM"][l]
    U = prefix["PU"][r + 1] - prefix["PU"][l]
    A = prefix["PA"][r + 1] - prefix["PA"][l]
    B = prefix["PB"][r + 1] - prefix["PB"][l]
    C = prefix["PC"][r + 1] - prefix["PC"][l]
    return M, U, A, B, C


def _interval_cost_values(prefix, l, r, bias_lambda=1.0, eps=1e-12):
    M = prefix["PM"][r + 1] - prefix["PM"][l]
    U = prefix["PU"][r + 1] - prefix["PU"][l]
    A = prefix["PA"][r + 1] - prefix["PA"][l]
    B = prefix["PB"][r + 1] - prefix["PB"][l]
    C = prefix["PC"][r + 1] - prefix["PC"][l]
    denom = bias_lambda * M * M + A
    numer = bias_lambda * M * U + B
    cost = bias_lambda * U * U + C
    valid = denom > eps
    cost = np.where(valid, cost - (numer * numer) / np.maximum(denom, eps), 0.0)
    return np.maximum(cost, 0.0)


def interval_cost(prefix, l, r, bias_lambda=1.0, eps=1e-12):
    M, U, A, B, C = interval_stats(prefix, l, r)
    denom = bias_lambda * M * M + A
    numer = bias_lambda * M * U + B
    if denom <= eps:
        c = float(prefix["x"][l : r + 1].mean()) if r >= l else 0.0
        return 0.0, c
    c = numer / denom
    cost = bias_lambda * U * U + C - (numer * numer) / denom
    if cost < 0.0 and cost > -1e-8:
        cost = 0.0
    return float(max(cost, 0.0)), float(c)


def best_split(l, r, cost_fn, min_size=1):
    if r - l + 1 < 2 * min_size:
        return -float("inf"), None
    base = cost_fn(l, r)
    best_gain = -float("inf")
    best_t = None
    for t in range(l + min_size - 1, r - min_size + 1):
        gain = base - cost_fn(l, t) - cost_fn(t + 1, r)
        if gain > best_gain:
            best_gain = gain
            best_t = t
    return best_gain, best_t


def _best_split_fast(prefix, l, r, bias_lambda=1.0, min_size=1, eps=1e-12):
    if r - l + 1 < 2 * min_size:
        return -float("inf"), None
    t_values = np.arange(l + min_size - 1, r - min_size + 1, dtype=np.int64)
    base = _interval_cost_values(prefix, l, r, bias_lambda, eps)
    left = _interval_cost_values(prefix, l, t_values, bias_lambda, eps)
    right = _interval_cost_values(prefix, t_values + 1, r, bias_lambda, eps)
    gains = base - left - right
    idx = int(np.argmax(gains))
    return float(gains[idx]), int(t_values[idx])


def bv_hier_split(x, mu, h, K, bias_lambda=1.0, min_size=1, eps=1e-12):
    prefix = build_prefix_sums(x, mu, h)
    intervals = [(0, len(x) - 1)]
    while len(intervals) < K:
        split_candidates = []
        for idx, (l, r) in enumerate(intervals):
            gain, t = _best_split_fast(prefix, l, r, bias_lambda, min_size, eps)
            if t is not None:
                split_candidates.append((gain, idx, t))
        if not split_candidates:
            break

        remaining = K - len(intervals)
        if len(split_candidates) > remaining:
            split_candidates = sorted(split_candidates, reverse=True)[:remaining]
        selected = {idx: t for _gain, idx, t in split_candidates}

        new_intervals = []
        for idx, (l, r) in enumerate(intervals):
            t = selected.get(idx)
            if t is None:
                new_intervals.append((l, r))
            else:
                new_intervals.append((l, t))
                new_intervals.append((t + 1, r))
        intervals = new_intervals
    intervals = sorted(intervals, key=lambda p: p[0])
    codewords = [interval_cost(prefix, l, r, bias_lambda, eps)[1] for l, r in intervals]
    return intervals, codewords


def bv_greedy_split(x, mu, h, K, bias_lambda=1.0, min_size=1, eps=1e-12):
    n = len(x)
    prefix = build_prefix_sums(x, mu, h)
    intervals = {(0, n - 1)}
    heap = []
    counter = 0
    gain, t = _best_split_fast(prefix, 0, n - 1, bias_lambda, min_size, eps)
    if t is not None:
        heapq.heappush(heap, (-gain, counter, 0, n - 1, t))
        counter += 1
    while len(intervals) < K and heap:
        neg_gain, _counter, l, r, t = heapq.heappop(heap)
        if (l, r) not in intervals:
            continue
        intervals.remove((l, r))
        left_interval = (l, t)
        right_interval = (t + 1, r)
        intervals.add(left_interval)
        intervals.add(right_interval)
        for child_l, child_r in (left_interval, right_interval):
            child_gain, child_t = _best_split_fast(prefix, child_l, child_r, bias_lambda, min_size, eps)
            if child_t is not None:
                heapq.heappush(heap, (-child_gain, counter, child_l, child_r, child_t))
                counter += 1
    intervals = sorted(intervals, key=lambda p: p[0])
    codewords = [interval_cost(prefix, l, r, bias_lambda, eps)[1] for l, r in intervals]
    return intervals, codewords


def build_assignment(n, intervals, order, codewords):
    labels_sorted = np.zeros(n, dtype=np.int64)
    q_sorted = np.zeros(n, dtype=np.float32)
    covered = np.zeros(n, dtype=bool)
    for k, (l, r) in enumerate(intervals):
        if l < 0 or r >= n or l > r:
            raise ValueError(f"Invalid BV-SQ interval {(l, r)} for row length {n}")
        labels_sorted[l : r + 1] = k
        q_sorted[l : r + 1] = codewords[k]
        covered[l : r + 1] = True
    if not covered.all():
        missing = np.nonzero(~covered)[0]
        preview = missing[:8].tolist()
        raise RuntimeError(
            f"BV-SQ intervals do not cover the full row; missing {missing.size} positions, "
            f"first missing={preview}"
        )
    labels_original = np.empty(n, dtype=np.int64)
    q_original = np.empty(n, dtype=np.float32)
    labels_original[order] = labels_sorted
    q_original[order] = q_sorted
    return labels_original, q_original


def bv_sq_row(w_row, mu, h, K, solver="greedy", bias_lambda=1.0, min_size=1, eps=1e-12):
    x, mu_sorted, h_sorted, order = sort_row(w_row, mu, h)
    if solver == "greedy":
        intervals, codewords = bv_greedy_split(x, mu_sorted, h_sorted, K, bias_lambda, min_size, eps)
    elif solver == "hier":
        intervals, codewords = bv_hier_split(x, mu_sorted, h_sorted, K, bias_lambda, min_size, eps)
    else:
        raise ValueError(f"Unsupported BV-SQ solver: {solver}")
    if len(codewords) < K:
        codewords = list(codewords) + [codewords[-1] if codewords else 0.0] * (K - len(codewords))
    labels, q_row = build_assignment(len(w_row), intervals, order, codewords)
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= len(codewords):
        raise RuntimeError(
            f"BV-SQ produced invalid labels: min={labels.min()} max={labels.max()} "
            f"num_codewords={len(codewords)}"
        )
    return np.asarray(codewords, dtype=np.float32), labels.astype(np.uint8), q_row


def _init_worker(mu, h, K, solver, bias_lambda, min_size, eps):
    global _WORKER_MU, _WORKER_H, _WORKER_K, _WORKER_SOLVER, _WORKER_BIAS_LAMBDA, _WORKER_MIN_SIZE, _WORKER_EPS
    _WORKER_MU = np.asarray(mu, dtype=np.float64)
    _WORKER_H = np.asarray(h, dtype=np.float64)
    _WORKER_K = K
    _WORKER_SOLVER = solver
    _WORKER_BIAS_LAMBDA = bias_lambda
    _WORKER_MIN_SIZE = min_size
    _WORKER_EPS = eps


def _worker_row(w_row):
    centers, labels, _q_row = bv_sq_row(
        w_row,
        _WORKER_MU,
        _WORKER_H,
        _WORKER_K,
        solver=_WORKER_SOLVER,
        bias_lambda=_WORKER_BIAS_LAMBDA,
        min_size=_WORKER_MIN_SIZE,
        eps=_WORKER_EPS,
    )
    return centers, labels


def sanitize_stat(vector, fallback=0.0):
    arr = np.asarray(vector, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, fallback)
    return arr


def load_h_vector(args, layer_idx, module_name, means, variances):
    stat_key = f"{layer_idx}:{module_name}"
    if args.h_source == "variance":
        h = variances[stat_key]
    elif args.h_source == "fisher_mean":
        if not args.fisher_chunks:
            raise ValueError("--h_source fisher_mean requires --fisher_chunks")
        fisher_layer = torch_load_cpu(os.path.join(args.fisher_chunks, f"layer_{layer_idx}.pt"))
        h = fisher_layer[module_name].float().mean(dim=0)
    else:
        raise ValueError(f"Unsupported h_source={args.h_source}")
    return h


def quantize_bv_sq(args):
    os.makedirs(os.path.join(args.output_folder, "lut"), exist_ok=True)
    means, variances = collect_activation_stats(args)
    layer_ids = sorted(
        int(name[len("layer_") : -len(".pt")])
        for name in os.listdir(args.model_chunks)
        if name.startswith("layer_") and name.endswith(".pt")
    )
    selected = set(parse_range(args.layer_range) or layer_ids)
    K = 2 ** args.bit

    metadata = {
        "method": "BV-SQ",
        "solver": args.solver,
        "bit": args.bit,
        "K": K,
        "bias_lambda": args.bias_lambda,
        "min_size": args.min_size,
        "eps": args.eps,
        "h_source": args.h_source,
    }
    os.makedirs(args.output_folder, exist_ok=True)
    with open(os.path.join(args.output_folder, "config.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    for layer_idx in layer_ids:
        if layer_idx not in selected:
            continue
        out_file = os.path.join(args.output_folder, "lut", f"l{layer_idx}.pkl")
        if os.path.exists(out_file) and not args.overwrite:
            print(f"Skipping layer {layer_idx}; {out_file} already exists.", flush=True)
            continue

        print(f"BV-SQ quantizing layer {layer_idx}", flush=True)
        model_layer = torch_load_cpu(os.path.join(args.model_chunks, f"layer_{layer_idx}.pt"))
        out_layer = {}
        for module_name, weight in model_layer.items():
            stat_key = f"{layer_idx}:{module_name}"
            if stat_key not in means:
                raise KeyError(f"Missing activation mean stats for {stat_key}")
            mu = sanitize_stat(means[stat_key].float().cpu().numpy())
            h = sanitize_stat(load_h_vector(args, layer_idx, module_name, means, variances).float().cpu().numpy())
            h = np.clip(h, 0.0, None)
            if args.h_floor > 0.0:
                h = np.maximum(h, args.h_floor)
            if mu.shape[0] != weight.shape[1] or h.shape[0] != weight.shape[1]:
                raise ValueError(
                    f"Stats shape mismatch for {stat_key}: mu={mu.shape} h={h.shape} "
                    f"weight={tuple(weight.shape)}"
                )

            rows = weight.float().numpy()
            print(
                f"Layer {layer_idx} module {module_name}: rows={rows.shape[0]} cols={rows.shape[1]} "
                f"solver={args.solver} bias_lambda={args.bias_lambda}",
                flush=True,
            )
            if args.cpu_count <= 1:
                results = [
                    bv_sq_row(
                        row,
                        mu,
                        h,
                        K,
                        solver=args.solver,
                        bias_lambda=args.bias_lambda,
                        min_size=args.min_size,
                        eps=args.eps,
                    )[:2]
                    for row in tqdm(rows, desc=f"BV-SQ l{layer_idx} {module_name}")
                ]
            else:
                with Pool(
                    processes=args.cpu_count,
                    initializer=_init_worker,
                    initargs=(mu, h, K, args.solver, args.bias_lambda, args.min_size, args.eps),
                ) as pool:
                    results = list(
                        tqdm(
                            pool.imap(_worker_row, rows, chunksize=args.row_chunksize),
                            total=rows.shape[0],
                            desc=f"BV-SQ l{layer_idx} {module_name}",
                        )
                    )
            out_layer[module_name] = [[(centers, labels)] for centers, labels in results]
            del rows, results
            gc.collect()
        atomic_pickle_dump(out_layer, out_file)
        print(f"Saved BV-SQ layer LUT to {out_file}", flush=True)
        del model_layer, out_layer
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="BV-SQ Bias-Variance Scalar Quantization.")
    parser.add_argument("mode", nargs="?", default="all", choices=["all", "stats", "quantize"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_chunks", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--model_type", default="llama", choices=["llama", "mistral", "qwen"])
    parser.add_argument("--dataset", default="redpajama", choices=["redpajama", "wikitext2", "c4"])
    parser.add_argument("--nsamples", type=int, default=1024)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", default="cache/bv_sq/tokens")
    parser.add_argument("--stats_path", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_dtype", default="default", choices=["default", "auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--n_calib", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"))
    parser.add_argument("--bit", type=int, default=3, choices=[3, 4])
    parser.add_argument("--solver", default="greedy", choices=["greedy", "hier"])
    parser.add_argument("--bias_lambda", type=float, default=1.0)
    parser.add_argument("--min_size", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--h_source", default="variance", choices=["variance", "fisher_mean"])
    parser.add_argument("--h_floor", type=float, default=0.0)
    parser.add_argument("--fisher_chunks", default="")
    parser.add_argument("--cpu_count", type=int, default=16)
    parser.add_argument("--row_chunksize", type=int, default=8)
    parser.add_argument("--layer_range", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_stats", action="store_true")
    args = parser.parse_args()

    # collect_activation_stats uses this flag to decide whether variance is needed.
    args.rbvt_lambda = 1.0

    if args.mode == "stats":
        collect_activation_stats(args)
    else:
        quantize_bv_sq(args)


if __name__ == "__main__":
    main()
