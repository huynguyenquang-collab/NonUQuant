import argparse
import gc
import inspect
import json
from pathlib import Path

import torch

from quantization.eval_nonuquantfix_ppl import (
    load_dense_lut_model,
    load_tokenizer,
    register_linear_input_cast_hooks,
)


def parse_gen_kwargs(value):
    parsed = {}
    for item in str(value or "").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, raw = item.split("=", 1)
        raw = raw.strip()
        if raw.lower() in {"true", "false"}:
            parsed[key.strip()] = raw.lower() == "true"
            continue
        try:
            parsed[key.strip()] = int(raw)
            continue
        except ValueError:
            pass
        try:
            parsed[key.strip()] = float(raw)
            continue
        except ValueError:
            pass
        parsed[key.strip()] = raw
    return parsed


def make_json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return make_json_safe(item())
        except Exception:
            pass
    return repr(value)


def summarize_results(payload):
    summary = {}
    for section_name in ("results", "groups"):
        section = payload.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for task_name, metrics in section.items():
            if not isinstance(metrics, dict):
                continue
            task_summary = summary.setdefault(task_name, {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                    task_summary[metric_name] = metric_value
    return summary


def main():
    parser = argparse.ArgumentParser(description="lm-eval for dense LUT checkpoints.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lut_folder", required=True)
    parser.add_argument("--model_type", default="llama", choices=["llama", "mistral", "opt", "qwen"])
    parser.add_argument("--dense_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="+", default=["mmlu", "gsm8k"])
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--output_dir", default="outputs/dense_lut_lm_eval")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--gen_kwargs", default="")
    parser.add_argument("--output_file", default="")
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dense_dtype]
    tokenizer = load_tokenizer(args.model, args.hf_token)
    model = load_dense_lut_model(
        args.model,
        args.lut_folder,
        args.model_type,
        args.device,
        dtype,
        args.hf_token,
    )
    hooks = register_linear_input_cast_hooks(model)
    try:
        import lm_eval

        try:
            model_lm = lm_eval.models.huggingface.HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                device=args.device,
                batch_size=args.batch_size,
                trust_remote_code=True,
            )
        except TypeError:
            model_lm = lm_eval.models.huggingface.HFLM(
                model,
                tokenizer=tokenizer,
                device=args.device,
            )
        eval_kwargs = {
            "model": model_lm,
            "tasks": args.tasks,
            "num_fewshot": args.num_fewshot,
            "limit": args.limit,
            "log_samples": False,
        }
        gen_kwargs = parse_gen_kwargs(args.gen_kwargs)
        params = inspect.signature(lm_eval.simple_evaluate).parameters
        if gen_kwargs and "gen_kwargs" in params:
            eval_kwargs["gen_kwargs"] = gen_kwargs
        elif gen_kwargs:
            print(f"lm_eval.simple_evaluate does not accept gen_kwargs; ignoring {gen_kwargs}")
        payload = lm_eval.simple_evaluate(**eval_kwargs)
        payload = {
            "tasks": args.tasks,
            "summary": summarize_results(payload),
            "raw": make_json_safe(payload),
            "lut_folder": args.lut_folder,
            "model": args.model,
            "gen_kwargs": gen_kwargs,
        }
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        raw_path = Path(args.output_dir) / f"{args.run_name or Path(args.lut_folder).name}.json"
        raw_path.write_text(
            json.dumps(make_json_safe(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    finally:
        for handle in hooks:
            handle.remove()
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
