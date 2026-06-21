#!/usr/bin/env python3
"""
run_benchmark_baselines.py

Wrapper to run gptvq_rbvt_benchmark.py with both baseline options and compare results.
This complements debug_ncc_bias_mae_mse.py by running full evaluation on the production benchmark.

Usage:
    python3 run_benchmark_baselines.py \
        --model-path meta-llama/Llama-3.1-8B \
        --device cuda:0 \
        --output-root ./outputs/benchmark_compare \
        --run-original \
        --run-adjusted \
        ... (other benchmark args)
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse


def run_benchmark(baseline: str, args_list: list, output_root: Path):
    """Run gptvq_rbvt_benchmark.py with specified baseline."""
    baseline_output = output_root / f"baseline_{baseline}"
    baseline_output.mkdir(parents=True, exist_ok=True)
    
    cmd = ["python3", "gptvq_rbvt_benchmark.py", "--output-root", str(baseline_output)]
    cmd.extend(args_list)
    
    # Only add --baseline if debug script is actually using it
    # (For now, benchmark only supports 'original', but we document it)
    if baseline != "original":
        print(f"⚠️  Benchmark script uses '--baseline original' by default.")
        print(f"   For '{baseline}' baseline eval, use: debug_ncc_bias_mae_mse.py --baseline {baseline}")
        return None
    
    log_file = baseline_output / f"benchmark_{baseline}.log"
    
    print(f"\n{'='*70}")
    print(f"Running benchmark – baseline: {baseline}")
    print(f"Output: {baseline_output}")
    print(f"Log: {log_file}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    
    if result.returncode == 0:
        print(f"✓ Benchmark with baseline={baseline} completed successfully.")
        print(f"  Results saved to: {baseline_output}")
        return baseline_output
    else:
        print(f"✗ Benchmark with baseline={baseline} failed (exit code {result.returncode})")
        print(f"  See: {log_file}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run gptvq_rbvt_benchmark with multiple baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          # Run only original baseline (current default)
          python3 run_benchmark_baselines.py \\
              --model-path meta-llama/Llama-3.1-8B --device cuda:0

          # For 'adjusted' baseline evaluation, use the debug script instead:
          python3 debug_ncc_bias_mae_mse.py \\
              --model-path meta-llama/Llama-3.1-8B --device cuda:0 \\
              --baseline adjusted

          # Full comparison workflow:
          # 1. Run benchmark (original baseline only)
          python3 run_benchmark_baselines.py ...
          # 2. Run debug for adjusted baseline
          python3 debug_ncc_bias_mae_mse.py --baseline adjusted ...
        """),
    )
    
    # Pass-through arguments for gptvq_rbvt_benchmark.py
    parser.add_argument("--model-path", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="./outputs/benchmark_compare")
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--wbits", type=int, default=3)
    parser.add_argument("--ncc-budget-p", type=float, default=0.02)
    parser.add_argument("--ncc-sweeps", type=int, default=1)
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--lm-eval-tasks", nargs="+",
                       default=["arc_easy", "arc_challenge", "hellaswag", "winogrande",
                               "piqa", "openbookqa", "boolq", "lambada_openai"])
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--run-original", action="store_true", default=True,
                       help="Run benchmark with baseline=original (default)")
    parser.add_argument("--run-adjusted", action="store_true",
                       help="Run debug script with baseline=adjusted (recommended: use debug_ncc_bias_mae_mse.py directly)")
    
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Build args list for benchmark
    benchmark_args = [
        "--model-path", args.model_path,
        "--device", args.device,
        "--groupsize", str(args.groupsize),
        "--wbits", str(args.wbits),
        "--ncc-budget-p", str(args.ncc_budget_p),
        "--ncc-sweeps", str(args.ncc_sweeps),
        "--n-calib", str(args.n_calib),
        "--max-length", str(args.max_length),
        "--eval-samples", str(args.eval_samples),
        "--include-lm-eval",
        "--lm-eval-tasks", *args.lm_eval_tasks,
        "--lm-eval-batch-size", args.lm_eval_batch_size,
    ]
    
    results = {}
    
    if args.run_original:
        result = run_benchmark("original", benchmark_args, output_root)
        if result:
            results["original"] = str(result)
    
    if args.run_adjusted:
        print("\n⚠️  Benchmark script does not support --baseline adjusted.")
        print("   Please use the debug script instead:")
        print(f"   python3 debug_ncc_bias_mae_mse.py --baseline adjusted ...")
        results["adjusted"] = "Use debug_ncc_bias_mae_mse.py instead"
    
    # Save comparison summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "args": {
            "model_path": args.model_path,
            "device": args.device,
            "wbits": args.wbits,
            "groupsize": args.groupsize,
            "ncc_budget_p": args.ncc_budget_p,
            "lm_eval_tasks": args.lm_eval_tasks,
        },
    }
    
    summary_file = output_root / "comparison_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Benchmark evaluation complete.")
    print(f"Summary: {summary_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import textwrap
    main()
