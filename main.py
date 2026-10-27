"""Simple entry point: python main.py --quick or python main.py --full."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="54 runs, one seed, 40 simulated time units")
    mode.add_argument("--full", action="store_true", help="162 runs, three seeds, 150 simulated time units")
    parser.add_argument("--output-dir", type=Path, help="Defaults to quick/ for quick mode, current directory for full")
    parser.add_argument("--time-budget", type=float, help="Optional common simulated horizon override")
    parser.add_argument("--seeds", nargs="+", type=int, help="Optional unique nonnegative training/delay seeds")
    args = parser.parse_args()
    if args.seeds and (len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0):
        parser.error("--seeds must be unique and nonnegative")
    selected_mode = "quick" if args.quick else "full"
    output_dir = args.output_dir or Path("quick" if args.quick else ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep plotting caches writable and make the CLI work without a display.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    from experiments.run_experiments import run_suite
    from experiments.report import generate_report
    from scripts.generate_plots import generate_plots

    history, events, summary, aggregate, diagnostics, manifest, sanity = run_suite(
        selected_mode, output_dir, args.time_budget, args.seeds,
    )
    generate_plots(history, events, summary, diagnostics, output_dir / "plots")
    generate_report(summary, aggregate, diagnostics, manifest, sanity, output_dir / "results")
    print(f"\n{selected_mode.upper()} suite complete: {output_dir.resolve()}")
    print(f"Report: {(output_dir / 'results' / 'week9_summary.md').resolve()}")


if __name__ == "__main__":
    main()
