import json
from pathlib import Path
from typing import Union

from repro.sec7_2_lm_decode.shared import SUMMARY_RESULTS_FILE

""" Shared functions for plotting the results from the LM decode experiments.
"""


def gather_summary_results(results_path: Union[str, Path]) -> list[dict]:
    results_path = Path(results_path)
    # Collect all summary results from individual files
    all_summary_results = []
    for bench_dir in results_path.iterdir():
        summary_results_path = bench_dir / SUMMARY_RESULTS_FILE
        if summary_results_path.exists():
            with open(summary_results_path, "r") as f:
                result = json.load(f)
            all_summary_results.append(result)

    # Sort results by name for consistent output
    all_summary_results.sort(key=lambda x: x["name"])

    return all_summary_results
