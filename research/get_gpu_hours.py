"""
The script walks each supplied directory (recursively) for files ending in
'.log', looks for the job-summary block that appears after a successful run,
extracts the value on the line that begins with 'Run time :', and sums all
such durations. Results are printed in both seconds and hours.

This script has been written to run on the TU Ilmenau cluster with my specific setup.
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Pattern captures the numeric value before "sec." (integer or float)
RUN_TIME_RE = re.compile(r"Run time\s*:\s*(\d+(?:\.\d+)?)\s*sec\.", re.IGNORECASE)

def extract_run_time(logfile: Path) -> Optional[float]:
    """
    Return run time in seconds extracted from 'logfile' or 'None' if not found.
    """
    try:
        with logfile.open(encoding="utf-8", errors="ignore") as fh:
            # read once and scan backwards—summary is at the end of the file.
            for line in reversed(fh.readlines()):
                match = RUN_TIME_RE.search(line)
                if match:
                    return float(match.group(1))
    except (OSError, UnicodeDecodeError):
        pass # skip unreadable files gracefully
    return None

def collect_logs(directories: List[Path]) -> List[Path]:
    """
    Recursively find all '*.log' files within 'directories'.
    """
    logs: List[Path] = []
    for d in directories:
        if not d.exists():
            print(f"Skipping {d} (does not exist)")
            continue
        if d.is_file() and d.suffix == ".log":
            logs.append(d.resolve())
        else:
            logs.extend(p.resolve() for p in d.rglob("*.log"))
    return logs


def summarize(logs: List[Path]) -> Tuple[Dict[Path, float], float]:
    """
    Return mapping of 'logfile' to seconds and grand-total seconds.
    """
    per_file: Dict[Path, float] = {}
    for log in logs:
        secs = extract_run_time(log)
        if secs is not None:
            per_file[log] = secs
    total = sum(per_file.values())
    return per_file, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sum GPU run time from '.log' files produced by the job scheduler"
    )
    parser.add_argument(
        "--paths", 
        nargs="+", 
        type=Path, 
        help="Directory or '.log' file paths to scan",
        default=[
            Path("/scratch/magr4985/MLP_AddNorm"),
            Path("/scratch/magr4985/Ablation"), 
            Path("/scratch/magr4985/Ablation_Interaction"),
            Path("/scratch/magr4985/Spectra_Dim"),
            Path("/scratch/magr4985/Transformer_Scaling"),
            Path("/scratch/magr4985/Scaling_Base"),
            Path("/scratch/magr4985/Scaling_Grid"),
        ]
    )
    args = parser.parse_args()
    logs = collect_logs(args.paths)
    if not logs:
        print("No .log files found")
        return
    per_file, total_secs = summarize(logs)
    total_hours = total_secs / 3600
    total_day = total_hours / 24
    # print summary to command line
    print(f"\nFound {len(per_file)} completed jobs:")
    for path, secs in sorted(per_file.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {path}: {secs/3600:.2f} h")
    print(f"\nTotal GPU time: {total_hours:.0f} h ({total_day:.2f} days)")
    # write summary to file
    output_dir = "gpu_hours"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "gpu_hours.txt"), "w") as f:
        f.write(f"Found {len(per_file)} completed jobs:\n")
        for path, secs in sorted(per_file.items(), key=lambda kv: kv[1], reverse=True):
            f.write(f"  {path}: {secs/3600:.2f} h\n")
        f.write(f"Total GPU time: {total_hours:.0f} h ({total_day:.2f} days)")

if __name__ == "__main__":
    # working directory setup, i.e., enable relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    # run the main function
    main()
