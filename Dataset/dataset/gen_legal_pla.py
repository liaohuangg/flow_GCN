#!/usr/bin/env python3
"""
Generate legal placements by running ILP search on all JSON inputs under ./input_test.

Note: the solve strategy is configured inside ilp_search_chiplet.py.
This script simply iterates inputs and calls search_multiple_solutions().
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator, List, Optional


class _TeeStream:
    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        n = 0
        for st in self._streams:
            n = st.write(s)
            st.flush()
        return n

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


@contextmanager
def _tee_stdout_stderr(log_fp: IO[str], also_console: bool) -> Iterator[None]:
    old_out, old_err = sys.stdout, sys.stderr
    try:
        if also_console:
            sys.stdout = _TeeStream(old_out, log_fp)  # type: ignore[assignment]
            sys.stderr = _TeeStream(old_err, log_fp)  # type: ignore[assignment]
        else:
            sys.stdout = log_fp  # type: ignore[assignment]
            sys.stderr = log_fp  # type: ignore[assignment]
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _console_print(msg: str) -> None:
    """
    Always print to the original console stream.
    Useful when --no-console is enabled but we still want a short progress summary.
    """
    try:
        print(msg, file=sys.__stdout__, flush=True)
    except Exception:
        # Fallback to regular print if console stream is unavailable
        print(msg, flush=True)


def generate_for_directory(
    input_dir: Path,
    log_dir: Path,
    start_idx: int = 1,
    end_idx: Optional[int] = None,
    also_console: bool = True,
) -> None:
    """
    Read all JSON files under input_dir and run ILP search for each.
    """
    import ilp_search_chiplet as solver_mod

    if start_idx < 1:
        raise SystemExit("--start must be >= 1")
    if end_idx is not None and end_idx < start_idx:
        raise SystemExit("--end must be >= --start")

    log_dir.mkdir(parents=True, exist_ok=True)

    # If --end is not provided, infer it from existing system_*.json files.
    if end_idx is None:
        max_i: Optional[int] = None
        for p in input_dir.glob("system_*.json"):
            stem = p.stem  # system_{i}
            try:
                i_str = stem.split("_", 1)[1]
                i_val = int(i_str)
            except Exception:
                continue
            if max_i is None or i_val > max_i:
                max_i = i_val
        if max_i is None:
            raise SystemExit(f"No system_*.json files found in: {input_dir}")
        end_idx = max_i

    total = end_idx - start_idx + 1

    print(f"[gen_legal_pla] Input dir: {input_dir}")
    print(f"[gen_legal_pla] Log dir: {log_dir}")
    print(f"[gen_legal_pla] Range: system_i, i={start_idx}..{end_idx} (total {total})")

    # Directly iterate system_{i}.json without skipping.
    for i in range(start_idx, end_idx + 1):
        json_path = input_dir / f"system_{i}.json"
        if not json_path.exists():
            raise SystemExit(f"Missing input JSON: {json_path}")

        # Use system_i as log prefix (no .json in name)
        log_path = log_dir / f"{json_path.stem}.log"
        sols = []
        with log_path.open("w", encoding="utf-8") as fp:
            with _tee_stdout_stderr(fp, also_console=also_console):
                try:
                    pos = i - start_idx + 1
                    print(f"\n[{pos}/{total}] Solving: {json_path.name}")
                    print(f"[gen_legal_pla] Logging to: {log_path}")
                    sols = solver_mod.search_multiple_solutions(
                        num_solutions=1,
                        input_json_path=str(json_path),
                    )
                    gap_val = None
                    if sols:
                        gap_val = getattr(sols[0], "mip_gap", None)
                    if gap_val is not None:
                        print(f"[gen_legal_pla] Random MIPGap used: {gap_val:.6f}")

                    print(f"  Returned {len(sols)} solution(s)")
                    if sols:
                        s0 = sols[0]
                        print(
                            f"[gen_legal_pla] Done: status={getattr(s0, 'status', None)}, "
                            f"solve_time={getattr(s0, 'solve_time', None)}s, "
                            f"objective={getattr(s0, 'objective_value', None)}"
                        )
                except Exception as e:
                    print(f"[gen_legal_pla] ERROR: case {i} failed with exception: {e}")
                    import traceback

                    traceback.print_exc()
                    sols = []

        # Always show a short summary in the terminal (even with --no-console).
        gap_val_console = None
        if sols:
            gap_val_console = getattr(sols[0], "mip_gap", None)
        status_console = getattr(sols[0], "status", None) if sols else None
        if (not sols) or status_console == "NoSolution":
            _console_print(f"[{i}] NoSolution (continue)")
            continue

        _console_print(f"[{i}] {json_path.name} finished; solutions={len(sols)}")
        if gap_val_console is not None:
            _console_print(f"[{i}] Random MIPGap used: {gap_val_console:.6f}")
        if sols:
            s0c = sols[0]
            _console_print(
                f"[{i}] status={getattr(s0c, 'status', None)}, "
                f"solve_time={getattr(s0c, 'solve_time', None)}s, "
                f"objective={getattr(s0c, 'objective_value', None)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate legal placements for JSON inputs.")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing input *.json files.")
    parser.add_argument("--start", type=int, default=1, help="Start idx (1-based, inclusive) for JSON enumeration.")
    parser.add_argument("--end", type=int, default=None, help="End idx (1-based, inclusive) for JSON enumeration.")
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Write logs to file only (suppress console output).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    input_dir = Path(args.input_dir).resolve() if args.input_dir else (script_dir / "input_test")
    log_dir = script_dir / "output" / "log"
    generate_for_directory(
        input_dir,
        log_dir=log_dir,
        start_idx=args.start,
        end_idx=args.end,
        also_console=(not args.no_console),
    )


if __name__ == "__main__":
    main()

#python gen_legal_pla.py --start 1 --end 20 --no-console
# python gen_legal_pla.py --start 4155 --end 8000 --no-console > /root/placement/flow_GCN/Dataset/dataset/dataset.log 2>&1