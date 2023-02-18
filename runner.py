import multiprocessing
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import cycle
from logging import INFO, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path
from typing import Any

TEST_CMD = ["cargo", "run", "--release", "--bin", "tester"]
FIDGET = cycle("⢄⢂⢁⡁⡈⡐⡠")


@dataclass
class TestResult:
    cost: int


def setup_logger(name: str) -> Logger:
    logger = getLogger(name)
    logger.setLevel(INFO)

    ch = StreamHandler()
    ch.setLevel(INFO)
    ch_formatter = Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%H:%M:%S"
    )
    ch.setFormatter(ch_formatter)

    logger.addHandler(ch)
    return logger


def init_worker(shared_progress_counter: Any) -> None:
    global progress_counter
    progress_counter = shared_progress_counter


def run_and_get_result(
    exe_tool_inout_file: tuple[Path, Path, Path, Path]
) -> TestResult:
    exe_path, tool_path, in_file, out_file = exe_tool_inout_file
    cmd = TEST_CMD + [exe_path.resolve()]
    res = subprocess.run(cmd, capture_output=True, cwd=tool_path, stdin=in_file.open("r"))
    global progress_counter
    progress_counter.value += 1
    out_file.write_text(res.stdout.decode("utf-8"))
    *_, cost = res.stderr.decode("utf-8").strip().split("=")
    return TestResult(int(cost))


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("exe", metavar="X", help="Executable")
    arg_parser.add_argument(
        "tool_dir", metavar="TOOL", help="Path to local tester directory"
    )
    arg_parser.add_argument("in_dir", metavar="IN", help="Path to input directory")
    arg_parser.add_argument("out_dir", metavar="OUT", help="Path to output directory")
    arg_parser.add_argument(
        "--verbose", action="store_true", help="Show detailed test result"
    )
    args = arg_parser.parse_args()

    lg = setup_logger("runner")

    exe_path = Path(args.exe)
    tool_path = Path(args.tool_dir).resolve()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir()

    in_files = sorted(filter(lambda p: p.suffix == ".txt", in_dir.iterdir()))
    n = len(in_files)
    in_out_files = [(in_file, out_dir / in_file.name) for in_file in in_files]

    run_arguments = [
        (exe_path, tool_path, in_file, out_file) for in_file, out_file in in_out_files
    ]

    # Limit the number of processes to max(cpu_count - CPU_RESERVE, 1),
    # to make room for processes which have nothing to do with this test.
    # If this approach is not taken, test scores will be unexpectedly affected.
    CPU_RESERVE = 4
    cpu_count = os.cpu_count()
    assert cpu_count is not None
    processes_count = max(cpu_count - CPU_RESERVE, 1)
    progress_counter: Any = multiprocessing.Value("i", 0)
    lg.info(f"#test_cases = {n}")
    lg.info(f"#processes = {processes_count}")
    with multiprocessing.Pool(
        initializer=init_worker, initargs=(progress_counter,), processes=processes_count
    ) as pool:
        async_result = pool.map_async(run_and_get_result, run_arguments)
        # periodically check the progress
        while not async_result.ready():
            print(
                f"> {next(FIDGET)} {progress_counter.value}/{n}",
                end="\r",
                flush=True,
                file=sys.stderr,
            )
            time.sleep(0.25)

    if args.verbose:
        for t, (in_file, out_file) in zip(async_result.get(), in_out_files):
            lg.info(f"`{in_file}` => `{out_file}`: {t}")

    costs = [t.cost for t in async_result.get()]
    sum_cost = sum(costs)
    max_cost, (max_in, max_out) = max(zip(costs, in_out_files))
    min_cost, (min_in, min_out) = min(zip(costs, in_out_files))
    lg.info("----- TEST SUMMARY -----")
    lg.info(f"SUM = {sum_cost}")
    lg.info(f"AVG = {sum_cost / n}")
    lg.info(f"MAX = {max_cost} (`{max_in}` => `{max_out}`)")
    lg.info(f"MIN = {min_cost} (`{min_in}` => `{min_out}`)")
