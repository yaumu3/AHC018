# AHC018

[AHC018](https://atcoder.jp/contests/ahc018) submission in Rust + parallel runner

## Prerequisites

- Python 3.10 or later
- Rust 1.42 or later
- Installed local tester (You can download it from the [Problem statement page](https://atcoder.jp/contests/ahc018/tasks/ahc018_a).)
- Generated test cases with the local tester.

## Runner script

You can run tests in parallel using `runner.py`. However, the number of parallel processes is limited to `cpu_count() - 4` to ensure that other applications do not affect the accuracy of the test results.

### Usage

Please refer to the help by `python runner.py --help`.

### Example

One-liner to build + run tests by `runner.py` when local tester is in `../ahc018-tools` relative to this document's directory.

```bash
cargo build --release && python runner.py target/release/ahc018 ../ahc018-tools ../ahc018-tools/in ../ahc018-tools/out
```
