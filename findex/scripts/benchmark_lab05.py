
"""Lab 05 benchmark: serial, threads and processes.

Supports Python 3.13 with and without GIL.
Windows spawn-safe entry point.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import psutil

from findex.parallel import build_parallel, gil_status

MODES = ("serial", "threads", "processes")


def plot_results(csv_path: Path, png_path: Path) -> bool:
    """Build a graph from an existing CSV file."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib відсутній. CSV збережено, графік пропущено.")
        return False

    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    if not rows:
        raise ValueError(f"CSV порожній: {csv_path}")

    counts = sorted({int(row["workers"]) for row in rows})

    for mode in MODES:
        subset = sorted(
            (row for row in rows if row["executor"] == mode),
            key=lambda row: int(row["workers"]),
        )
        if subset:
            plt.plot(
                [int(row["workers"]) for row in subset],
                [float(row["speedup"]) for row in subset],
                marker="o",
                label=mode,
            )

    plt.plot(counts, counts, linestyle="--", label="ideal")
    plt.xlabel("Workers")
    plt.ylabel("Speedup vs serial 1")
    plt.title(
        f"Findex Lab 05 - Python {rows[0]['python']}, "
        f"GIL={rows[0]['gil_enabled']}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"Графік збережено: {png_path}")
    return True


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark parallel Findex indexing"
    )
    parser.add_argument("corpus", nargs="?", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--out", type=Path, default=Path("results/lab05")
    )
    parser.add_argument(
        "--plot-only",
        type=Path,
        metavar="CSV",
        help="Build PNG from CSV without new measurements",
    )

    args = parser.parse_args()

    if args.plot_only is not None:
        input_csv: Path = args.plot_only
        png_path = input_csv.with_name(
            input_csv.stem.replace("benchmark_", "speedup_") + ".png"
        )
        plot_results(input_csv, png_path)
        return

    if args.corpus is None:
        parser.error("corpus is required unless --plot-only is given")

    if args.runs < 3:
        parser.error("--runs must be at least 3")

    args.out.mkdir(parents=True, exist_ok=True)

    cpu_count = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, 8, cpu_count})
    rows: list[dict[str, str | int | float]] = []
    gil = gil_status()

    print(
        f"CPU: {platform.processor()} | logical CPUs: {cpu_count}"
    )
    print(
        f"Python: {sys.version.split()[0]} | GIL enabled: {gil}"
    )

    for executor in MODES:
        for n in worker_counts:
            samples: list[
                tuple[float, float, float, float]
            ] = []

            for repeat in range(args.runs):
                parent = psutil.Process()
                cpu_before = parent.cpu_times()
                rss_before = parent.memory_info().rss

                start = time.perf_counter()

                index, metrics = build_parallel(
                    args.corpus,
                    workers=n,
                    executor=executor,
                    with_positions=True,
                )

                wall = time.perf_counter() - start
                cpu_after = parent.cpu_times()

                parent_cpu = max(
                    0.0,
                    (cpu_after.user + cpu_after.system)
                    - (cpu_before.user + cpu_before.system),
                )

                if executor == "processes":
                    cpu_seconds = (
                        parent_cpu + metrics.worker_cpu_seconds
                    )
                else:
                    cpu_seconds = max(
                        parent_cpu, metrics.worker_cpu_seconds
                    )

                rss_sampled = max(
                    rss_before,
                    parent.memory_info().rss,
                )

                samples.append(
                    (
                        wall,
                        cpu_seconds,
                        rss_sampled / (1024 * 1024),
                        metrics.merge_seconds,
                    )
                )

                print(
                    f"{executor:9s} {n:3d} "
                    f"run {repeat + 1}/{args.runs}: "
                    f"{wall:.3f}s, docs={index.num_docs}",
                    flush=True,
                )

            median_values = [
                statistics.median(sample[i] for sample in samples)
                for i in range(4)
            ]

            rows.append(
                {
                    "executor": executor,
                    "workers": n,
                    "wall_seconds": median_values[0],
                    "cpu_seconds_estimate": median_values[1],
                    "parent_rss_mib_sampled": median_values[2],
                    "merge_seconds": median_values[3],
                    "speedup": 0.0,
                    "python": sys.version.split()[0],
                    "gil_enabled": gil,
                    "cpu": platform.processor(),
                    "logical_cpus": cpu_count,
                }
            )

    baseline = next(
        float(row["wall_seconds"])
        for row in rows
        if row["executor"] == "serial"
        and row["workers"] == 1
    )

    for row in rows:
        row["speedup"] = round(
            baseline / float(row["wall_seconds"]), 4
        )

    suffix = "gil_off" if gil == "False" else "gil_on"
    csv_path = args.out / f"benchmark_{suffix}.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV збережено: {csv_path}")

    plot_results(
        csv_path,
        args.out / f"speedup_{suffix}.png",
    )

    print(
        "Увага: CPU time є оцінкою; RSS вимірюється "
        "вибірково, не як гарантований пік."
    )


if __name__ == "__main__":
    run()
