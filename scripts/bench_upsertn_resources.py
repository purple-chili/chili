#!/usr/bin/env python3
"""Run isolated Linux upsertn resource benchmarks (Python standard library only).

Build: cargo bench --locked -p chili-core --bench upsertn --no-run
Run: python scripts/bench_upsertn_resources.py --binary target/release/deps/upsertn-<hash>

Each cell/method gets a separate full-column correctness check, then fresh
processes for each measured repetition. CPU is process-wide user+system time;
RSS includes the runtime, resident input batch, table, and allocator retention.
"""

import argparse
import csv
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time


METHODS = ("upsertn", "value_upsert_take", "symbol_upsert_take")


def summarize(rows):
    result = []
    keys = sorted({(r.get("variant", "current"), r["batch"], r["keep"], r["method"]) for r in rows})
    metrics = (
        "wall_us_per_update", "cpu_us_per_update", "cpu_percent",
        "peak_rss_mib", "fixture_rss_mib", "start_rss_mib", "end_rss_mib",
        "peak_over_fixture_mib", "million_input_rows_per_second",
    )
    for variant, batch, keep, method in keys:
        samples = [r for r in rows if (r.get("variant", "current"), r["batch"], r["keep"], r["method"]) == (variant, batch, keep, method)]
        entry = dict(variant=variant, batch=batch, keep=keep, method=method, repetitions=len(samples))
        for metric in metrics:
            values = [r[metric] for r in samples]
            entry[metric] = statistics.median(values)
            entry[metric + "_min"] = min(values)
            entry[metric + "_max"] = max(values)
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--baseline-binary", type=Path, help="Interleave a saved baseline executable with the current binary")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--batches", nargs="+", type=int, default=[1000, 10000, 100000, 1000000])
    parser.add_argument("--keeps", nargs="+", type=int, default=[5000, 50000, 500000])
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("tmp/upsertn-resources"))
    args = parser.parse_args()
    if args.seconds < 1 or args.repeats < 1 or min(args.batches + args.keeps) < 1:
        parser.error("seconds must be >=1; repetitions and row counts must be positive")
    binary = args.binary.resolve(strict=True)
    variants = {"current": binary}
    if args.baseline_binary:
        variants["baseline"] = args.baseline_binary.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "raw.jsonl").exists():
        parser.error("output already contains raw.jsonl; choose a fresh output directory")
    clock_ticks = os.sysconf("SC_CLK_TCK")
    cpu_model = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    metadata = {
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "binary": str(binary), "binary_sha256": hashlib.file_digest(binary.open("rb"), "sha256").hexdigest(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "platform": platform.platform(), "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "memory_total_kib": int(Path("/proc/meminfo").read_text().splitlines()[0].split()[1]),
        "clock_ticks_per_second": clock_ticks, "seconds_per_sample": args.seconds,
        "repetitions": args.repeats, "batches": args.batches, "keeps": args.keeps,
        "methods": args.methods, "warmup_seconds": 0.25, "seed": 20260920,
        "variants": {name: {"binary": str(path), "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()} for name, path in variants.items()},
        "environment": {k: os.environ.get(k) for k in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "MALLOC_CONF")},
        "schema": "seq:Int64, symbol:Categorical(4 symbols), price:Float64, qty:Int64",
        "methodology": "Full retained table; same resident batch reused; warm parse cache; fresh process per sample; sequential execution; median of repeats; Linux VmHWM reset after warmup; process CPU from /proc/self/stat; RSS is not heap allocation size.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cases = list(itertools.product(args.batches, args.keeps, args.methods, variants))

    def run(case, mode):
        batch, keep, method, variant = case
        command = [str(variants[variant]), "--resources", str(batch), str(keep), method,
                   str(args.seconds), str(clock_ticks), mode]
        proc = subprocess.run(command, text=True, capture_output=True, timeout=180, check=True)
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        row["variant"] = variant
        return row

    for i, case in enumerate(cases, 1):
        assert run(case, "verify")["verified"]
        print(f"verify {i}/{len(cases)}: {case}", flush=True)

    rng = random.Random(metadata["seed"])
    samples = []
    with (args.output / "raw.jsonl").open("w") as raw:
        for repeat in range(args.repeats):
            shuffled = cases.copy()
            rng.shuffle(shuffled)
            for case in shuffled:
                row = run(case, "measure")
                row["repeat"] = repeat + 1
                row["measured_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                cpu = row["user_cpu_seconds"] + row["system_cpu_seconds"]
                row["wall_us_per_update"] = row["wall_seconds"] * 1e6 / row["updates"]
                row["cpu_us_per_update"] = cpu * 1e6 / row["updates"]
                row["cpu_percent"] = 100 * cpu / row["wall_seconds"]
                row["million_input_rows_per_second"] = row["batch"] * row["updates"] / row["wall_seconds"] / 1e6
                for name in ("fixture", "start", "peak", "end"):
                    row[f"{name}_rss_mib"] = row[f"{name}_rss_kib"] / 1024
                row["peak_over_fixture_mib"] = row["peak_rss_mib"] - row["fixture_rss_mib"]
                samples.append(row)
                raw.write(json.dumps(row) + "\n")
                raw.flush()
                print(f"sample {len(samples)}/{len(cases) * args.repeats}: "
                      f"batch={row['batch']} keep={row['keep']} {row['variant']}/{row['method']} "
                      f"{row['wall_us_per_update']:.1f} us/update "
                      f"CPU={row['cpu_percent']:.1f}% peak={row['peak_rss_mib']:.1f} MiB", flush=True)
                # Give the machine a small break between sustained samples.
                time.sleep(0.05)
    summary = summarize(samples)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "summary.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    metadata["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
