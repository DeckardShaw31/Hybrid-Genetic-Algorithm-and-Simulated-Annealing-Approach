"""Measure simulation throughput with W parallel workers, to estimate experiment wall time.

  python Phase1_rerun/benchmark_throughput.py --workers 15 --reps 100
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

PLANS = {"medium": ([4, 5, 3, 0, 1, 0, 0, 0, 0], [3, 4, 2, 0, 1, 0, 0, 0, 0]),
         "large": ([8, 9, 5, 0, 1, 0, 0, 0, 0, 0, 0, 0], [6, 7, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0])}


def one(args):
    instance, reps, seed = args
    import hgasa
    import instances
    inst = instances.get_instance(instance)
    ev = hgasa.Evaluator("nurse_physician", inst, 0, reps, seed)
    t0 = time.perf_counter()
    ev(np.array(PLANS[instance]))
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    for instance in ("medium", "large"):
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(one, [(instance, 5, s) for s in range(args.workers)]))   # warm-up (imports)
            t0 = time.perf_counter()
            secs = list(pool.map(one, [(instance, args.reps, 100 + s) for s in range(args.workers)]))
            wall = time.perf_counter() - t0
        per_eval_500 = np.mean(secs) / args.reps * 500
        print(f"{instance}: {args.workers} parallel evaluations of {args.reps} reps in {wall:.1f} s wall; "
              f"one 500-rep evaluation takes ~{per_eval_500:.1f} s per worker under this load")


if __name__ == "__main__":
    main()
