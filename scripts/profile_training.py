"""Profile the retail demand forecasting training pipeline.

This script runs the main training pipeline with CPU profiling and memory
tracking so profiling can be reproduced outside notebooks.

Outputs:
    reports/profiling/training_cpu_profile.prof
    reports/profiling/training_cpu_profile.txt
    reports/profiling/training_memory_usage.txt
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

from memory_profiler import memory_usage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlops_se489.logging_config import get_logger, setup_logging  # noqa: E402
from mlops_se489.models.train_model import run_training  # noqa: E402

setup_logging()
logger = get_logger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports" / "profiling"
CPU_PROFILE_PATH = REPORTS_DIR / "training_cpu_profile.prof"
CPU_PROFILE_TXT_PATH = REPORTS_DIR / "training_cpu_profile.txt"
MEMORY_PROFILE_PATH = REPORTS_DIR / "training_memory_usage.txt"


def _run_training_for_profile() -> str:
    """Run the training pipeline for profiling.

    Returns:
        Path to the saved champion model.
    """
    return run_training()


def profile_cpu() -> None:
    """Run cProfile on the training pipeline and save raw/text results."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    logger.info("Starting CPU profiling for training pipeline")

    start_time = time.perf_counter()
    profiler.enable()
    _run_training_for_profile()
    profiler.disable()
    elapsed = time.perf_counter() - start_time

    profiler.dump_stats(CPU_PROFILE_PATH)

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(40)

    CPU_PROFILE_TXT_PATH.write_text(stream.getvalue(), encoding="utf-8")

    logger.info("CPU profiling completed in %.2f seconds", elapsed)
    logger.info("CPU profile saved to %s", CPU_PROFILE_PATH)
    logger.info("CPU profile text saved to %s", CPU_PROFILE_TXT_PATH)


def profile_memory() -> None:
    """Run memory_profiler on the training pipeline and save usage results."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting memory profiling for training pipeline")

    start_time = time.perf_counter()
    memory_values = memory_usage(
        (_run_training_for_profile, (), {}),
        interval=1.0,
        timeout=None,
        retval=False,
    )
    elapsed = time.perf_counter() - start_time

    peak_memory = max(memory_values)
    start_memory = memory_values[0]
    end_memory = memory_values[-1]

    output = [
        "Training Memory Profile",
        "=======================",
        f"Elapsed time seconds: {elapsed:.2f}",
        f"Start memory MiB: {start_memory:.2f}",
        f"Peak memory MiB: {peak_memory:.2f}",
        f"End memory MiB: {end_memory:.2f}",
        f"Memory increase MiB: {peak_memory - start_memory:.2f}",
        "",
        "Samples MiB:",
    ]

    output.extend(f"{idx}: {value:.2f}" for idx, value in enumerate(memory_values))

    MEMORY_PROFILE_PATH.write_text("\n".join(output), encoding="utf-8")

    logger.info("Memory profiling completed in %.2f seconds", elapsed)
    logger.info("Peak memory usage: %.2f MiB", peak_memory)
    logger.info("Memory profile saved to %s", MEMORY_PROFILE_PATH)


def main() -> None:
    """Run CPU and memory profiling."""
    profile_cpu()
    profile_memory()
    logger.info("Profiling finished successfully")


if __name__ == "__main__":
    main()
