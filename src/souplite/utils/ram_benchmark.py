"""
Sub-2GB RAM Benchmark & Memory Allocator Diagnostics Utility for SoupLite.
Author: Muhtar Jaksilikov
"""

import time
import os
import gc
import psutil
import platform
import ctypes
from typing import Dict, Any

class RAMBenchmark:
    """Diagnostic tool to profile system RAM budget, GC efficiency, and PyTorch memory allocations."""

    @staticmethod
    def get_system_memory_info() -> Dict[str, Any]:
        """Collect current process RSS memory, total system RAM, and OS virtual memory statistics."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        vm = psutil.virtual_memory()

        return {
            "pid": os.getpid(),
            "process_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            "process_vsz_mb": round(mem_info.vms / (1024 * 1024), 2),
            "system_total_gb": round(vm.total / (1024**3), 2),
            "system_used_gb": round(vm.used / (1024**3), 2),
            "system_available_gb": round(vm.available / (1024**3), 2),
            "system_percent_used": vm.percent,
            "within_2gb_budget": (mem_info.rss / (1024 * 1024)) <= 2048,
        }

    @staticmethod
    def run_malloc_trim_benchmark() -> Dict[str, Any]:
        """Test C-heap memory reclamation using libc.malloc_trim(0)."""
        process = psutil.Process(os.getpid())
        before_rss = process.memory_info().rss / (1024 * 1024)

        # Allocate temporary buffer
        temp_data = [bytearray(1024 * 1024) for _ in range(50)]
        during_rss = process.memory_info().rss / (1024 * 1024)

        # Free python references
        del temp_data
        gc.collect()

        after_gc_rss = process.memory_info().rss / (1024 * 1024)

        # Call C malloc_trim
        trim_supported = False
        if platform.system() == "Linux":
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
                trim_supported = True
            except Exception:
                pass

        after_trim_rss = process.memory_info().rss / (1024 * 1024)

        return {
            "alloc_size_mb": 50,
            "before_rss_mb": round(before_rss, 2),
            "peak_rss_mb": round(during_rss, 2),
            "after_gc_rss_mb": round(after_gc_rss, 2),
            "after_trim_rss_mb": round(after_trim_rss, 2),
            "freed_by_trim_mb": round(after_gc_rss - after_trim_rss, 2),
            "trim_supported": trim_supported,
        }

    @staticmethod
    def run_full_diagnostics() -> Dict[str, Any]:
        """Run complete low-RAM health diagnostics for SoupLite."""
        start_time = time.time()
        sys_info = RAMBenchmark.get_system_memory_info()
        trim_info = RAMBenchmark.run_malloc_trim_benchmark()
        elapsed = time.time() - start_time

        return {
            "status": "PASS" if sys_info["within_2gb_budget"] else "WARNING",
            "execution_time_sec": round(elapsed, 4),
            "memory": sys_info,
            "trim_benchmark": trim_info,
        }


def print_ram_report():
    """Print clean formatted CLI report for RAM Benchmark."""
    diag = RAMBenchmark.run_full_diagnostics()
    print("=" * 60)
    print(" 🚀 SoupLite Sub-2GB RAM Diagnostics & Benchmark Report")
    print("=" * 60)
    print(f" Status:              {diag['status']}")
    print(f" Process RSS Memory:  {diag['memory']['process_rss_mb']} MB")
    print(f" System Total RAM:    {diag['memory']['system_total_gb']} GB")
    print(f" System Available:    {diag['memory']['system_available_gb']} GB")
    print(f" Within 2GB Budget:   {diag['memory']['within_2gb_budget']}")
    print("-" * 60)
    print(" 🛠️  malloc_trim C-Heap Test:")
    print(f" Peak Allocated:      {diag['trim_benchmark']['peak_rss_mb']} MB")
    print(f" RSS After GC:        {diag['trim_benchmark']['after_gc_rss_mb']} MB")
    print(f" RSS After Trim:      {diag['trim_benchmark']['after_trim_rss_mb']} MB")
    print(f" Freed by Trim:       {diag['trim_benchmark']['freed_by_trim_mb']} MB")
    print("=" * 60)


if __name__ == "__main__":
    print_ram_report()
