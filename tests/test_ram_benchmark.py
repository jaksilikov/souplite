"""Unit tests for SoupLite RAM Benchmark Utility."""

import pytest
from souplite.utils.ram_benchmark import RAMBenchmark

def test_get_system_memory_info():
    info = RAMBenchmark.get_system_memory_info()
    assert "process_rss_mb" in info
    assert "system_total_gb" in info
    assert "within_2gb_budget" in info
    assert isinstance(info["within_2gb_budget"], bool)

def test_run_malloc_trim_benchmark():
    trim_info = RAMBenchmark.run_malloc_trim_benchmark()
    assert "before_rss_mb" in trim_info
    assert "peak_rss_mb" in trim_info
    assert "after_trim_rss_mb" in trim_info
    assert trim_info["peak_rss_mb"] >= trim_info["before_rss_mb"]

def test_run_full_diagnostics():
    diag = RAMBenchmark.run_full_diagnostics()
    assert diag["status"] in ["PASS", "WARNING"]
    assert "memory" in diag
    assert "trim_benchmark" in diag
