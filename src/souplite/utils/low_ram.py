"""
SoupLite Low-RAM Optimization Utilities.
Designed to run LLM workflows on low-memory machines (<= 2 GB RAM).

Original base project created by Muhtar Jaksilikov (https://github.com/MuhtarJaksilikov/Soup).
"""

import os
import sys
import gc
import ctypes
import platform
import logging

logger = logging.getLogger("souplite.low_ram")

# System RAM Threshold in GB for low-RAM mode
LOW_RAM_THRESHOLD_GB = 2.5
DEFAULT_2GB_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_135M_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

def setup_low_ram_environment() -> None:
    """Set low-overhead environment variables before importing heavy frameworks."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "100000")
    os.environ.setdefault("PYTHONMALLOC", "malloc")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # Avoid spawning excessive thread pools on 1-2 core low-RAM VPS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def get_total_system_ram_gb() -> float:
    """Get total system physical RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass

    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return (kb * 1024) / (1024 ** 3)
        except Exception:
            pass

    return 4.0  # Safe default assumption if unmeasurable


def get_available_system_ram_gb() -> float:
    """Get currently available system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass

    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                mem_avail = 0
                for line in f:
                    if line.startswith("MemAvailable:"):
                        mem_avail = int(line.split()[1])
                        return (mem_avail * 1024) / (1024 ** 3)
        except Exception:
            pass

    return get_total_system_ram_gb() * 0.5


def is_low_ram_system(threshold_gb: float = LOW_RAM_THRESHOLD_GB) -> bool:
    """Check if the system has <= threshold_gb RAM."""
    return get_total_system_ram_gb() <= threshold_gb


def force_garbage_collection() -> None:
    """Aggressively collect Python garbage, empty PyTorch cache, and trim malloc heap."""
    gc.collect()
    
    # Clear PyTorch CUDA cache if loaded
    if "torch" in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    # Call glibc malloc_trim(0) on Linux to return unused heap blocks to Linux OS kernel
    if platform.system() == "Linux":
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass


def get_low_ram_model_preset(model_override: str | None = None) -> str:
    """Pick an appropriate model for low RAM operation (< 2 GB RAM)."""
    if model_override:
        return model_override
    
    total_ram = get_total_system_ram_gb()
    if total_ram <= 1.5:
        return FALLBACK_135M_MODEL
    elif total_ram <= 2.5:
        return DEFAULT_2GB_MODEL
    return DEFAULT_2GB_MODEL


def get_recommended_dataloader_kwargs() -> dict:
    """Return dataloader kwargs tuned for low-RAM setups."""
    if is_low_ram_system():
        return {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }
    return {
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
    }


# Auto-setup on module load
setup_low_ram_environment()
