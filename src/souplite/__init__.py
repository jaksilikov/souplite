"""
SoupLite — Ultra-lightweight LLM fine-tuning & post-training CLI & UI (optimized for <= 2GB RAM).

Author & Lead Developer: Muhtar Jaksilikov
Inspired by the concepts of Soup (by Makazhan Alpamys).
Specially re-engineered to run LLM fine-tuning and post-training on machines with up to 2 GB RAM.
"""

from souplite.utils.low_ram import setup_low_ram_environment
from souplite.utils.memory_opt import LowRAMContext, purge_memory

setup_low_ram_environment()

__version__ = "0.75.0"
__author__ = "Muhtar Jaksilikov"
__all__ = ["setup_low_ram_environment", "LowRAMContext", "purge_memory", "__version__", "__author__"]
