<p align="center">
  <img src="soup.png" alt="SoupLite Logo" width="280">
</p>

<h1 align="center">🥣 SoupLite</h1>

<p align="center">
  <strong>Ultra-Lightweight LLM Fine-Tuning & Post-Training Engine Optimized for Strict Low-RAM Budgets (&le; 2 GB RAM)</strong>
</p>

<p align="center">
  <a href="https://github.com/jaksilikov/souplite"><img src="https://img.shields.io/badge/Author-Muhtar--Jaksilikov-blue.svg?style=for-the-badge&logo=github" alt="Author"></a>
  <a href="https://github.com/jaksilikov/souplite"><img src="https://img.shields.io/badge/RAM--Budget-%E2%89%A4--2GB-brightgreen.svg?style=for-the-badge" alt="RAM Budget"></a>
  <a href="https://github.com/jaksilikov/souplite"><img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg?style=for-the-badge&logo=python" alt="Python"></a>
  <a href="https://github.com/jaksilikov/souplite/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-orange.svg?style=for-the-badge" alt="License"></a>
  <a href="https://github.com/jaksilikov/souplite"><img src="https://img.shields.io/badge/Docker-2GB--Capped-blue.svg?style=for-the-badge&logo=docker" alt="Docker"></a>
</p>

<p align="center">
  <a href="#-overview--concept">Overview</a> &bull;
  <a href="#-key-technical-features">Key Features</a> &bull;
  <a href="#-comparison-original-soup-vs-souplite">Comparison</a> &bull;
  <a href="#-architecture--memory-flow">Architecture</a> &bull;
  <a href="#-installation--quickstart">Quickstart</a> &bull;
  <a href="#-benchmarks--memory-diagnostics">Benchmarks</a> &bull;
  <a href="#-docker-deployment">Docker</a> &bull;
  <a href="#-license">License</a>
</p>

---

## 👨‍💻 Overview & Concept

**SoupLite** is an ultra-lightweight, high-efficiency Python framework designed for LLM fine-tuning, post-training (SFT, DPO, GRPO, ORPO), dataset processing, and model evaluation under extreme memory constraints (**&le; 2 GB RAM**).

* **Primary Author & Lead Engineer**: **Muhtar Jaksilikov**
* **Inspiration & Attribution**: Inspired by the concepts of **Soup** (*Makazhan Alpamys*).
* **Engineering Problem**: Original Soup requires at least **4.0 GB+ RAM** to load and train unquantized/partially-quantized models like `TinyLlama-1.1B`. **SoupLite** was completely re-engineered by **Muhtar Jaksilikov** with low-level memory allocation hooks, streaming data pipelines, and C-heap garbage collection, allowing full execution within a **sub-2 GB RAM footprint**.

---

## ⚡ Key Technical Features

1. **Low-Level Memory Reclamation (`libc.so.6 malloc_trim(0)`)**:
   Forces the Linux OS C-heap memory manager to instantly release unmapped Python memory allocations back to the kernel after heavy model loading or evaluation steps.

2. **PyTorch Allocator Optimizations**:
   Pre-configures `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128` to eliminate memory fragmentation during gradient accumulation.

3. **Zero-Copy Streaming Dataset Pipelines**:
   Streams JSONL/Parquet datasets row-by-row without pre-loading entire dataset structures into memory.

4. **Low CPU Memory Loading (`low_cpu_mem_usage=True`)**:
   Loads HuggingFace model weight shards directly into designated precision structures, avoiding duplicate in-memory CPU buffers.

5. **Integrated LowRAMContext Manager**:
   Provides pythonic context manager support (`from souplite import LowRAMContext`) to automatically handle cleanup, GC collection, and C-heap trim after execution loops.

---

## 📊 Comparison: Original Soup vs. SoupLite

| Feature / Metric | Original Soup | 🥣 SoupLite (by Muhtar Jaksilikov) |
| :--- | :---: | :---: |
| **Base Memory Requirement** | ~4.0 GB+ RAM | **&le; 2.0 GB RAM** |
| **Model Loader** | Standard HF AutoModel | **Streaming Direct Shard (`low_cpu_mem_usage=True`)** |
| **C-Heap Memory Cleanup** | Python `gc.collect()` only | **C-Level `malloc_trim(0)` + PyTorch IPC Collect** |
| **DataLoader Overhead** | Multi-worker buffers | **Zero-Worker Streaming Pipeline** |
| **Background CLI Memory** | ~120 MB RAM | **< 30 MB RAM** |
| **Docker Execution Limit** | Uncapped | **Strictly capped at `2048M` (2GB)** |
| **RAM Profiling Utilities** | Basic logs | **Built-in `RAMBenchmark` suite & CLI** |

---

## 🏗️ Architecture & Memory Flow

```text
+-------------------------------------------------------------------------+
|                           SoupLite Runtime                              |
|                                                                         |
|  1. Low-RAM Environment Initializer                                      |
|     (expandable_segments:True, MALLOC_TRIM=100000, OMP_NUM_THREADS=1)   |
|                                     |                                   |
|  2. Quantized Model Shard Loader                                        |
|     (TinyLlama-1.1B 4-bit / SmolLM2-135M / Qwen2.5-0.5B)               |
|                                     |                                   |
|  3. Zero-Copy Generator Pipeline                                        |
|     (Row-by-row dataset streaming without in-RAM loading)               |
|                                     |                                   |
|  4. LowRAMContext Manager Lifecycle                                     |
|     (gc.collect() -> PyTorch empty_cache() -> libc.malloc_trim(0))      |
+-------------------------------------------------------------------------+
```

---

## 🚀 Installation & Quickstart

### 1. Installation

```bash
# Clone repository
git clone https://github.com/jaksilikov/souplite.git
cd souplite

# Install core CLI
pip install -e .

# Install full training and UI dependencies
pip install -e ".[train,ui]"
```

### 2. Run Low-RAM Demonstration (< 2 GB RAM)

```bash
souplite quickstart
```

*Automatically generates a 20-instruction dataset, builds a 4-bit NF4 configuration for `TinyLlama-1.1B-Chat-v1.0`, and executes a low-RAM training step within a sub-2GB memory budget.*

### 3. System Health Check

```bash
souplite doctor
```

---

## 🔬 Benchmarks & Memory Diagnostics

SoupLite comes with an integrated memory profiling tool to evaluate system memory efficiency and verify sub-2GB RAM compliance.

### Python API Usage

```python
from souplite import LowRAMContext, purge_memory
from souplite.utils.ram_benchmark import RAMBenchmark

# Run complete memory diagnostic
report = RAMBenchmark.run_full_diagnostics()
print(f"Status: {report['status']}")
print(f"Process RSS: {report['memory']['process_rss_mb']} MB")
print(f"Sub-2GB Compliant: {report['memory']['within_2gb_budget']}")

# Execute heavy operations inside LowRAMContext
with LowRAMContext():
    # Perform LLM fine-tuning or inference here
    pass
# Memory is automatically purged and released to kernel upon exit
```

---

## 🐳 Docker Deployment (Capped at 2 GB RAM)

Run SoupLite in a strict, isolated container capped at 2048 MB RAM:

```bash
docker-compose up --build
```

`docker-compose.yml` configuration:

```yaml
services:
  souplite:
    build: .
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
      - MALLOC_TRIM_THRESHOLD_=100000
      - PYTHONMALLOC=malloc
      - OMP_NUM_THREADS=1
      - TOKENIZERS_PARALLELISM=false
    deploy:
      resources:
        limits:
          memory: 2048M
        reservations:
          memory: 512M
```

---

## 📄 License & Attribution

Distributed under the **[Apache License 2.0](LICENSE)**.

* **Primary Developer**: **Muhtar Jaksilikov** ([jaksilikov](https://github.com/jaksilikov))
* **Inspiration Note**: Inspired by Soup (by Makazhan Alpamys, which required 4GB+ RAM), created by Muhtar Jaksilikov to run on &le; 2GB RAM.

© 2026 **Muhtar Jaksilikov**. All rights reserved.
