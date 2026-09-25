# Changelog

All notable changes to **SoupLite** (Ultra-Lightweight LLM Fine-Tuning & Post-Training Engine for <= 2 GB RAM) will be documented in this file.

## [0.75.0] - 2026-09-25

### Added & Optimized (Low-RAM Architecture)
- **LowRAMMemoryManager (`souplite.utils.memory_opt`)**: Introduced automated C-heap garbage collection (`malloc_trim(0)`) and PyTorch memory allocator segment management (`expandable_segments:True`, `max_split_size_mb:128`).
- **LowRAMContext Manager**: Added pythonic context manager for zero-leak LLM fine-tuning loops and post-training evaluation under 2 GB RAM budget.
- **Multimodal Telegram Bot Core (`souplite-tgbot`)**: Full deployment of private Telegram Bot integration (`@encona_kz_bot`) with text generation, Flux.1 image generation, AI video animation, and TTS audio synthesis.
- **Cloud Continuous Integration**: Added GitHub Actions 24/7 server workflow with concurrency cancellation groups to eliminate process conflicts (`409 Conflict`).
- **Optimized Model Execution**: Configured `torch_dtype=float32`, `low_cpu_mem_usage=True`, and `repetition_penalty=1.2` for SmolLM2 and Qwen2.5 lightweight instruct models.

### Maintenance & Re-branding
- Fully branded as **SoupLite** under lead developer **Muhtar Jaksilikov**.
- Acknowledged inspiration from Soup (Makazhan Alpamys) while reducing memory requirements from 4GB+ down to <= 2GB.
