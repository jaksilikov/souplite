<p align="center">
  <img src="soup.png" alt="SoupLite Logo" width="280">
</p>

<h1 align="center">🥣 SoupLite</h1>

<p align="center">
  <strong>Ultra-lightweight LLM Fine-Tuning & Post-Training Framework (< 2 GB RAM)</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Muhtar--Jaksilikov-blue.svg?style=for-the-badge&logo=github" alt="Author">
  <img src="https://img.shields.io/badge/RAM--Limit-%E2%89%A4--2GB-brightgreen.svg?style=for-the-badge" alt="RAM Limit">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg?style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/License-Apache--2.0-orange.svg?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Docker-2GB--Capped-blue.svg?style=for-the-badge&logo=docker" alt="Docker">
</p>

<p align="center">
  <a href="#-авторство-и-концепция">Авторство</a> &bull;
  <a href="#-сравнение-soup-vs-souplite">Сравнение</a> &bull;
  <a href="#-архитектура-оптимизации-памяти">Архитектура</a> &bull;
  <a href="#-быстрый-старт">Быстрый Старт</a> &bull;
  <a href="#-запуск-в-docker-2-гб-озу-лимит">Docker</a>
</p>

---

## 👨‍💻 Авторство и Концепция

> **Автор и Разработчик**: **Muhtar Jaksilikov**  
> **Идея**: Вдохновлено проектом **Soup** (*Makazhan Alpamys*).  
> **Ключевая цель**: В оригинальном проекте Soup для работы требуется от **4 ГБ+ ОЗУ**. Проект **SoupLite** был создан **Muhtar Jaksilikov** для снижения системных требований и запуска полного цикла обучения, дообучения (LoRA/QLoRA), квантования и Web UI на бюджетных серверах, Raspberry Pi и слабых машинах с оперативкой **до 2 ГБ ОЗУ (и менее)**.

---

## 📊 Сравнение: Original Soup vs SoupLite

| Параметр / Фича | Original Soup | 🥣 SoupLite (by Muhtar Jaksilikov) |
| :--- | :---: | :---: |
| **Минимальная RAM** | ~4.0 ГБ+ ОЗУ | **< 2.0 ГБ ОЗУ** |
| **Базовая модель демо** | TinyLlama-1.1B (~2 ГБ ОЗУ) | **Qwen2.5-0.5B / SmolLM2-135M (~270MB–1GB)** |
| **Загрузка весов в CPU RAM** | Полное дублирование | **`low_cpu_mem_usage=True` (Прямая загрузка)** |
| **Сброс системной кучи** | `gc.collect()` | **`malloc_trim(0)` + Си-уровень очистки C-heap** |
| **Загрузка датасета** | Загрузка в ОЗУ | **Потоковый генератор (Zero-Copy)** |
| **Фоновый расход CLI** | ~120 МБ ОЗУ | **< 30 МБ ОЗУ** |
| **Docker-лимит** | Без жесткого капа | **Capped at `2048M` (2GB)** |

---

## ⚡ Архитектура оптимизации памяти

```
+-----------------------------------------------------------------------+
|                         SoupLite Runtime                              |
|                                                                       |
|  1. Environment Hook (expandable_segments, MALLOC_TRIM, OMP_THREADS=1)  |
|                                  |                                    |
|  2. Memory Presets (Qwen2.5-0.5B / SmolLM2-135M + 4-bit NF4 Quant)    |
|                                  |                                    |
|  3. Streaming Model Loader (low_cpu_mem_usage=True)                    |
|                                  |                                    |
|  4. C-Level Heap Purge (libc.so.6 -> malloc_trim(0) + torch empty_cache)|
+-----------------------------------------------------------------------+
```

---

## 🚀 Быстрый старт

### 1. Клонирование и установка

```bash
# Клонируйте репозиторий
git clone https://github.com/jaksilikov/souplite.git
cd souplite

# Установка базового CLI
pip install -e .

# Установка полного стека для обучения и Web UI
pip install -e ".[train,ui]"
```

### 2. Запуск быстрой демонстрации (< 2 ГБ ОЗУ)

```bash
souplite quickstart
```

*Автоматически создаст набор данных из 20 инструкций, сформирует конфигурацию для `Qwen2.5-0.5B-Instruct` и выполнит тестовый шаг обучения.*

### 3. Диагностика и проверка ресурсов

```bash
souplite doctor
```

### 4. Запуск Web UI

```bash
souplite ui --public --port 7860
```

---

## 🐳 Запуск в Docker (2 ГБ ОЗУ Лимит)

```bash
docker-compose up --build
```

Конфигурация `docker-compose.yml`:
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

## 📄 Лицензия

Распространяется по лицензии [Apache License 2.0](LICENSE).  
© 2026 **Muhtar Jaksilikov**.
