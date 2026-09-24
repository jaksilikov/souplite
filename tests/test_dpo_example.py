"""Tests for the DPO example config and sample data.

These tests lock the example config + data to a working state so users who
follow README steps never hit a broken example. Values are asserted in
logical groups so a deliberate example change surfaces in one focused test
rather than a 22-line wall.
"""

from __future__ import annotations

import json
from pathlib import Path

from souplite.config.loader import load_config
from souplite.config.schema import SoupConfig
from souplite.data.formats import detect_format, format_to_messages
from souplite.data.validator import validate_and_stats

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
DPO_CONFIG_PATH = EXAMPLES_DIR / "configs" / "dpo_example.yaml"
DPO_DATA_PATH = EXAMPLES_DIR / "data" / "dpo_sample.jsonl"


def test_dpo_example_config_loads():
    """dpo_example.yaml should load and validate without errors."""
    cfg = load_config(DPO_CONFIG_PATH)
    assert isinstance(cfg, SoupConfig)


def test_dpo_example_task_and_data():
    """Core task + data wiring: what matters for DPO to run at all."""
    cfg = load_config(DPO_CONFIG_PATH)
    assert cfg.task == "dpo"
    assert cfg.data.format == "dpo"
    assert cfg.data.train == "examples/data/dpo_sample.jsonl"
    assert cfg.base == "meta-llama/Llama-3.1-8B-Instruct"


def test_dpo_example_training_hyperparams():
    """Training hyperparameters match the example's advertised config."""
    cfg = load_config(DPO_CONFIG_PATH)
    assert cfg.training.epochs == 3
    assert cfg.training.lr == 5e-6
    assert cfg.training.dpo_beta == 0.1
    assert cfg.training.quantization == "4bit"
    assert cfg.training.batch_size == 4
    assert cfg.training.gradient_accumulation_steps == 4


def test_dpo_example_lora_config():
    """LoRA config matches the example's advertised rank/alpha."""
    cfg = load_config(DPO_CONFIG_PATH)
    assert cfg.training.lora.r == 16
    assert cfg.training.lora.alpha == 32
    assert cfg.training.lora.target_modules == "auto"


def test_dpo_sample_data_exists():
    """dpo_sample.jsonl should exist and have the issue-#4 minimum of 5+ pairs."""
    assert DPO_DATA_PATH.exists()
    data = _load_jsonl(DPO_DATA_PATH)
    assert len(data) >= 5, "Expected at least 5 preference pairs"


def test_dpo_sample_data_format_detected():
    """Auto-detect should identify dpo_sample.jsonl as DPO format."""
    data = _load_jsonl(DPO_DATA_PATH)
    detected = detect_format(data)
    assert detected == "dpo"


def test_dpo_sample_data_has_required_keys():
    """Every row in dpo_sample.jsonl should have prompt, chosen, rejected."""
    data = _load_jsonl(DPO_DATA_PATH)
    for i, row in enumerate(data):
        assert "prompt" in row, f"Row {i} missing 'prompt'"
        assert "chosen" in row, f"Row {i} missing 'chosen'"
        assert "rejected" in row, f"Row {i} missing 'rejected'"


def test_dpo_sample_data_validates_clean():
    """dpo_sample.jsonl should pass validation with zero issues."""
    data = _load_jsonl(DPO_DATA_PATH)
    stats = validate_and_stats(data, expected_format="dpo")
    assert stats["valid_rows"] == stats["total"]
    assert stats["duplicates"] == 0
    assert stats["issues"] == []


def test_dpo_sample_data_converts():
    """Every row should convert successfully via format_to_messages."""
    data = _load_jsonl(DPO_DATA_PATH)
    for i, row in enumerate(data):
        result = format_to_messages(row, "dpo")
        assert result is not None, f"Row {i} failed conversion"
        assert "prompt" in result
        assert "chosen" in result
        assert "rejected" in result


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
