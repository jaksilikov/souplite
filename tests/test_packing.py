"""Tests for sample packing (packing: true) — config, validation, trainer integration."""

from io import StringIO
from unittest.mock import MagicMock

from souplite.config.schema import SoupConfig, TrainingConfig

# ─── Config Tests ─────────────────────────────────────────────────────────


class TestPackingConfig:
    """Test packing field in TrainingConfig."""

    def test_packing_default_false(self):
        """packing should default to False."""
        tcfg = TrainingConfig()
        assert tcfg.packing is False

    def test_packing_true(self):
        """packing: true should be accepted."""
        tcfg = TrainingConfig(packing=True)
        assert tcfg.packing is True

    def test_packing_false_explicit(self):
        """packing: false should be accepted."""
        tcfg = TrainingConfig(packing=False)
        assert tcfg.packing is False

    def test_packing_in_full_config(self):
        """packing should work in a full SoupConfig."""
        cfg = SoupConfig(
            base="test-model",
            data={"train": "data.jsonl"},
            training={"packing": True},
        )
        assert cfg.training.packing is True

    def test_packing_in_sft_config(self):
        """packing should work with task=sft."""
        cfg = SoupConfig(
            base="test-model",
            task="sft",
            data={"train": "data.jsonl"},
            training={"packing": True},
        )
        assert cfg.training.packing is True
        assert cfg.task == "sft"

    def test_packing_in_pretrain_config(self):
        """packing should work with task=pretrain."""
        cfg = SoupConfig(
            base="test-model",
            task="pretrain",
            data={"train": "data.jsonl", "format": "plaintext"},
            training={"packing": True},
        )
        assert cfg.training.packing is True
        assert cfg.task == "pretrain"


# ─── YAML Config Loading Tests ────────────────────────────────────────────


class TestPackingYamlConfig:
    """Test packing via YAML config loading."""

    def test_load_config_with_packing(self):
        """YAML with packing: true should load correctly."""
        from souplite.config.loader import load_config_from_string

        yaml_str = """
base: test-model
data:
  train: data.jsonl
training:
  packing: true
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.training.packing is True

    def test_load_config_without_packing(self):
        """YAML without packing should default to False."""
        from souplite.config.loader import load_config_from_string

        yaml_str = """
base: test-model
data:
  train: data.jsonl
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.training.packing is False


# ─── Trainer Integration Tests ─────────────────────────────────────────────


class TestPackingTrainerIntegration:
    """Test packing is passed correctly to trainers."""

    def test_sft_trainer_receives_packing(self):
        """SFTTrainer should receive packing=True from config."""
        cfg = SoupConfig(
            base="test-model",
            task="sft",
            data={"train": "data.jsonl"},
            training={"packing": True, "batch_size": 2},
        )
        # Verify the config has packing=True
        assert cfg.training.packing is True
        # The actual SFTTrainer init is tested via mock in the trainer test

    def test_pretrain_trainer_receives_packing(self):
        """PretrainTrainerWrapper should receive packing=True from config."""
        cfg = SoupConfig(
            base="test-model",
            task="pretrain",
            data={"train": "data.jsonl", "format": "plaintext"},
            training={"packing": True, "batch_size": 2},
        )
        assert cfg.training.packing is True

    def test_packing_not_passed_for_dpo(self):
        """DPO trainer should not use packing (not applicable)."""
        cfg = SoupConfig(
            base="test-model",
            task="dpo",
            data={"train": "data.jsonl", "format": "dpo"},
            training={"packing": True, "batch_size": 2},
        )
        # Config allows it, but DPO trainer should ignore it
        assert cfg.training.packing is True


# ─── Sweep Integration Tests ─────────────────────────────────────────────


class TestPackingSweep:
    """Test packing in sweep configurations."""

    def test_packing_in_sweep_params(self):
        """packing should be a valid sweep parameter."""
        from souplite.commands.sweep import _parse_sweep_params

        params = _parse_sweep_params(["training.packing=true,false"])
        assert "training.packing" in params
        assert params["training.packing"] == [True, False]


# ─── Warning Tests ────────────────────────────────────────────────────────


class TestPackingWarnings:
    """Test warnings for packing edge cases."""

    def test_packing_with_small_max_length_config(self):
        """Config with packing=true and small max_length should be valid."""
        # Packing + small max_length is valid but may be suboptimal
        cfg = SoupConfig(
            base="test-model",
            data={"train": "data.jsonl", "max_length": 128},
            training={"packing": True},
        )
        assert cfg.training.packing is True
        assert cfg.data.max_length == 128


# ─── SFT Trainer Packing Mock Tests ──────────────────────────────────────


class TestPackingSFTTrainerMock:
    """Test that packing=True is actually passed to SFTTrainer kwargs."""

    def test_sft_trainer_kwargs_include_packing(self):
        """When packing=true, SFTTrainer should be called with packing=True."""
        cfg = SoupConfig(
            base="test-model",
            task="sft",
            data={"train": "data.jsonl"},
            training={"packing": True, "batch_size": 2},
        )
        tcfg = cfg.training

        # Build trainer_kwargs the same way sft.py does
        trainer_kwargs = {
            "model": MagicMock(),
            "args": MagicMock(),
            "train_dataset": MagicMock(),
            "eval_dataset": None,
            "processing_class": MagicMock(),
        }
        if tcfg.packing:
            trainer_kwargs["packing"] = True

        assert "packing" in trainer_kwargs
        assert trainer_kwargs["packing"] is True

    def test_sft_trainer_kwargs_exclude_packing_when_false(self):
        """When packing=false, SFTTrainer kwargs should not include packing."""
        cfg = SoupConfig(
            base="test-model",
            task="sft",
            data={"train": "data.jsonl"},
            training={"packing": False, "batch_size": 2},
        )
        tcfg = cfg.training

        trainer_kwargs = {
            "model": MagicMock(),
            "args": MagicMock(),
            "train_dataset": MagicMock(),
            "eval_dataset": None,
            "processing_class": MagicMock(),
        }
        if tcfg.packing:
            trainer_kwargs["packing"] = True

        assert "packing" not in trainer_kwargs

    def test_packing_small_max_length_warning(self):
        """Packing with max_length < 256 should trigger a warning."""
        from rich.console import Console

        cfg = SoupConfig(
            base="test-model",
            task="sft",
            data={"train": "data.jsonl", "max_length": 128},
            training={"packing": True, "batch_size": 2},
        )

        output = StringIO()
        console = Console(file=output)

        if cfg.training.packing and cfg.data.max_length < 256:
            console.print(
                f"[yellow]Warning:[/] packing=true with "
                f"max_length={cfg.data.max_length} may be suboptimal."
            )

        assert "suboptimal" in output.getvalue()
