"""soup diagnose — post-training model report card (v0.56.0).

Seven independent failure-mode probes scored against a base reference:

- forgetting        — catastrophic forgetting on held-out tasks
- refusal           — refusal-rate regression on advbench / xstest
- format            — JSON / regex / tool-call validity drift
- mode_collapse     — self-BLEU + diversity at T=0 and T=1
- memorization      — training-prefix echo on partial-prompt probes
- contamination     — overlap of training data with public benchmarks
- citation          — golden ``[doc-N]`` citation faithfulness on RAFT rows

Each probe returns a ``FailureScore`` with ``score in [0, 1]`` (lower is
worse) and a verdict OK / MINOR / MAJOR — same taxonomy as v0.26.0
Quant-Lobotomy.

Every probe accepts a caller-supplied generator callable so the CLI stays
unit-testable without a GPU; the live model-loading runner is in
``utils/diagnose/live.py`` (shipped in v0.71.7). The seven ``score_*`` probe
functions are re-exported here so callers need not reach into each submodule.
"""

from __future__ import annotations

from souplite.utils.diagnose.citation import score_citation
from souplite.utils.diagnose.contamination import score_contamination
from souplite.utils.diagnose.forgetting import score_forgetting
from souplite.utils.diagnose.format import score_format
from souplite.utils.diagnose.memorization import score_memorization
from souplite.utils.diagnose.mode_collapse import score_mode_collapse
from souplite.utils.diagnose.refusal import score_refusal
from souplite.utils.diagnose.report import (
    FAILURE_MODES,
    VERDICTS,
    FailureReport,
    FailureScore,
    classify_score,
    compose_report,
    overall_verdict,
)

__all__ = [
    "FAILURE_MODES",
    "VERDICTS",
    "FailureReport",
    "FailureScore",
    "classify_score",
    "compose_report",
    "overall_verdict",
    "score_citation",
    "score_contamination",
    "score_forgetting",
    "score_format",
    "score_memorization",
    "score_mode_collapse",
    "score_refusal",
]
