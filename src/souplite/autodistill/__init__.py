"""Model-free foundations for Soup's future offline AutoDistill pipeline."""

from souplite.autodistill.capture import (
    TeacherExpertExample,
    build_teacher_expert_capture_token,
    capture_teacher_expert_trajectory,
)
from souplite.autodistill.contract import (
    AUTODISTILL_PLAN_SCHEMA,
    CAPTURE_TOKEN_SCHEMA,
    CONSUMPTION_EVENT_SCHEMA,
    SHARD_MANIFEST_SCHEMA,
    AutoDistillPlan,
    CaptureToken,
    ShardManifest,
    build_plan_estimate,
)
from souplite.autodistill.fingerprints import (
    verified_dataset_bytes,
    verify_dataset_fingerprint,
    verify_teacher_fingerprint,
    verify_tokenizer_file_fingerprint,
    verify_tokenizer_fingerprint,
)
from souplite.autodistill.publisher import CaptureShardPublisher

__all__ = [
    "AUTODISTILL_PLAN_SCHEMA",
    "CAPTURE_TOKEN_SCHEMA",
    "CONSUMPTION_EVENT_SCHEMA",
    "SHARD_MANIFEST_SCHEMA",
    "AutoDistillPlan",
    "CaptureToken",
    "CaptureShardPublisher",
    "ShardManifest",
    "TeacherExpertExample",
    "build_plan_estimate",
    "build_teacher_expert_capture_token",
    "capture_teacher_expert_trajectory",
    "verify_dataset_fingerprint",
    "verify_teacher_fingerprint",
    "verify_tokenizer_file_fingerprint",
    "verify_tokenizer_fingerprint",
    "verified_dataset_bytes",
]
