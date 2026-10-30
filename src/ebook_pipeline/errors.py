"""Exception hierarchy shared by all pipeline stages."""

from __future__ import annotations


class PipelineError(RuntimeError):
    """Base class for expected pipeline failures."""


class ConfigError(PipelineError):
    """Configuration is missing, invalid, or unsafe for the selected profile."""


class DecodeError(PipelineError):
    """Input bytes cannot be decoded into a usable text document."""


class GateRejected(PipelineError):
    """A quality gate intentionally rejected a document."""


class StateError(PipelineError):
    """A stage state marker or manifest is inconsistent."""


class PackingError(PipelineError):
    """A packing invariant was violated."""


class ValidationError(PipelineError):
    """A produced artifact failed release validation."""