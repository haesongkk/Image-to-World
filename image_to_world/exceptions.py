class PipelineError(Exception):
    """Base exception for pipeline failures."""


class MissingArtifactError(PipelineError):
    """Raised when an expected file or artifact is missing."""


class StageExecutionError(PipelineError):
    """Raised when a stage cannot complete its work."""
