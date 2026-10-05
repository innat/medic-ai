"""Dataset analysis result types for the MedicAI nnU-Net workflow."""

from __future__ import annotations

from typing import Any


class AnalysisReport:
    """Structured findings from a read-only dataset analysis.

    Errors and warnings are human-readable and include case context where
    possible. ``fingerprint`` is populated only when all integrity checks pass.
    """

    def __init__(
        self,
        *,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        recommendations: list[str] | None = None,
        fingerprint: Any = None,
        class_prevalence: dict[str, float] | None = None,
        region_prevalence: dict[str, float] | None = None,
    ) -> None:
        self.errors = errors or []
        self.warnings = warnings or []
        self.recommendations = recommendations or []
        self.fingerprint = fingerprint
        self.class_prevalence = class_prevalence or {}
        self.region_prevalence = region_prevalence or {}

    def raise_if_errors(self) -> None:
        """Raise one actionable exception when analysis found invalid data."""
        if self.errors:
            details = "\n".join(f"- {error}" for error in self.errors)
            raise ValueError(f"Dataset analysis found {len(self.errors)} error(s):\n{details}")

    @property
    def is_valid(self) -> bool:
        """Whether the analyzed dataset passed all error-level checks."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the report."""
        return {
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "recommendations": list(self.recommendations),
            "class_prevalence": dict(self.class_prevalence),
            "region_prevalence": dict(self.region_prevalence),
            "fingerprint": (
                self.fingerprint.to_dict() if self.fingerprint is not None else None
            ),
        }
