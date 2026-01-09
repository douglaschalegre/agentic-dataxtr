"""LangGraph state schema with custom reducers."""

from typing import Annotated, Any, Literal, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from dataxtr.schemas.fields import FieldDefinition, FieldGroup
from dataxtr.schemas.quality import QualityReport
from dataxtr.schemas.results import GroupExtractionResult


def merge_dicts(existing: dict, new: dict) -> dict:
    """Merge dictionaries, with new values overriding existing."""
    result = existing.copy()
    result.update(new)
    return result


def merge_extraction_results(
    existing: list[GroupExtractionResult], new: list[GroupExtractionResult]
) -> list[GroupExtractionResult]:
    """Merge extraction results, updating existing groups or adding new ones."""
    result_map = {r.group_id: r for r in existing}
    for item in new:
        result_map[item.group_id] = item
    return list(result_map.values())


def merge_quality_reports(
    existing: list[QualityReport], new: list[QualityReport]
) -> list[QualityReport]:
    """Merge quality reports, keeping latest per group."""
    report_map = {r.group_id: r for r in existing}
    for item in new:
        report_map[item.group_id] = item
    return list(report_map.values())


def append_errors(existing: list[str], new: list[str]) -> list[str]:
    """Append new errors to existing list."""
    return existing + new


class ExtractionState(TypedDict):
    """Main state for the extraction workflow.

    This state is passed between all nodes in the LangGraph workflow.
    Custom reducers handle proper merging of results from parallel executions.
    """

    # Input fields
    document_path: str
    document_type: Literal["pdf", "image", "docx", "xlsx", "csv"]
    schema_fields: list[FieldDefinition]

    # Document analysis
    document_content: Annotated[dict, merge_dicts]
    document_metadata: Annotated[dict, merge_dicts]

    # Field preparation output
    field_groups: list[FieldGroup]

    # Extraction tracking
    pending_groups: list[str]  # Group IDs yet to process
    in_progress_groups: list[str]  # Currently being processed
    completed_groups: list[str]  # Successfully completed

    # Results with custom reducers for parallel merge
    extraction_results: Annotated[list[GroupExtractionResult], merge_extraction_results]
    quality_reports: Annotated[list[QualityReport], merge_quality_reports]

    # Workflow control
    current_iteration: int
    max_iterations: int
    retry_queue: list[tuple[str, Optional[str]]]  # (group_id, model_hint)

    # Messages for agent communication
    messages: Annotated[list, add_messages]

    # Final output
    final_results: Optional[dict[str, Any]]
    workflow_status: Literal["pending", "in_progress", "completed", "failed"]
    errors: Annotated[list[str], append_errors]


def create_initial_state(
    document_path: str,
    document_type: Literal["pdf", "image", "docx", "xlsx", "csv"],
    schema_fields: list[FieldDefinition],
    max_iterations: int = 3,
) -> ExtractionState:
    """Create initial state for a new extraction workflow."""
    return ExtractionState(
        document_path=document_path,
        document_type=document_type,
        schema_fields=schema_fields,
        document_content={},
        document_metadata={},
        field_groups=[],
        pending_groups=[],
        in_progress_groups=[],
        completed_groups=[],
        extraction_results=[],
        quality_reports=[],
        current_iteration=0,
        max_iterations=max_iterations,
        retry_queue=[],
        messages=[],
        final_results=None,
        workflow_status="pending",
        errors=[],
    )
