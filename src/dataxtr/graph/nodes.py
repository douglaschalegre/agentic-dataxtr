"""LangGraph node implementations."""

from typing import Any, Literal, Optional

from dataxtr.agents.extraction import ExtractionAgent
from dataxtr.agents.field_prep import FieldPrepAgent
from dataxtr.agents.quality import QualityAgent
from dataxtr.graph.state import ExtractionState
from dataxtr.models.router import ModelRouter
from dataxtr.schemas.fields import ExtractionComplexity
from dataxtr.services.document_parser import load_document


async def document_loader_node(state: ExtractionState) -> dict[str, Any]:
    """Load and parse the document.

    Args:
        state: Current workflow state

    Returns:
        State updates with document content and metadata
    """
    doc_path = state["document_path"]
    doc_type = state["document_type"]

    try:
        content, metadata = await load_document(doc_path, doc_type)

        return {
            "document_content": content,
            "document_metadata": metadata,
            "workflow_status": "in_progress",
        }
    except Exception as e:
        return {
            "errors": [f"Failed to load document: {str(e)}"],
            "workflow_status": "failed",
        }


async def field_prep_node(state: ExtractionState) -> dict[str, Any]:
    """Analyze fields and create logical groupings.

    Args:
        state: Current workflow state

    Returns:
        State updates with field groups and pending queue
    """
    router = ModelRouter()

    # Use a standard model for field preparation
    model_config = router.select_model(ExtractionComplexity.COMPLEX)
    model = router.get_chat_model(model_config)

    agent = FieldPrepAgent(model=model)

    try:
        field_groups = await agent.execute(
            schema_fields=state["schema_fields"],
            document_metadata=state["document_metadata"],
            document_content=state["document_content"],
        )

        pending_groups = [g.group_id for g in field_groups]

        return {
            "field_groups": field_groups,
            "pending_groups": pending_groups,
        }
    except Exception as e:
        return {
            "errors": [f"Field preparation failed: {str(e)}"],
        }


async def extraction_node(
    state: ExtractionState,
    group_id: str,
    model_hint: Optional[str] = None,
) -> dict[str, Any]:
    """Extract fields for a specific group.

    Args:
        state: Current workflow state
        group_id: ID of the group to extract
        model_hint: Optional hint for model selection

    Returns:
        State updates with extraction results
    """
    # Find the field group
    field_group = next(
        (g for g in state["field_groups"] if g.group_id == group_id),
        None,
    )

    if not field_group:
        return {
            "errors": [f"Field group {group_id} not found"],
        }

    # Select appropriate model
    router = ModelRouter()
    model, model_config = router.get_model_for_task(
        field_group.extraction_strategy,
        model_hint=model_hint,
    )

    agent = ExtractionAgent(
        model=model,
        model_name=model_config.model_id,
    )

    try:
        result = await agent.execute(
            field_group=field_group,
            document_content=state["document_content"],
            document_metadata=state["document_metadata"],
        )

        # Update retry count if this is a retry
        existing_result = next(
            (r for r in state["extraction_results"] if r.group_id == group_id),
            None,
        )
        if existing_result:
            result.retry_count = existing_result.retry_count + 1

        return {
            "extraction_results": [result],
        }
    except Exception as e:
        return {
            "errors": [f"Extraction failed for {group_id}: {str(e)}"],
        }


async def quality_node(state: ExtractionState) -> dict[str, Any]:
    """Validate extraction results.

    Args:
        state: Current workflow state

    Returns:
        State updates with quality reports and retry queue
    """
    router = ModelRouter()

    # Use a powerful model for quality assessment
    model_config = router.select_model(ExtractionComplexity.COMPLEX)
    model = router.get_chat_model(model_config)

    agent = QualityAgent(model=model)

    new_reports = []
    retry_queue = []

    # Only validate results not yet validated
    validated_group_ids = {r.group_id for r in state["quality_reports"]}

    for result in state["extraction_results"]:
        if result.group_id in validated_group_ids:
            continue

        # Find corresponding field group
        field_group = next(
            (g for g in state["field_groups"] if g.group_id == result.group_id),
            None,
        )

        if not field_group:
            continue

        try:
            report = await agent.execute(
                extraction_result=result,
                field_group=field_group,
                schema_fields=state["schema_fields"],
            )

            new_reports.append(report)

            # Add to retry queue if needed (and haven't exceeded retry limit)
            if (
                report.recommendation in ["retry_same_model", "retry_different_model"]
                and result.retry_count < 2  # Max 2 retries
            ):
                model_hint = None
                if report.recommendation == "retry_different_model":
                    model_hint = "upgrade"
                retry_queue.append((result.group_id, model_hint))

        except Exception as e:
            new_reports.append(
                {
                    "group_id": result.group_id,
                    "passed": False,
                    "errors": [str(e)],
                }
            )

    return {
        "quality_reports": new_reports,
        "retry_queue": state["retry_queue"] + retry_queue,
        "current_iteration": state["current_iteration"] + 1,
    }


async def aggregator_node(state: ExtractionState) -> dict[str, Any]:
    """Aggregate all results into final output.

    Args:
        state: Current workflow state

    Returns:
        State updates with final results
    """
    final_results = {}

    for result in state["extraction_results"]:
        for field_result in result.results:
            # Get quality info if available
            quality_report = next(
                (r for r in state["quality_reports"] if r.group_id == result.group_id),
                None,
            )

            final_results[field_result.field_name] = {
                "value": field_result.extracted_value,
                "confidence": field_result.confidence,
                "source": field_result.source_location,
                "method": field_result.extraction_method,
                "quality_passed": quality_report.passed if quality_report else None,
            }

    return {
        "final_results": final_results,
        "workflow_status": "completed",
    }


def supervisor_router(
    state: ExtractionState,
) -> Literal["extract", "quality", "aggregate", "end"]:
    """Determine next action based on current state.

    Args:
        state: Current workflow state

    Returns:
        Next node to route to
    """
    # Check for failures
    if state["workflow_status"] == "failed":
        return "end"

    # Check iteration limit
    if state["current_iteration"] >= state["max_iterations"]:
        return "aggregate"

    # Process retry queue first
    if state["retry_queue"]:
        return "extract"

    # Process pending groups
    if state["pending_groups"]:
        return "extract"

    # Check if quality validation needed
    extraction_ids = {r.group_id for r in state["extraction_results"]}
    quality_ids = {r.group_id for r in state["quality_reports"]}
    unvalidated = extraction_ids - quality_ids

    if unvalidated:
        return "quality"

    # All done
    return "aggregate"
