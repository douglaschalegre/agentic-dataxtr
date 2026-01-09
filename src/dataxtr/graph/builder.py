"""LangGraph workflow builder."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from dataxtr.graph.nodes import (
    aggregator_node,
    document_loader_node,
    extraction_node,
    field_prep_node,
    quality_node,
    supervisor_router,
)
from dataxtr.graph.state import ExtractionState


async def supervisor_node(state: ExtractionState) -> dict[str, Any]:
    """Supervisor node that orchestrates the workflow.

    Determines which groups to process and manages the extraction flow.

    Args:
        state: Current workflow state

    Returns:
        State updates based on routing decision
    """
    # Get next action from router
    next_action = supervisor_router(state)

    if next_action == "extract":
        # Determine what to extract
        if state["retry_queue"]:
            # Process retries
            group_id, model_hint = state["retry_queue"][0]
            remaining_retries = state["retry_queue"][1:]

            # Run extraction
            result = await extraction_node(state, group_id, model_hint)

            return {
                **result,
                "retry_queue": remaining_retries,
                "in_progress_groups": [group_id],
            }

        elif state["pending_groups"]:
            # Process next pending group
            group_id = state["pending_groups"][0]
            remaining_pending = state["pending_groups"][1:]

            # Run extraction
            result = await extraction_node(state, group_id)

            return {
                **result,
                "pending_groups": remaining_pending,
                "in_progress_groups": [group_id],
                "completed_groups": state["completed_groups"] + [group_id],
            }

    elif next_action == "quality":
        # Run quality validation
        return await quality_node(state)

    elif next_action == "aggregate":
        # Run aggregation
        return await aggregator_node(state)

    # End - no updates
    return {}


def should_continue(state: ExtractionState) -> str:
    """Determine if workflow should continue or end.

    Args:
        state: Current workflow state

    Returns:
        Either "continue" or "end"
    """
    next_action = supervisor_router(state)

    if next_action in ["extract", "quality"]:
        return "continue"

    return "end"


def build_extraction_graph() -> StateGraph:
    """Build the complete extraction workflow graph.

    Returns:
        Compiled LangGraph workflow
    """
    # Create the graph
    builder = StateGraph(ExtractionState)

    # Add nodes
    builder.add_node("document_loader", document_loader_node)
    builder.add_node("field_prep", field_prep_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("aggregator", aggregator_node)

    # Define edges
    # Start -> Document Loader
    builder.add_edge(START, "document_loader")

    # Document Loader -> Field Prep
    builder.add_edge("document_loader", "field_prep")

    # Field Prep -> Supervisor
    builder.add_edge("field_prep", "supervisor")

    # Supervisor -> conditional (continue or end)
    builder.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "continue": "supervisor",  # Loop back to supervisor
            "end": "aggregator",
        },
    )

    # Aggregator -> END
    builder.add_edge("aggregator", END)

    return builder.compile()


def build_simple_extraction_graph() -> StateGraph:
    """Build a simplified extraction graph without supervisor loop.

    This version processes all groups sequentially without retry logic.
    Useful for simpler use cases or debugging.

    Returns:
        Compiled LangGraph workflow
    """
    builder = StateGraph(ExtractionState)

    async def extract_all_node(state: ExtractionState) -> dict[str, Any]:
        """Extract all field groups sequentially."""
        all_results = []

        for group in state["field_groups"]:
            result = await extraction_node(state, group.group_id)
            if "extraction_results" in result:
                all_results.extend(result["extraction_results"])

        return {
            "extraction_results": all_results,
            "completed_groups": [g.group_id for g in state["field_groups"]],
        }

    async def validate_all_node(state: ExtractionState) -> dict[str, Any]:
        """Validate all extractions."""
        return await quality_node(state)

    # Add nodes
    builder.add_node("document_loader", document_loader_node)
    builder.add_node("field_prep", field_prep_node)
    builder.add_node("extract_all", extract_all_node)
    builder.add_node("validate_all", validate_all_node)
    builder.add_node("aggregator", aggregator_node)

    # Linear flow
    builder.add_edge(START, "document_loader")
    builder.add_edge("document_loader", "field_prep")
    builder.add_edge("field_prep", "extract_all")
    builder.add_edge("extract_all", "validate_all")
    builder.add_edge("validate_all", "aggregator")
    builder.add_edge("aggregator", END)

    return builder.compile()
