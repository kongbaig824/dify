"""Capability profile for the unified Dify Agent experience.

This module is intentionally provider-agnostic. Existing Dify runtime layers remain
responsible for actually resolving models, tools, knowledge retrieval, memory, and
network integrations. The profile provides one stable place to describe which
capabilities a run is allowed to use and to derive concise run instructions.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AgentCapability(StrEnum):
    """Capabilities that can participate in a unified agent run."""

    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    WEB_SEARCH = "web_search"
    TOOL_EXECUTION = "tool_execution"


class UnifiedAgentProfile(BaseModel):
    """Configuration for a unified agent without owning any integrations."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="Dify Unified Agent", min_length=1, max_length=120)
    capabilities: frozenset[AgentCapability] = Field(default_factory=frozenset)
    max_tool_steps: int = Field(default=8, ge=1, le=64)

    def instructions(self) -> str:
        """Build deterministic instructions for the enabled capabilities."""
        enabled = ", ".join(sorted(capability.value for capability in self.capabilities))
        capability_text = enabled or "none"

        return (
            f"You are {self.name}.\n"
            "Use only capabilities enabled for this run.\n"
            f"Enabled capabilities: {capability_text}.\n"
            f"Maximum tool steps: {self.max_tool_steps}.\n"
            "Plan before acting: identify which enabled capabilities are needed, "
            "use the smallest useful sequence of tool calls, and synthesize the "
            "results only after the required calls finish.\n"
            "Do not call the same tool repeatedly when the available result already "
            "answers the question.\n"
            "When a capability is unavailable, say so instead of pretending it was used."
        )

        return (
            f"You are {self.name}.\n"
            "Use only capabilities enabled for this run.\n"
            f"Enabled capabilities: {capability_text}.\n"
            f"Maximum tool steps: {self.max_tool_steps}.\n"
            "When a capability is unavailable, say so instead of pretending it was used."
        )


__all__ = ["AgentCapability", "UnifiedAgentProfile"]
