from dify_agent.runtime.unified_agent_profile import AgentCapability, UnifiedAgentProfile


def test_default_profile_has_safe_empty_capabilities() -> None:
    profile = UnifiedAgentProfile()

    assert profile.name == "Dify Unified Agent"
    assert profile.capabilities == frozenset()
    assert "Enabled capabilities: none." in profile.instructions()


def test_profile_instructions_list_enabled_capabilities_deterministically() -> None:
    profile = UnifiedAgentProfile(
        name="My Agent",
        capabilities=frozenset(
            {
                AgentCapability.TOOL_EXECUTION,
                AgentCapability.MEMORY,
                AgentCapability.WEB_SEARCH,
            }
        ),
        max_tool_steps=12,
    )

    instructions = profile.instructions()

    assert instructions.startswith("You are My Agent.")
    assert "Enabled capabilities: memory, tool_execution, web_search." in instructions
    assert "Maximum tool steps: 12." in instructions


def test_profile_rejects_unknown_configuration() -> None:
    try:
        UnifiedAgentProfile(unknown=True)
    except ValueError:
        return

    raise AssertionError("unknown configuration should be rejected")
