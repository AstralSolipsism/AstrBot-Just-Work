from typing import Generic

from .agent import Agent
from .run_context import TContext
from .tool import FunctionTool


class HandoffTool(FunctionTool, Generic[TContext]):
    """Handoff tool for delegating tasks to another agent."""

    provider_id: str | None
    dispatch_mode: str
    allowed_children: list[str]
    parent_name: str | None

    def __init__(
        self,
        agent: Agent[TContext],
        parameters: dict | None = None,
        tool_description: str | None = None,
        **kwargs,
    ) -> None:
        del kwargs

        # `tool_description` is the public description shown to the caller LLM.
        description = tool_description or self.default_description(agent.name)

        super().__init__(
            name=f"transfer_to_{agent.name}",
            description=description,
            parameters=parameters or self.default_parameters(),
        )

        # Optional provider override for this subagent. When set, the handoff
        # execution will use this chat provider id instead of the global/default.
        self.provider_id = None
        self.dispatch_mode = "free"
        self.allowed_children = []
        self.parent_name = None
        # Note: Must assign after initialization to avoid parent class overrides.
        self.agent = agent

    def default_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The input to be handed off to another agent. This should be a clear and concise request or task.",
                },
                "image_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: An array of image sources (public HTTP URLs or local file paths) used as references in multimodal tasks such as video generation.",
                },
                "background_task": {
                    "type": "boolean",
                    "description": (
                        "Whether the handoff should run in background. "
                        "If the target handoff dispatch_mode is sync, this value is always forced to false. "
                        "If dispatch_mode is async, this value is always forced to true. "
                        "Only when dispatch_mode is free can the model choose true or false."
                    ),
                },
            },
        }

    def default_description(self, agent_name: str | None) -> str:
        agent_name = agent_name or "another"
        return f"Delegate tasks to {agent_name} agent to handle the request."
