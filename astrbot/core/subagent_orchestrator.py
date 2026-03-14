from __future__ import annotations

from typing import Any

from astrbot import logger
from astrbot.core.agent.agent import Agent
from astrbot.core.agent.handoff import HandoffTool
from astrbot.core.persona_mgr import PersonaManager
from astrbot.core.provider.func_tool_manager import FunctionToolManager


class SubAgentOrchestrator:
    """Loads subagent definitions from config and registers handoff tools.

    This is intentionally lightweight: it does not execute agents itself.
    Execution happens via HandoffTool in FunctionToolExecutor.
    """

    DEFAULT_DISPATCH_MODE = "free"
    VALID_DISPATCH_MODES = {"sync", "async", "free"}

    def __init__(
        self, tool_mgr: FunctionToolManager, persona_mgr: PersonaManager
    ) -> None:
        self._tool_mgr = tool_mgr
        self._persona_mgr = persona_mgr
        self.handoffs: list[HandoffTool] = []
        self.raw_specs_by_name: dict[str, dict[str, Any]] = {}
        self.children_map: dict[str | None, list[HandoffTool]] = {}

    def _normalize_agent_name(self, value: Any) -> str:
        return str(value or "").strip()

    def _normalize_dispatch_mode(self, value: Any) -> str:
        mode = str(value or self.DEFAULT_DISPATCH_MODE).strip().lower()
        if mode not in self.VALID_DISPATCH_MODES:
            logger.warning(
                "Invalid subagent dispatch_mode %r, fallback to %s.",
                value,
                self.DEFAULT_DISPATCH_MODE,
            )
            return self.DEFAULT_DISPATCH_MODE
        return mode

    def _normalize_allowed_children(self, value: Any, parent_name: str) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            logger.warning(
                "Subagent %s allowed_children must be a list, got %s.",
                parent_name,
                type(value).__name__,
            )
            return []

        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            child_name = self._normalize_agent_name(item)
            if not child_name or child_name == parent_name or child_name in seen:
                continue
            seen.add(child_name)
            result.append(child_name)
        return result

    def _get_authorized_child_handoffs(
        self, parent_name: str | None
    ) -> list[HandoffTool]:
        return list(self.children_map.get(parent_name, []))

    def get_authorized_child_handoffs(
        self, parent_name: str | None
    ) -> list[HandoffTool]:
        normalized_parent = (
            self._normalize_agent_name(parent_name) if parent_name else None
        )
        return self._get_authorized_child_handoffs(normalized_parent)

    async def reload_from_config(self, cfg: dict[str, Any]) -> None:
        from astrbot.core.astr_agent_context import AstrAgentContext

        agents = cfg.get("agents", [])
        if not isinstance(agents, list):
            logger.warning("subagent_orchestrator.agents must be a list")
            self.handoffs = []
            self.raw_specs_by_name = {}
            self.children_map = {}
            return

        raw_specs_by_name: dict[str, dict[str, Any]] = {}
        handoffs_by_name: dict[str, HandoffTool] = {}
        ordered_names: list[str] = []

        for item in agents:
            if not isinstance(item, dict):
                continue
            if not item.get("enabled", True):
                continue

            name = self._normalize_agent_name(item.get("name"))
            if not name:
                continue
            if name in raw_specs_by_name:
                logger.warning(
                    "Duplicate subagent name %s found, later config wins.", name
                )

            persona_id = item.get("persona_id")
            persona_data = None
            if persona_id:
                try:
                    persona_data = await self._persona_mgr.get_persona(persona_id)
                except StopIteration:
                    logger.warning(
                        "SubAgent persona %s not found, fallback to inline prompt.",
                        persona_id,
                    )

            instructions = str(item.get("system_prompt", "")).strip()
            public_description = str(item.get("public_description", "")).strip()
            provider_id = item.get("provider_id")
            if provider_id is not None:
                provider_id = str(provider_id).strip() or None
            tools = item.get("tools", [])
            begin_dialogs = None

            if persona_data:
                instructions = persona_data.system_prompt or instructions
                begin_dialogs = persona_data.begin_dialogs
                tools = persona_data.tools
                if public_description == "" and persona_data.system_prompt:
                    public_description = persona_data.system_prompt[:120]
            if tools is None:
                normalized_tools = None
            elif not isinstance(tools, list):
                normalized_tools = []
            else:
                normalized_tools = [str(t).strip() for t in tools if str(t).strip()]

            dispatch_mode = self._normalize_dispatch_mode(item.get("dispatch_mode"))
            allowed_children = self._normalize_allowed_children(
                item.get("allowed_children"),
                name,
            )

            agent = Agent[AstrAgentContext](
                name=name,
                instructions=instructions,
                tools=normalized_tools,  # type: ignore[arg-type]
            )
            agent.begin_dialogs = begin_dialogs
            handoff = HandoffTool(
                agent=agent,
                tool_description=public_description or None,
            )
            handoff.provider_id = provider_id
            handoff.dispatch_mode = dispatch_mode
            handoff.allowed_children = allowed_children
            handoff.parent_name = None

            raw_specs_by_name[name] = {
                "name": name,
                "persona_id": persona_id,
                "provider_id": provider_id,
                "tools": normalized_tools,
                "begin_dialogs": begin_dialogs,
                "instructions": instructions,
                "public_description": public_description,
                "dispatch_mode": dispatch_mode,
                "allowed_children": allowed_children,
                "source": item,
            }
            handoffs_by_name[name] = handoff
            if name not in ordered_names:
                ordered_names.append(name)

        children_map: dict[str | None, list[HandoffTool]] = {None: []}
        main_handoffs: list[HandoffTool] = []
        for name in ordered_names:
            handoff = handoffs_by_name[name]
            main_handoffs.append(handoff)
        children_map[None] = main_handoffs

        for parent_name, raw_spec in raw_specs_by_name.items():
            child_handoffs: list[HandoffTool] = []
            for child_name in raw_spec.get("allowed_children", []):
                child_handoff = handoffs_by_name.get(child_name)
                if child_handoff is None:
                    logger.warning(
                        "Subagent %s references missing child %s in allowed_children.",
                        parent_name,
                        child_name,
                    )
                    continue
                child_handoffs.append(child_handoff)
            children_map[parent_name] = child_handoffs

        self.raw_specs_by_name = raw_specs_by_name
        self.children_map = children_map
        self.handoffs = main_handoffs

        for handoff in self.handoffs:
            logger.info(f"Registered subagent handoff tool: {handoff.name}")
