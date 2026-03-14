import traceback

from quart import jsonify, request

from astrbot.core import logger
from astrbot.core.agent.handoff import HandoffTool
from astrbot.core.core_lifecycle import AstrBotCoreLifecycle

from .route import Response, Route, RouteContext


class SubAgentRoute(Route):
    def __init__(
        self,
        context: RouteContext,
        core_lifecycle: AstrBotCoreLifecycle,
    ) -> None:
        super().__init__(context)
        self.core_lifecycle = core_lifecycle
        # NOTE: dict cannot hold duplicate keys; use list form to register multiple
        # methods for the same path.
        self.routes = [
            ("/subagent/config", ("GET", self.get_config)),
            ("/subagent/config", ("POST", self.update_config)),
            ("/subagent/available-tools", ("GET", self.get_available_tools)),
        ]
        self.register_routes()

    def _normalize_agent_config(self, agent: dict) -> dict:
        normalized = dict(agent)
        normalized.setdefault("provider_id", None)
        normalized.setdefault("persona_id", None)
        normalized.setdefault("allowed_children", [])
        normalized.setdefault("dispatch_mode", "free")

        if not isinstance(normalized.get("allowed_children"), list):
            normalized["allowed_children"] = []
        else:
            normalized["allowed_children"] = [
                str(item).strip()
                for item in normalized["allowed_children"]
                if str(item).strip()
            ]

        dispatch_mode = str(normalized.get("dispatch_mode") or "free").strip().lower()
        if dispatch_mode not in {"sync", "async", "free"}:
            dispatch_mode = "free"
        normalized["dispatch_mode"] = dispatch_mode
        return normalized

    def _normalize_subagent_config(self, data) -> dict:
        if not isinstance(data, dict):
            data = {}

        normalized = dict(data)
        if "main_enable" not in normalized and "enable" in normalized:
            normalized["main_enable"] = bool(normalized.get("enable", False))

        normalized.setdefault("main_enable", False)
        normalized.setdefault("remove_main_duplicate_tools", False)
        normalized.setdefault("router_system_prompt", "")

        agents = normalized.get("agents")
        if not isinstance(agents, list):
            agents = []
        normalized["agents"] = [
            self._normalize_agent_config(agent)
            for agent in agents
            if isinstance(agent, dict)
        ]
        return normalized

    async def get_config(self):
        try:
            cfg = self.core_lifecycle.astrbot_config
            data = self._normalize_subagent_config(cfg.get("subagent_orchestrator"))
            return jsonify(Response().ok(data=data).__dict__)
        except Exception as e:
            logger.error(traceback.format_exc())
            return jsonify(Response().error(f"获取 subagent 配置失败: {e!s}").__dict__)

    async def update_config(self):
        try:
            data = await request.json
            if not isinstance(data, dict):
                return jsonify(Response().error("配置必须为 JSON 对象").__dict__)

            normalized = self._normalize_subagent_config(data)
            cfg = self.core_lifecycle.astrbot_config
            cfg["subagent_orchestrator"] = normalized

            # Persist to cmd_config.json
            # AstrBotConfigManager does not expose a `save()` method; persist via AstrBotConfig.
            cfg.save_config()

            # Reload dynamic handoff tools if orchestrator exists
            orch = getattr(self.core_lifecycle, "subagent_orchestrator", None)
            if orch is not None:
                await orch.reload_from_config(normalized)

            return jsonify(Response().ok(message="保存成功").__dict__)
        except Exception as e:
            logger.error(traceback.format_exc())
            return jsonify(Response().error(f"保存 subagent 配置失败: {e!s}").__dict__)

    async def get_available_tools(self):
        """Return all registered tools (name/description/parameters/active/origin).

        UI can use this to build a multi-select list for subagent tool assignment.
        """
        try:
            tool_mgr = self.core_lifecycle.provider_manager.llm_tools
            tools_dict = []
            for tool in tool_mgr.func_list:
                # Prevent recursive routing: subagents should not be able to select
                # the handoff (transfer_to_*) tools as their own mounted tools.
                if isinstance(tool, HandoffTool):
                    continue
                if tool.handler_module_path == "core.subagent_orchestrator":
                    continue
                tools_dict.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "active": tool.active,
                        "handler_module_path": tool.handler_module_path,
                    }
                )
            return jsonify(Response().ok(data=tools_dict).__dict__)
        except Exception as e:
            logger.error(traceback.format_exc())
            return jsonify(Response().error(f"获取可用工具失败: {e!s}").__dict__)
