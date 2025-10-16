import os
import logging
from typing import Optional, Any
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

from .tools.canvas_tools import build_canvas_tools, build_canvas_tools_react


logger = logging.getLogger("canvas_agent")


def _truncate(text: Optional[str], max_len: int = 2000) -> str:
    if text is None:
        return ""
    t = str(text)
    return t if len(t) <= max_len else (t[: max_len] + "…[truncated]")


class AgentDebugHandler(BaseCallbackHandler):
    """Logs prompts, tool calls and responses. Does NOT log secrets."""

    def __init__(self, request_id: Optional[str] = None) -> None:
        super().__init__()
        self.request_id = request_id or "-"

    def on_chat_model_start(self, serialized: dict, messages, **kwargs: Any) -> None:  # type: ignore[override]
        try:
            model_name = serialized.get("name") or serialized.get("id") or "chat_model"
            logger.info("[LLM] start model=%s req_id=%s", model_name, self.request_id)
            # messages is List[List[BaseMessage]]
            batches = messages if isinstance(messages, list) else [messages]
            for bi, batch in enumerate(batches):
                try:
                    for mi, m in enumerate(batch):
                        role = getattr(m, "type", m.__class__.__name__)
                        logger.info("[LLM] prompt[%d/%d] %s: %s req_id=%s", bi, mi, role, _truncate(m.content), self.request_id)
                except Exception:
                    pass
        except Exception:
            logger.exception("[LLM] on_chat_model_start logging failed req_id=%s", self.request_id)

    def on_llm_end(self, response, **kwargs: Any) -> None:  # type: ignore[override]
        try:
            generations = getattr(response, "generations", [])
            for gi, gen_list in enumerate(generations):
                if not gen_list:
                    continue
                gen = gen_list[0]
                text = getattr(gen, "text", None)
                if text is None and hasattr(gen, "message"):
                    text = getattr(gen.message, "content", None)
                logger.info("[LLM] response[%d]: %s req_id=%s", gi, _truncate(text), self.request_id)
        except Exception:
            logger.exception("[LLM] on_llm_end logging failed req_id=%s", self.request_id)

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:  # type: ignore[override]
        try:
            name = serialized.get("name", "tool")
            logger.info("[Tool] start name=%s input=%s req_id=%s", name, _truncate(input_str), self.request_id)
        except Exception:
            pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:  # type: ignore[override]
        try:
            logger.info("[Tool] output: %s req_id=%s", _truncate(output), self.request_id)
        except Exception:
            pass

    # Agent actions (for ReAct-style agents)
    def on_agent_action(self, action, **kwargs: Any) -> None:  # type: ignore[override]
        try:
            tool = getattr(action, "tool", "-")
            tool_input = getattr(action, "tool_input", "")
            log = getattr(action, "log", "")
            logger.info("[Agent] action tool=%s input=%s log=%s req_id=%s", tool, _truncate(tool_input), _truncate(log), self.request_id)
        except Exception:
            pass

    def on_agent_finish(self, finish, **kwargs: Any) -> None:  # type: ignore[override]
        try:
            return_values = getattr(finish, "return_values", {})
            log = getattr(finish, "log", "")
            logger.info("[Agent] finish return_keys=%s log=%s req_id=%s", list(return_values.keys()) if isinstance(return_values, dict) else type(return_values), _truncate(log), self.request_id)
        except Exception:
            pass

    # Chain-level tracing
    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs: Any) -> None:  # type: ignore[override]
        try:
            name = serialized.get("name") or serialized.get("id") or "chain"
            logger.info("[Chain] start name=%s inputs_keys=%s req_id=%s", name, list(inputs.keys()) if isinstance(inputs, dict) else type(inputs), self.request_id)
        except Exception:
            pass

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:  # type: ignore[override]
        try:
            out_keys = list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)
            logger.info("[Chain] end outputs_keys=%s req_id=%s", out_keys, self.request_id)
        except Exception:
            pass

    # Error hooks
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:  # type: ignore[override]
        try:
            logger.exception("[LLM] error: %s req_id=%s", str(error), self.request_id)
        except Exception:
            pass

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:  # type: ignore[override]
        try:
            logger.exception("[Tool] error: %s req_id=%s", str(error), self.request_id)
        except Exception:
            pass


def run_agent(
    user_message: str,
    canvas_token: str,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    verbose: bool = False,
    request_id: Optional[str] = None,
) -> str:
	"""创建一次性 Agent 并执行用户问题，严格使用本次请求携带的 Canvas Token。"""
	resolved_llm_api_key = llm_api_key or os.environ.get("LLM_API_KEY")
	resolved_llm_base_url = llm_base_url or os.environ.get("LLM_BASE_URL")
	resolved_llm_model = llm_model or os.environ.get("LLM_MODEL", "gpt-4o-mini")

	if not resolved_llm_api_key or not resolved_llm_base_url:
		raise RuntimeError("LLM 基础配置缺失：请设置 LLM_API_KEY 与 LLM_BASE_URL")

	canvas_base_url = os.environ.get("CANVAS_BASE_URL")
	if not canvas_base_url:
		raise RuntimeError("缺少 CANVAS_BASE_URL，例如 https://your-school.instructure.com")

	llm = ChatOpenAI(
		api_key=resolved_llm_api_key,
		base_url=resolved_llm_base_url,
		model=resolved_llm_model,
		temperature=0.2,
	)

	tools: list[Tool] = build_canvas_tools(canvas_token=canvas_token, canvas_base_url=canvas_base_url, request_id=request_id)
	if verbose:
		try:
			tool_names = ", ".join([t.name for t in tools])
			logger.info("[Agent] tools: %s req_id=%s", tool_names, request_id or "-")
		except Exception:
			pass

	system_message = (
		"你是一个面向 Canvas 学习管理平台的助理。\n"
		"- 当用户询问作业/DDL/截止日期时，优先调用 get_upcoming_assignments\n"
		"- 当用户需要课程列表或你无法确定课程指代时，调用 list_my_courses\n"
		"- 当用户询问公告/通知时，调用 get_announcements（可带或不带 course_name）\n"
		"- 使用工具返回的结构化列表进行总结与排序。\n"
		"- 回答请使用简洁中文，并包含日期/课程/作业名或公告标题等关键信息。\n"
	)

	agent = initialize_agent(
		tools=tools,
		llm=llm,
		agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
		verbose=verbose,
		agent_kwargs={"system_message": system_message},
		handle_parsing_errors=True,
	)
	if verbose:
		logger.info("[Agent] system_message: %s req_id=%s", _truncate(system_message), request_id or "-")
		logger.info("[Agent] user_message: %s req_id=%s", _truncate(user_message), request_id or "-")
		logger.info("[Agent] type=%s req_id=%s", "STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION", request_id or "-")

	callbacks = [AgentDebugHandler(request_id=request_id)] if verbose else None

	# Prefer invoke() to avoid deprecation of run().
	def _extract_text(o: Any) -> str:
		if isinstance(o, dict) and "output" in o:
			return str(o["output"]) if o["output"] is not None else ""
		return str(o) if o is not None else ""

	try:
		output = agent.invoke({"input": user_message}, config={"callbacks": callbacks} if callbacks else None)
		final_text = _extract_text(output)
	except Exception:
		# As a fallback, try deprecated run()
		if verbose:
			logger.exception("[Agent] invoke failed; falling back to run() req_id=%s", request_id or "-")
		final_text = agent.run(user_message, callbacks=callbacks)  # type: ignore[arg-type]

	# If structured chat yields empty output, fallback to OpenAI functions agent
	if not final_text.strip():
		try:
			if verbose:
				logger.info("[Agent] empty output; fallback to OPENAI_FUNCTIONS req_id=%s", request_id or "-")
			func_agent = initialize_agent(
				tools=tools,
				llm=llm,
				agent=AgentType.OPENAI_FUNCTIONS,
				verbose=verbose,
				agent_kwargs={"system_message": system_message},
				handle_parsing_errors=True,
			)
			try:
				func_out = func_agent.invoke({"input": user_message}, config={"callbacks": callbacks} if callbacks else None)
				final_text = _extract_text(func_out)
			except Exception:
				if verbose:
					logger.exception("[Agent] functions invoke failed; trying run() req_id=%s", request_id or "-")
				final_text = func_agent.run(user_message, callbacks=callbacks)  # type: ignore[arg-type]
		except Exception:
			if verbose:
				logger.exception("[Agent] fallback to OpenAI functions failed req_id=%s", request_id or "-")

	if verbose:
		logger.info("[Agent] final_answer: %s req_id=%s", _truncate(final_text), request_id or "-")
	return final_text
