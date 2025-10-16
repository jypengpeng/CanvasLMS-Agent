import os
import sys
import logging
from typing import Optional
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .agent import run_agent
from .tools.canvas_tools import (
    CanvasClient,
    list_my_courses_func,
    get_upcoming_assignments_func,
    get_announcements_func,
)

# --- Logging setup for agent debug output ---
logger = logging.getLogger("canvas_agent")
if not logger.handlers:
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # 默认 INFO，真正是否打印详细内容由 verbose 参数控制
    logger.setLevel(logging.INFO)

# 加载环境变量（容器/本地均可使用）
load_dotenv()

app = FastAPI(title="Canvas AI Chat Assistant")

# CORS（如需限制来源，可把 * 改为你的前端域名）
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户问题")
    canvas_token: Optional[str] = Field(None, description="Canvas API Token（必填，前端需提供）")


class ChatResponse(BaseModel):
	answer: str


@app.get("/api/health")
def health_check():
	return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
	if not req.message:
		raise HTTPException(status_code=400, detail="缺少必要字段: message")

	# Canvas Token：必须由前端提供，后端不再使用环境变量兜底
	canvas_token = (req.canvas_token or "").strip()
	if not canvas_token:
		raise HTTPException(status_code=400, detail="缺少 Canvas Token，请先在前端设置 Canvas API Token")

	# 从请求头可选覆盖 LLM 配置（仅用于开发/演示；生产请在服务器配置）
	headers = request.headers
	llm_base_override = headers.get("X-LLM-BASE") or os.getenv("LLM_BASE_URL")
	llm_key_override = headers.get("X-LLM-KEY") or os.getenv("LLM_API_KEY")
	llm_model_override = headers.get("X-LLM-MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
	agent_verbose = (headers.get("X-AGENT-VERBOSE") or os.getenv("AGENT_VERBOSE") or "false").lower() in ("1", "true", "yes")
	request_id = headers.get("X-REQUEST-ID") or str(uuid.uuid4())

	if agent_verbose:
		logger.info("[API] message: %s req_id=%s", req.message, request_id)
		logger.info("[API] llm_base=%s model=%s req_id=%s", llm_base_override, llm_model_override, request_id)

	if not llm_key_override:
		raise HTTPException(status_code=500, detail="后端 LLM_API_KEY 未配置且未通过头部提供")
	if not llm_base_override:
		raise HTTPException(status_code=500, detail="后端 LLM_BASE_URL 未配置且未通过头部提供")

	# 运行 Agent（注意：不要打印/记录用户 Token）
	try:
		answer = run_agent(
			user_message=req.message,
			canvas_token=canvas_token,
			llm_base_url=llm_base_override,
			llm_api_key=llm_key_override,
			llm_model=llm_model_override,
			verbose=agent_verbose,
			request_id=request_id,
		)
		if not (isinstance(answer, str) and answer.strip()):
			logger.info("[API] empty answer -> 502 req_id=%s", request_id)
			raise HTTPException(status_code=502, detail="LLM 返回空响应，请检查 LLM 配置或重试")
		return ChatResponse(answer=answer)
	except HTTPException:
		raise
	except Exception as e:
		# 仅返回安全的错误信息，不泄漏 Token
		raise HTTPException(status_code=500, detail=f"Agent 处理失败: {str(e)}")


class ToolTestRequest(BaseModel):
    tool: str = Field(..., description="工具名称: list_my_courses | get_upcoming_assignments | get_announcements")
    canvas_token: Optional[str] = Field(None, description="Canvas API Token（必填，前端需提供）")
    course_name: Optional[str] = Field(None, description="当 tool=get_announcements 时可选的课程名称")


class ToolTestResponse(BaseModel):
    result: str


@app.post("/api/tool_test", response_model=ToolTestResponse)
def tool_test(req: ToolTestRequest, request: Request):
	# Canvas Token：必须由前端提供，后端不再使用环境变量兜底
	canvas_token = (req.canvas_token or "").strip()
	if not canvas_token:
		raise HTTPException(status_code=400, detail="缺少 Canvas Token，请先在前端设置 Canvas API Token")

	canvas_base_url = os.getenv("CANVAS_BASE_URL")
	if not canvas_base_url:
		raise HTTPException(status_code=500, detail="缺少 CANVAS_BASE_URL，例如 https://your-school.instructure.com")

	request_id = request.headers.get("X-REQUEST-ID") or str(uuid.uuid4())

	client = CanvasClient(base_url=canvas_base_url, api_token=canvas_token, request_id=request_id)

	try:
		tool_name = (req.tool or "").strip()
		if tool_name == "list_my_courses":
			out = list_my_courses_func(client)
		elif tool_name == "get_upcoming_assignments":
			out = get_upcoming_assignments_func(client)
		elif tool_name == "get_announcements":
			name = (req.course_name or "").strip() or None
			out = get_announcements_func(client, name)
		else:
			raise HTTPException(status_code=400, detail=f"未知工具名称: {tool_name}")
		return ToolTestResponse(result=str(out or ""))
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"工具执行失败: {str(e)}")


# 静态文件托管（将前端文件放在 project_root/frontend 下）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
frontend_dir = os.path.join(project_root, "frontend")
if os.path.isdir(frontend_dir):
	app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
