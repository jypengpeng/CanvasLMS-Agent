import os
import sys
import logging
from typing import Optional, Dict, Any, List
import uuid
import time
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
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
    history: Optional[List[Dict[str, str]]] = Field(None, description="多轮对话历史：[{role, content}]，用于生成提示中的 <chat_history>")


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
			history=req.history or [],
		)
		if agent_verbose:
			logger.info("[API] final_answer(forward)=%s req_id=%s", answer, request_id)
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


def _extract_canvas_token_from_header(request: Request) -> str:
	"""从请求头提取 Canvas Token。"""
	token = (request.headers.get("X-Canvas-Token") or "").strip()
	if not token:
		raise HTTPException(status_code=400, detail="缺少 X-Canvas-Token 请求头")
	return token


def _build_course_file_tree(client: CanvasClient, course_id: int) -> Dict[str, Any]:
	"""基于 Canvas Files/Folders API 构建课程文件树。"""
	# 列出所有文件夹与文件（按课程聚合，减少逐文件夹请求次数）
	folders: List[Dict[str, Any]] = list(client.paginate(f"/courses/{course_id}/folders"))
	files: List[Dict[str, Any]] = list(client.paginate(f"/courses/{course_id}/files"))

	folder_by_id: Dict[int, Dict[str, Any]] = {}
	children_map: Dict[Optional[int], List[int]] = {}
	for f in folders:
		fid = f.get("id")
		if not fid:
			continue
		fid = int(fid)
		folder_by_id[fid] = f
		pid_raw = f.get("parent_folder_id")
		pid = int(pid_raw) if pid_raw is not None else None
		children_map.setdefault(pid, []).append(fid)

	files_by_folder: Dict[int, List[Dict[str, Any]]] = {}
	for file_obj in files:
		folder_id = file_obj.get("folder_id")
		if folder_id is None:
			continue
		folder_id = int(folder_id)
		files_by_folder.setdefault(folder_id, []).append(file_obj)

	# 子文件夹按名称排序，文件按名称排序
	for pid, flist in list(children_map.items()):
		flist.sort(key=lambda _fid: (str(folder_by_id.get(_fid, {}).get("name", "")).lower(), _fid))

	def _map_file(file_obj: Dict[str, Any]) -> Dict[str, Any]:
		return {
			"id": file_obj.get("id"),
			"display_name": file_obj.get("display_name") or file_obj.get("filename") or "未命名",
			"size": file_obj.get("size"),
			"content_type": file_obj.get("content-type") or file_obj.get("mime_class"),
			"updated_at": file_obj.get("updated_at") or file_obj.get("modified_at"),
		}

	def _make_node(fid: int) -> Dict[str, Any]:
		f = folder_by_id.get(fid, {})
		node: Dict[str, Any] = {
			"id": fid,
			"name": f.get("name") or "",
			"full_name": f.get("full_name") or "",
			"locked": bool(f.get("locked", False)),
			"hidden": bool(f.get("hidden", False)),
			"folders": [],
			"files": [],
		}
		mapped_files = [_map_file(x) for x in files_by_folder.get(fid, [])]
		mapped_files.sort(key=lambda x: (str(x.get("display_name", "")).lower(), int(x.get("id") or 0)))
		node["files"] = mapped_files
		for child_id in children_map.get(fid, []) or []:
			node["folders"].append(_make_node(child_id))
		return node

	# 选择根：parent_folder_id 为空且 context 为课程的文件夹优先
	root_candidates = children_map.get(None, []) or []
	root_folder_id = None
	if root_candidates:
		cands = []
		for fid in root_candidates:
			f = folder_by_id.get(fid, {})
			if str(f.get("context_type", "")).lower() == "course" and int(f.get("context_id", 0)) == int(course_id):
				cands.append(fid)
		# 更偏好名称以 course files 开头的
		prefer = [fid for fid in cands if str(folder_by_id.get(fid, {}).get("full_name", "")).lower().startswith("course files")]
		root_folder_id = prefer[0] if prefer else (cands[0] if cands else root_candidates[0])

	if root_folder_id is not None:
		return {"course_id": int(course_id), "root": _make_node(root_folder_id)}
	else:
		# 退化：不存在显式根，构造聚合根
		return {
			"course_id": int(course_id),
			"root": {
				"id": None,
				"name": "Course Files",
				"full_name": "course files",
				"locked": False,
				"hidden": False,
				"folders": [_make_node(fid) for fid in root_candidates],
				"files": [],
			},
		}


@app.get("/api/courses")
def list_courses(request: Request):
	"""列出用户当前在读课程，返回 JSON：[{id,name,course_code}]。"""
	canvas_token = _extract_canvas_token_from_header(request)
	canvas_base_url = os.getenv("CANVAS_BASE_URL")
	if not canvas_base_url:
		raise HTTPException(status_code=500, detail="缺少 CANVAS_BASE_URL，例如 https://your-school.instructure.com")
	request_id = request.headers.get("X-REQUEST-ID") or str(uuid.uuid4())
	client = CanvasClient(base_url=canvas_base_url, api_token=canvas_token, request_id=request_id)
	try:
		items: List[Dict[str, Any]] = []
		for c in client.paginate("/courses", params={"enrollment_state": "active"}):
			cid = c.get("id")
			name = c.get("name")
			if not cid or not name:
				continue
			items.append({
				"id": int(cid),
				"name": str(name),
				"course_code": c.get("course_code") or c.get("code") or None,
			})
		# 排序：按课程代码/名称
		items.sort(key=lambda o: (str(o.get("course_code") or ""), str(o.get("name") or "")))
		return JSONResponse(content={"courses": items})
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"课程列表获取失败: {str(e)}")


@app.get("/api/courses/{course_id}/file_tree")
def get_course_file_tree(course_id: int, request: Request):
	canvas_token = _extract_canvas_token_from_header(request)
	canvas_base_url = os.getenv("CANVAS_BASE_URL")
	if not canvas_base_url:
		raise HTTPException(status_code=500, detail="缺少 CANVAS_BASE_URL，例如 https://your-school.instructure.com")
	request_id = request.headers.get("X-REQUEST-ID") or str(uuid.uuid4())
	client = CanvasClient(base_url=canvas_base_url, api_token=canvas_token, request_id=request_id)
	try:
		tree = _build_course_file_tree(client, int(course_id))
		return JSONResponse(content=tree)
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"文件树构建失败: {str(e)}")


@app.get("/api/files/{file_id}/download")
def download_file(file_id: int, request: Request):
	"""代理下载 Canvas 文件，避免在前端暴露 Token。使用 X-Canvas-Token 进行后端鉴权与下游获取。"""
	canvas_token = _extract_canvas_token_from_header(request)
	canvas_base_url = os.getenv("CANVAS_BASE_URL")
	if not canvas_base_url:
		raise HTTPException(status_code=500, detail="缺少 CANVAS_BASE_URL，例如 https://your-school.instructure.com")
	request_id = request.headers.get("X-REQUEST-ID") or str(uuid.uuid4())
	client = CanvasClient(base_url=canvas_base_url, api_token=canvas_token, request_id=request_id)
	try:
		# 元数据请求：增加轻量重试以缓解偶发 DNS/网络波动
		last_err: Optional[Exception] = None
		meta = {}
		for attempt in range(3):
			try:
				meta_resp = client.get(f"/files/{int(file_id)}")
				meta_resp.raise_for_status()
				j = meta_resp.json()
				meta = j if isinstance(j, dict) else {}
				last_err = None
				break
			except requests.exceptions.RequestException as e:
				last_err = e
				time.sleep(0.4 * (2 ** attempt))
			except Exception as e:
				last_err = e
				time.sleep(0.2 * (2 ** attempt))
		if last_err is not None and not meta:
			raise HTTPException(status_code=502, detail=f"下载失败: 元数据请求异常: {str(last_err)}")

		# 优先使用文件对象中的 url；缺失则尝试 public_url 端点
		url = meta.get("url")
		if not url:
			try:
				pub_resp = client.get(f"/files/{int(file_id)}/public_url")
				pub_resp.raise_for_status()
				pub_json = pub_resp.json() if isinstance(pub_resp.json(), dict) else {}
				url = pub_json.get("public_url") or url
			except Exception:
				pass
		if not url:
			raise HTTPException(status_code=404, detail="文件不存在或没有可用的下载链接")
		filename = meta.get("display_name") or meta.get("filename") or f"file_{file_id}"
		# 直接跟随重定向并流式转发
		dl_resp = client.session.get(url, stream=True, timeout=60)
		if getattr(dl_resp, "status_code", 500) >= 400:
			raise HTTPException(status_code=dl_resp.status_code, detail=f"下游文件下载失败（{dl_resp.status_code}）")
		content_type = dl_resp.headers.get("Content-Type") or meta.get("content-type") or "application/octet-stream"
		from urllib.parse import quote as _urlquote
		disposition = f"attachment; filename*=UTF-8''{_urlquote(str(filename))}"

		def _iter_stream():
			try:
				for chunk in dl_resp.iter_content(chunk_size=65536):
					if chunk:
						yield chunk
			except Exception:
				pass
			finally:
				try:
					dl_resp.close()
				except Exception:
					pass

		return StreamingResponse(_iter_stream(), media_type=content_type, headers={"Content-Disposition": disposition})
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")


# 静态文件托管（将前端文件放在 project_root/frontend 下）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
frontend_dir = os.path.join(project_root, "frontend")
if os.path.isdir(frontend_dir):
	app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
