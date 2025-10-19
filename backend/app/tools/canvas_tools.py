import re
import time
import logging
import requests
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool, Tool

logger = logging.getLogger("canvas_agent")


class CanvasClient:
	def __init__(self, base_url: str, api_token: str, request_id: Optional[str] = None) -> None:
		# 兼容三种输入：仅域名、到 /api、到 /api/v1
		self.base_url = base_url.rstrip("/")
		self.session = requests.Session()
		self.session.headers.update({
			"Authorization": f"Bearer {api_token}",
		})
		self.request_id = request_id or "-"
		# 归一化 API 根路径
		lowered = self.base_url.lower()
		if lowered.endswith("/api/v1"):
			self.api_root = self.base_url
		elif lowered.endswith("/api"):
			self.api_root = f"{self.base_url}/v1"
		else:
			self.api_root = f"{self.base_url}/api/v1"
		logger.info("[HTTP] api_root=%s req_id=%s", self.api_root, self.request_id)

	def _url(self, path: str) -> str:
		if not path:
			return self.api_root
		return f"{self.api_root}{path if path.startswith('/') else '/' + path}"

	def get(self, path: str, params: Optional[dict] = None) -> requests.Response:
		url = self._url(path)
		start = time.monotonic()
		logger.info("[HTTP] GET %s params=%s req_id=%s", url, params, self.request_id)
		resp = self.session.get(url, params=params, timeout=30)
		elapsed_ms = int((time.monotonic() - start) * 1000)
		logger.info("[HTTP] %s GET %s elapsedMs=%d req_id=%s", getattr(resp, "status_code", "-"), url, elapsed_ms, self.request_id)
		return resp

	def paginate(self, path: str, params: Optional[dict] = None) -> Iterable[dict]:
		params = dict(params or {})
		params.setdefault("per_page", 100)
		url = self._url(path)
		while url:
			start = time.monotonic()
			logger.info("[HTTP] GET %s params=%s req_id=%s", url, params, self.request_id)
			resp = self.session.get(url, params=params, timeout=30)
			elapsed_ms = int((time.monotonic() - start) * 1000)
			logger.info("[HTTP] %s GET %s elapsedMs=%d req_id=%s", getattr(resp, "status_code", "-"), url, elapsed_ms, self.request_id)
			resp.raise_for_status()
			data = resp.json()
			if isinstance(data, list):
				for item in data:
					yield item
			else:
				# 某些 API 可能返回 dict
				yield data

			link = resp.headers.get("Link")
			next_url = None
			if link:
				for part in link.split(","):
					segs = part.split(";")
					if len(segs) >= 2 and 'rel="next"' in segs[1]:
						next_url = segs[0].strip()[1:-1]
						break
			logger.info("[HTTP] pagination next=%s req_id=%s", bool(next_url), self.request_id)
			url = next_url


# --------- 工具实现 ---------

def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
	if not ts:
		return None
	try:
		return datetime.fromisoformat(ts.replace("Z", "+00:00"))
	except Exception:
		return None


def _format_time(dt: Optional[datetime]) -> str:
	if not dt:
		return "无"
	# 统一输出为本地时间的可读格式
	local_dt = dt.astimezone()
	return local_dt.strftime("%Y-%m-%d %H:%M")


def list_my_courses_func(client: CanvasClient) -> str:
    courses = []
    for c in client.paginate("/courses", params={"enrollment_state": "active"}):
        # 跳过没有 id 或 name 的条目
        if not c.get("id") or not c.get("name"):
            continue
        code = c.get("course_code") or c.get("code") or ""
        courses.append((c["name"], c["id"], code))
    if not courses:
        return "未找到任何在读课程。"
    courses.sort(key=lambda x: x[0])
    lines = []
    for name, cid, code in courses:
        if code:
            lines.append(f"课程: {name} | 代码: {code} | id: {cid}")
        else:
            lines.append(f"课程: {name} | id: {cid}")
    return "\n".join(lines)


def get_upcoming_assignments_func(client: CanvasClient) -> str:
	now_utc = datetime.now(timezone.utc)
	items: List[Tuple[datetime, str]] = []
	lines: List[str] = []

	for course in client.paginate("/courses", params={"enrollment_state": "active"}):
		course_id = course.get("id")
		course_name = course.get("name") or "未知课程"
		if not course_id:
			continue
		for a in client.paginate(f"/courses/{course_id}/assignments"):
			due = _parse_iso(a.get("due_at"))
			if not due or due <= now_utc:
				continue
			items.append((due, f"课程: {course_name} | 作业: {a.get('name','未命名')} | 截止: {_format_time(due)}"))

	if not items:
		return "当前没有未截止的作业。"

	items.sort(key=lambda x: x[0])
	for _, line in items:
		lines.append(line)
	return "\n".join(lines)


def _find_course_ids_by_name(client: CanvasClient, course_name: str) -> List[int]:
    results: List[int] = []
    target = course_name.strip().lower()
    for c in client.paginate("/courses", params={"enrollment_state": "active"}):
        name = str(c.get("name", "")).lower()
        if target == name or target in name:
            if c.get("id"):
                results.append(int(c["id"]))
    return results


def _find_course_ids_by_hint(client: CanvasClient, hint_raw: str) -> List[int]:
    """根据用户提供的 hint（课程标题、课程代码如 SDSC5003，或代码后缀如 5003）查找课程ID。
    匹配优先级：course_code 完整匹配 > 代码后缀匹配 > 名称包含。
    """
    if not hint_raw:
        return []
    hint = hint_raw.strip()
    lowered = hint.lower()

    # 采集必要字段
    items: List[Tuple[int, str, str, str]] = []  # (id, name, course_code, short_suffix)
    for c in client.paginate("/courses", params={"enrollment_state": "active"}):
        cid = c.get("id")
        if not cid:
            continue
        name = str(c.get("name") or "")
        code = str(c.get("course_code") or c.get("code") or "")
        suffix = ""
        # 提取 code 中的数字后缀（如 SDSC5003 -> 5003）
        if code:
            m = re.search(r"(\d{3,6})$", code)
            if m:
                suffix = m.group(1)
        items.append((int(cid), name, code, suffix))

    # 1) 完整匹配 course_code（不区分大小写）
    exact: List[int] = [cid for cid, _n, code, _suf in items if code and code.lower() == lowered]
    if exact:
        return exact

    # 2) 若 hint 是纯数字，优先按代码后缀匹配（避免将 5003 当作课程ID）
    if hint.isdigit():
        tail: List[int] = [cid for cid, _n, _c, suf in items if suf and suf == hint]
        if tail:
            return tail

    # 3) 名称包含匹配
    name_hits: List[int] = [cid for cid, name, _c, _s in items if lowered in name.lower()]
    if name_hits:
        return name_hits

    # 4) 回退：如果是数字且前述都没有，允许将其当作课程ID尝试（但最后才用）
    try:
        num = int(hint)
        return [num]
    except Exception:
        return []


def _strip_html(html: str, max_len: int = 240) -> str:
	# 简单去 HTML 标签
	text = re.sub(r"<[^>]+>", " ", html or "")
	text = re.sub(r"\s+", " ", text).strip()
	if len(text) > max_len:
		text = text[:max_len] + "…"
	return text


def get_announcements_func(client: CanvasClient, course_name: Optional[str]) -> str:
	context_codes: List[str] = []
	if course_name:
		# 兼容名称 / 课程代码 / 数字后缀
		ids = _find_course_ids_by_hint(client, course_name)
		if not ids:
			return f"未找到匹配课程: {course_name}"
		context_codes = [f"course_{cid}" for cid in ids]
	else:
		ids = [c.get("id") for c in client.paginate("/courses", params={"enrollment_state": "active"}) if c.get("id")]
		context_codes = [f"course_{cid}" for cid in ids]

	params = {"per_page": 5}
	# announcements 是聚合接口
	# https://canvas.instructure.com/doc/api/announcements.html
	for code in context_codes:
		params.setdefault("context_codes[]", [])
		params["context_codes[]"].append(code)

	resp = client.get("/announcements", params=params)
	resp.raise_for_status()
	data = resp.json() if isinstance(resp.json(), list) else []

	if not data:
		return "未找到相关公告。"

	# 排序：按创建时间倒序
	ann_list: List[Tuple[datetime, str]] = []
	for a in data:
		created = _parse_iso(a.get("created_at")) or _parse_iso(a.get("posted_at"))
		title = a.get("title") or "无标题公告"
		message = a.get("message") or ""
		course_id = a.get("context_code", "").replace("course_", "")
		course_name_resolved = str(a.get("course_id") or course_id)
		line = f"公告: {title} | 课程: {course_name_resolved} | 日期: {_format_time(created)} | 摘要: {_strip_html(message)}"
		ann_list.append((created or datetime.fromtimestamp(0, tz=timezone.utc), line))

	ann_list.sort(key=lambda x: x[0], reverse=True)
	return "\n".join([line for _, line in ann_list])



def build_canvas_tools(canvas_token: str, canvas_base_url: str, request_id: Optional[str] = None) -> List[StructuredTool]:
	client = CanvasClient(base_url=canvas_base_url, api_token=canvas_token, request_id=request_id)

	def list_courses_wrapper() -> str:
		start = time.monotonic()
		logger.info("[ToolFn] list_my_courses start req_id=%s", client.request_id)
		try:
			result = list_my_courses_func(client)
			return result
		except Exception as e:
			logger.exception("[ToolFn] list_my_courses error=%s req_id=%s", str(e), client.request_id)
			raise
		finally:
			elapsed_ms = int((time.monotonic() - start) * 1000)
			logger.info("[ToolFn] list_my_courses end elapsedMs=%d req_id=%s", elapsed_ms, client.request_id)

	def upcoming_assignments_wrapper() -> str:
		start = time.monotonic()
		logger.info("[ToolFn] get_upcoming_assignments start req_id=%s", client.request_id)
		try:
			result = get_upcoming_assignments_func(client)
			return result
		except Exception as e:
			logger.exception("[ToolFn] get_upcoming_assignments error=%s req_id=%s", str(e), client.request_id)
			raise
		finally:
			elapsed_ms = int((time.monotonic() - start) * 1000)
			logger.info("[ToolFn] get_upcoming_assignments end elapsedMs=%d req_id=%s", elapsed_ms, client.request_id)

	def get_announcements_wrapper(course_name: Optional[str] = None) -> str:
		start = time.monotonic()
		name = course_name.strip() if course_name else None
		logger.info("[ToolFn] get_announcements start args.course_name=%s req_id=%s", name, client.request_id)
		try:
			result = get_announcements_func(client, name)
			return result
		except Exception as e:
			logger.exception("[ToolFn] get_announcements error=%s req_id=%s", str(e), client.request_id)
			raise
		finally:
			elapsed_ms = int((time.monotonic() - start) * 1000)
			logger.info("[ToolFn] get_announcements end elapsedMs=%d req_id=%s", elapsed_ms, client.request_id)

	class _EmptyInput(BaseModel):
		"""空输入模型，用于无参数工具。"""
		pass

	class _AnnouncementsInput(BaseModel):
		course_name: Optional[str] = Field(None, description="课程名称/课程代码/数字后缀，留空查询全部")

	class _BrowseFilesInput(BaseModel):
		course_id: Optional[int] = Field(None, description="课程ID（仅在你明确知道是ID时使用）")
		course_name: Optional[str] = Field(None, description="课程名称，可用英文/中文部分匹配")
		course_hint: Optional[str] = Field(None, description="更通用的课程线索：课程代码（如 SDSC5003）或数字后缀（如 5003）")

	def browse_course_files_wrapper(course_id: Optional[int] = None, course_name: Optional[str] = None, course_hint: Optional[str] = None) -> str:
		start = time.monotonic()
		resolved_id: Optional[int] = None
		try:
			logger.info("[ToolFn] browse_course_files.start args=%s req_id=%s", {"course_id": course_id, "course_name": course_name, "course_hint": course_hint}, client.request_id)
			if course_id:
				resolved_id = int(course_id)
			else:
				# 优先使用通用 hint；其次 course_name
				hint = (course_hint or course_name or "").strip()
				if not hint:
					return "请提供 course_id 或 course_name/course_hint"
				ids = _find_course_ids_by_hint(client, hint)
				if not ids:
					return f"未找到匹配课程: {hint}"
				if len(ids) > 1:
					# 返回候选列表，请用户指定
					cands = ", ".join([str(i) for i in ids[:10]])
					return f"找到多门匹配课程，请指定课程ID后再试。候选ID: {cands}…"
				resolved_id = int(ids[0])
			# 返回前端 UI 指令：仅当解析出唯一课程ID时（卡片形式，前端解析后插入到对话）
			directive = f"__UI_FILE_BROWSER_CARD__{{\"courseId\": {resolved_id}}}__"
			logger.info("[ToolFn] browse_course_files.directive=%s req_id=%s", directive, client.request_id)
			return directive
		except Exception as e:
			logger.exception("[ToolFn] browse_course_files error=%s req_id=%s", str(e), client.request_id)
			raise
		finally:
			elapsed_ms = int((time.monotonic() - start) * 1000)
			logger.info("[ToolFn] browse_course_files end elapsedMs=%d req_id=%s", elapsed_ms, client.request_id)

	tools = [
		StructuredTool.from_function(
			name="list_my_courses",
			description=(
				"获取用户所有在读课程的完整名称和课程ID。当用户询问课程列表或课程指代不明确时使用。"
			),
			func=list_courses_wrapper,
			args_schema=_EmptyInput,
		),
		StructuredTool.from_function(
			name="get_upcoming_assignments",
			description=(
				"查询所有活跃课程中尚未截止的作业与DDL。用户提到作业、DDL、截止日期时使用。"
			),
			func=upcoming_assignments_wrapper,
			args_schema=_EmptyInput,
		),
		StructuredTool.from_function(
			name="get_announcements",
			description=(
				"查询课程公告。输入可为课程名称/课程代码/数字后缀，或留空以查询全部。"
			),
			func=get_announcements_wrapper,
			args_schema=_AnnouncementsInput,
		),
		StructuredTool.from_function(
			name="browse_course_files",
			description=(
				"当用户想要以文件树方式浏览课程文件并进行下载时调用。输入可为 course_id，或课程名称/课程代码/数字后缀（如 5003）。若匹配多门课会返回候选并要求指定。重要：工具将返回一个 UI 指令字符串（以 __UI_FILE_BROWSER_CARD__ 开头），你必须在最终回答中原样包含该字符串，且不要改动其任何字符。"
			),
			func=browse_course_files_wrapper,
			args_schema=_BrowseFilesInput,
		),
	]

	return tools


def build_canvas_tools_react(canvas_token: str, canvas_base_url: str, request_id: Optional[str] = None) -> List[Tool]:
	"""为 ReAct/ChatAgent 构造单参数工具（字符串输入）。"""
	client = CanvasClient(base_url=canvas_base_url, api_token=canvas_token, request_id=request_id)

	def list_courses_react(_: str = "") -> str:
		start = time.monotonic()
		logger.info("[ToolFn] list_my_courses(start,react) req_id=%s", client.request_id)
		try:
			return list_my_courses_func(client)
		finally:
			elapsed_ms = int((time.monotonic() - start) * 1000)
			logger.info("[ToolFn] list_my_courses(end,react) elapsedMs=%d req_id=%s", elapsed_ms, client.request_id)

	def upcoming_assignments_react(_: str = "") -> str:
		start = time.monotonic()
		logger.info("[ToolFn] get_upcoming_assignments(start,react) req_id=%s", client.request_id)
		try:
			return get_upcoming_assignments_func(client)
		finally:
			elapsed_ms = int((time.monotonic() - start) * 1000)
			logger.info("[ToolFn] get_upcoming_assignments(end,react) elapsedMs=%d req_id=%s", elapsed_ms, client.request_id)

	def get_announcements_react(course_name: str = "") -> str:
		start = time.monotonic()
		name = course_name.strip() if course_name else None
		logger.info("[ToolFn] get_announcements(start,react) args.course_name=%s req_id=%s", name, client.request_id)
		try:
			return get_announcements_func(client, name)
		finally:
			elapsed_ms = int((time.monotonic() - start) * 1000)
			logger.info("[ToolFn] get_announcements(end,react) elapsedMs=%d req_id=%s", elapsed_ms, client.request_id)

	return [
		Tool(
			name="list_my_courses",
			description=(
				"获取用户所有在读课程的完整名称和课程ID。当用户询问课程列表或课程指代不明确时使用。"
			),
			func=list_courses_react,
		),
		Tool(
			name="get_upcoming_assignments",
			description=(
				"查询所有活跃课程中尚未截止的作业与DDL。用户提到作业、DDL、截止日期时使用。"
			),
			func=upcoming_assignments_react,
		),
		Tool(
			name="get_announcements",
			description=(
				"查询课程公告。输入为课程名称字符串或留空以查询全部。"
			),
			func=get_announcements_react,
		),
	]
