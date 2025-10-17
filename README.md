# CanvasLMS-Agent

一个基于 FastAPI + LangChain 的极简 Web 应用。前端输入 Canvas API Token，后端由 LangChain Agent 调用 Canvas 工具，帮你查询学业信息（DDL、课程、公告），并用中文总结回复。

### 功能特性
- **DDL 快速查询**: 罗列未截止作业，按截止时间排序。
- **课程列表**: 获取当前在读课程名称与 ID。
- **公告查询**: 支持全局或按课程名称筛选的最新公告摘要。
- **即开即用前端**: 纯静态页，开箱即用。
- **Docker 部署**: 提供精简镜像与一条命令运行。

### 技术栈
- **后端**: FastAPI, LangChain, langchain-openai
- **前端**: 原生 HTML + Pico.css
- **运行/部署**: Uvicorn, Docker

---

## 环境变量
复制 `ENV.sample` 为 `.env`（或在部署平台配置环境变量）：

```
LLM_BASE_URL=你的 OpenAI 兼容接口地址（需包含 /v1，例如 https://api.openai-compatible.example/v1）
LLM_API_KEY=你的 LLM API Key
LLM_MODEL=gpt-4o-mini
CANVAS_BASE_URL=https://your-school.instructure.com
# 可选：调试开关（true/false/1/0）
# AGENT_VERBOSE=false
```

> 安全提示：Canvas Token 不会被后端持久化；前端仅使用 `sessionStorage` 保存，关闭标签页即清除。

---

## 本地运行（Python 3.11+）
1) 安装依赖：
```bash
pip install -r requirements.txt
```

2) 配置环境变量（PowerShell 示例）：
```powershell
$env:LLM_BASE_URL="https://api.openai-compatible.example/v1"
$env:LLM_API_KEY="sk-xxx"
$env:LLM_MODEL="gpt-4o-mini"
$env:CANVAS_BASE_URL="https://your-school.instructure.com"
# 可选：$env:AGENT_VERBOSE="true"
```

3) 启动服务：
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

4) 打开浏览器访问：
- 前端页面: `http://localhost:8000/`
- 健康检查: `http://localhost:8000/api/health`

> 前端页面提供 Token、LLM Base、LLM Key 输入框；会以可选的 `X-LLM-BASE`、`X-LLM-KEY` 请求头透传给后端（便于本地演示，生产环境请在服务器侧配置）。

---

## Docker 运行
```bash
# 构建镜像
docker build -t canvas-ai-chat .

# 运行容器（传入必要环境变量）
docker run -d --name canvas-ai -p 8000:8000 \
  -e LLM_BASE_URL=https://api.openai-compatible.example/v1 \
  -e LLM_API_KEY=sk-xxx \
  -e LLM_MODEL=gpt-4o-mini \
  -e CANVAS_BASE_URL=https://your-school.instructure.com \
  canvas-ai-chat
```

---

## API 说明

### POST `/api/chat`
- 请求体（JSON）：
```json
{ "message": "最近有什么作业要交？", "canvas_token": "<你的 Canvas Token>" }
```
- 响应体（JSON）：
```json
{ "answer": "..." }
```
- 可选请求头（便于本地演示/调试）：
  - `X-LLM-BASE`: 覆盖 LLM_BASE_URL
  - `X-LLM-KEY`: 覆盖 LLM_API_KEY
  - `X-LLM-MODEL`: 覆盖 LLM_MODEL（默认 `gpt-4o-mini`）
  - `X-AGENT-VERBOSE`: `true/false` 控制详细日志
  - `X-REQUEST-ID`: 自定义请求追踪 ID

### GET `/api/health`
- 返回 `{ "status": "ok" }`

### POST `/api/tool_test`
- 用于直连测试工具（不经过 Agent 推理）。
- 请求体（JSON）：
```json
{ "tool": "list_my_courses | get_upcoming_assignments | get_announcements", "canvas_token": "<token>", "course_name": "<可选，公告时使用>" }
```
- 响应体（JSON）：
```json
{ "result": "..." }
```

---

## 安全与隐私
- **不持久化 Token**：不会在磁盘或日志中存储用户的 Canvas Token。
- **最小化暴露**：Token 仅存在于单次请求的内存中。
- **建议**：生产环境请在网关/反代层面启用 HTTPS，并在服务侧配置 LLM 凭证，不从前端透传。

---

## 目录结构
```
backend/
  app/
    main.py            # FastAPI 入口，API 与静态托管
    agent.py           # LangChain Agent 初始化与调用
    tools/
      canvas_tools.py  # Canvas 工具实现（课程/作业/公告）
frontend/
  index.html          # 极简前端页面
requirements.txt
Dockerfile
```

---

## 开发建议
- 使用 `AGENT_VERBOSE=true` 观察详细的 LLM/工具日志。
- 需要扩展能力时，可在 `backend/app/tools/canvas_tools.py` 新增工具，并在 `agent.py` 注册。

---

## Roadmap
- 对话记忆与 RAG 支持
- SSE 流式响应

---

如果这个项目对你有帮助，欢迎点亮 Star ✨。
