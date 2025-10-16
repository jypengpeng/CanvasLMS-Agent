# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app

# 安装依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY backend ./backend
COPY frontend ./frontend

# 环境变量
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# 运行 FastAPI（uvicorn）
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
