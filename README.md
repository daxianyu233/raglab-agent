# RAGLab Agent

面向工程实践的 GitHub 技术情报与知识检索 Agent。项目基于 LangGraph 和
FastAPI，覆盖 Tool Calling、动态 Skill、长期记忆、上下文管线、HITL
人工审批、安全工具策略、任务调度、SSE 运行状态以及自动化评测。

> 当前 Agent 使用 LLM 完成意图识别、路由与回答生成，因此运行服务必须
> 提供有效的 `DEEPSEEK_API_KEY`。项目没有用固定答案伪造无模型 Demo。

## 核心能力

- LangGraph Agent Runtime 与 SQLite Checkpoint
- BM25 / Dense / RRF / Cross-Encoder 混合检索
- Tool Calling 与 Dynamic Skill Runtime
- 短期上下文、长期记忆与上下文预算控制
- Tool Policy、Fail-Closed 与 HITL 中断恢复
- Scheduler、Single-Flight 与执行状态持久化
- FastAPI、SSE、多会话记录与 Web UI
- API 自动化测试与 Agent Benchmark

## Docker 启动

要求：Docker Engine，以及可用的 DeepSeek API Key。

PowerShell：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，填写：

```dotenv
DEEPSEEK_API_KEY=your-real-key
GITHUB_TOKEN=your-optional-github-token
```

构建并启动：

```powershell
docker-compose up --build
```

当前机器若安装了 Docker Compose Plugin，也可以使用：

```powershell
docker compose up --build
```

启动后访问：

- Web UI: <http://127.0.0.1:8765>
- OpenAPI: <http://127.0.0.1:8765/docs>
- Health: <http://127.0.0.1:8765/api/v1/health>

停止服务：

```powershell
docker-compose down
```

运行数据保存在 Docker named volumes 中，不会写入镜像或提交到 GitHub。

## Conda 本地启动

```powershell
conda activate raglab
python -m pip install -r requirements.txt
$env:DEEPSEEK_API_KEY="your-real-key"
python -m uvicorn raglab.api.app:app --host 127.0.0.1 --port 8765
```

## 自动化测试

FastAPI 契约测试不调用真实 LLM，而是注入有状态 Fake Runtime：

```powershell
python -m pytest tests/test_api.py -q -p no:cacheprovider
```

当前测试覆盖会话持久化、用户隔离、SSE 事件、HITL 查询与恢复、并发会话锁、
错误映射和删除流程。完整 Agent Benchmark 会调用真实模型，因此会产生 API
费用。

## 环境变量

| 变量 | 必需 | 用途 |
|---|---:|---|
| `DEEPSEEK_API_KEY` | 是 | Agent 意图识别、路由与回答生成 |
| `GITHUB_TOKEN` | 否 | 提高 GitHub REST API 限额 |
| `DEEPSEEK_BASE_URL` | 否 | DeepSeek 兼容接口地址 |
| `DEEPSEEK_MODEL` | 否 | 部分情报处理脚本的模型覆盖项 |

不要把 `.env`、API Key、SQLite 数据库或个人数据提交到仓库。

## 项目结构

```text
raglab/       Agent、Runtime、Memory、Retrieval、Security、Scheduler、API
config/       Agent、检索和 GitHub Intelligence 配置
skills/       可动态发现和加载的 Agent Skills
evaluation/   评测指标与数据结构
tests/        自动化测试
scripts/      CLI、数据采集、索引和 Benchmark 入口
data/corpus/  示例 PDF 知识库
```

## 当前边界

- 本项目不是离线模型应用；没有 DeepSeek Key 时不会启动完整 Agent。
- GitHub 更新属于高风险外部操作，执行前需要 HITL 审批。
- 公开部署时还需要在现有安全控制之外配置鉴权、限流和费用配额。
