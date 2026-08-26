---
id: github-intelligence-update
name: GitHub 技术情报更新
description: 负责重新采集 GitHub 技术情报、检测项目变化、生成分析结果并更新本地 RAG 索引。
tool: update_github_intelligence
version: 1.0.0
---

# GitHub 技术情报更新

## 技能目标

本 Skill 只负责 GitHub 技术情报更新。

它通过项目中已经存在的确定性流水线完成：

1. GitHub Trending 与 Search 候选采集；
2. 仓库详情、README、Release 和 Issue 深度采集；
3. 同日成功摘要复用与跨日变化检测；
4. 对需要更新的项目调用 DeepSeek；
5. 对无有效变化或同日已处理项目复用已有摘要；
6. 生成跨项目热点、完整日报和精简日报；
7. 更新 SQLite、RAG JSONL、BM25 和 Chroma。

本 Skill 不负责普通 GitHub 情报查询。

普通查询已有 GitHub 技术情报时，应继续使用：

`search_github_intelligence`

而不是加载并执行本 Skill。

---

## 何时加载本 Skill

仅当用户明确要求执行 GitHub 信息更新时加载本 Skill。

典型请求包括：

- 更新今天的 GitHub 技术情报；
- 获取最新 GitHub 热点并更新知识库；
- 运行 GitHub 每日信息更新；
- 刷新 GitHub 技术日报；
- 重新采集 GitHub 项目并更新 RAG；
- 同步今天的 GitHub Agent、RAG、MCP 或 LLM 应用动态；
- 重新运行 GitHub 情报流水线；
- 更新本地 GitHub 情报索引。

对于这些请求：

1. 如果本 Skill 尚未加载，应先加载本 Skill；
2. 加载完成后，再使用本 Skill 提供的专属 Tool；
3. 不应绕过 Skill 加载机制直接调用专属 Tool。

---

## 何时不加载本 Skill

以下请求不应加载或执行本 Skill：

- 只查询已有知识库内容；
- 只询问某个 GitHub 项目是什么；
- 查询某个已经采集过的 GitHub 项目；
- 比较已有 GitHub 项目；
- 查询已有 GitHub 每日热点；
- 只总结已经生成的日报；
- 对已有 GitHub 情报进行归纳；
- 普通技术问答；
- 用户没有明确要求重新采集、刷新或更新；
- 非 GitHub 网站的信息获取；
- 修改、润色或总结上一轮回答。

对于已有 GitHub 情报查询，应优先使用：

`search_github_intelligence`

对于 PDF 学习资料查询，应使用：

`search_knowledge_base`

对于其他网站，应使用对应网站自己的 Tool 或 Skill。

不得把本 Skill 泛化为通用网页采集器。

---

## 唯一执行入口

本 Skill 加载后提供以下专属 Tool：

`update_github_intelligence`

执行 GitHub 技术情报更新时，只允许通过该 Tool 启动完整流水线。

不得由 Agent 自行拆分、重排或分别调用以下子脚本：

- `collect_github_intelligence.py`
- `collect_github_repository_details.py`
- `detect_github_repository_updates.py`
- `analyze_github_projects.py`
- `summarize_github_daily_hotspots.py`
- `generate_github_daily_brief.py`
- `build_intelligence_indexes.py`

上述步骤的执行顺序、依赖关系和失败处理统一由：

`scripts/run_daily_intelligence.py`

负责。

因此从 Agent 角度看：

```text
GitHub 技术情报更新
        ↓
update_github_intelligence
        ↓
run_daily_intelligence.py
        ↓
完整流水线