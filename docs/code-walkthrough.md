# 源码回顾路线

目标不是背文件名，而是能完整回答：“一条用户请求进入系统后发生了什么，哪里可能
失败，系统如何恢复？”

## 第一遍：只看主链路

按下面顺序阅读，每次只回答该层的三个问题：输入是什么、输出是什么、失败怎么处理。

1. `raglab/api/app.py`
   - 从 `chat_stream()` 开始，找到 `_chat_event_stream()` 和 `_execute()`。
   - 重点理解：为什么阻塞式 Agent 放到 Worker，SSE 生成器只负责传事件；为什么同一
     `thread_id` 要加锁；取消状态保存在哪里。
2. `raglab/application/agent_factory.py`
   - 从 `build_agent()` 看模型、检索器、Tools、SQLite persistence 和 Context Pipeline
     如何组装。
   - 这一层解决依赖装配，不负责 Agent 的具体推理。
3. `raglab/agent/persistent_langgraph_agent.py`
   - 看 `PersistentRetrievalGraphState`、图的构建、节点与条件边、Checkpoint 配置。
   - 重点理解 StateGraph 每一步修改了哪些状态，以及 `thread_id` 为什么能恢复上下文。
4. `raglab/agent/tool_policy.py`
   - 看 Tool 如何声明只读/写入、副作用、重放策略、是否需要审批。
   - 重点理解 Fail-Closed：没有完成分类的 Tool 为什么不能执行。
5. `raglab/control/approval_gate_tool_node.py`
   - 看 Tool Call 如何在真正执行前形成 approval request，以及批准/拒绝如何恢复。
6. `raglab/persistence/sqlite_backend.py`
   - 区分 Checkpointer（图执行状态）与 Store（跨会话长期数据）。
7. `raglab/observability/runtime_events.py`
   - 看节点内部事件如何绑定到当前请求，再由 API 转成 SSE。

主链路可以压缩成一句话：

```text
HTTP/SSE -> 会话锁与执行记录 -> Agent StateGraph -> Tool Policy/Approval Gate
-> Tool 执行 -> Checkpoint/事件持久化 -> SSE 返回状态与答案
```

## 第二遍：带着故障场景看代码

| 场景 | 先看哪里 | 需要讲清楚的结论 |
|---|---|---|
| 两个请求同时写同一会话 | `raglab/api/app.py` 的 thread lock | 当前仅保证单进程内同一 Thread 串行 |
| 高风险 Tool 被模型直接调用 | `tool_policy.py`、`approval_gate_tool_node.py` | 模型不能绕过代码层策略，未批准不执行 |
| 审批后如何继续 | `runtime_guard.py`、Checkpoint 相关方法 | 使用原 `thread_id` 恢复图状态，不重新猜测整条任务 |
| SSE 页面关闭 | `_chat_event_stream()` 与 execution store | 当前保留后端状态，但没有完整跨进程断线补发 |
| Tool 执行到一半失败 | `external_effect_repository.py`、`compensation.py` | 需要区分可重试、不可重放和需人工修复的副作用 |
| 上下文越来越长 | Agent 的 Context Pipeline | 预算、近期消息、摘要和历史检索共同控制输入 |
| 模型选错 Tool | `evaluation/` 与 Benchmark 报告 | 用固定案例检查路由，而不是只看最终回答是否像真的 |

## 第三遍：准备面试表达

每个模块准备四句话，不背代码：

1. 原始问题是什么。
2. 当前方案怎么工作。
3. 为什么没有选更简单/更复杂的替代方案。
4. 当前方案在哪个规模或故障条件下会失效。

以会话锁为例：

> 同一 Thread 的两个请求并发更新 StateGraph 会造成消息或 Checkpoint 顺序污染，所以
> API 按 thread_id 串行执行。个人项目是单进程部署，因此用了进程内 RLock；它不解决
> 多实例竞争，真正水平扩容时需要数据库锁或 Redis 分布式锁，并配合执行租约和超时。

## 手动核验命令

按你的安排，阅读阶段不自动运行测试。需要核验时手动执行：

```powershell
conda activate pdf-layout-lab
cd D:\AIProjects\rag-lab

# 查看 FastAPI 实际收集的测试用例数
python -m pytest tests/test_api.py --collect-only -q -p no:cacheprovider

# 运行 API 契约测试（不调用真实 LLM）
python -m pytest tests/test_api.py -q -p no:cacheprovider

# 运行完整 Agent Benchmark（调用真实模型，会产生费用）
python -m scripts.run_full_agent_benchmark
```

