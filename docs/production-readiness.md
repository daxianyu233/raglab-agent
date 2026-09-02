# 生产化能力与边界

## 先说明项目口径

本项目是个人工程项目，不声称已经承载真实商业流量。这里的“生产化”指：设计时主动
处理状态、并发、副作用、恢复、安全和可观测性，并为这些行为留下代码与测试证据。
真实上线还需要结合组织的账号体系、基础设施、SLA、数据合规与运维制度继续建设。

面试时可以说：

> 我没有把个人项目包装成真实线上系统。我的工作是把常见生产风险变成明确的机制：
> 对高风险 Tool 做策略分类和人工审批，用 Checkpoint 恢复中断状态，用会话锁避免同一
> Thread 并发污染，用 SQLite 保存状态和事件，并用契约测试与 E2E Benchmark 验证。
> 同时我能明确说明单机实现距离多实例上线还缺哪些能力。

## 能力矩阵

| 关注点 | 当前实现与代码证据 | 当前验证 | 真实上线前仍需补充 |
|---|---|---|---|
| 状态恢复 | LangGraph Checkpoint 与 Store；`raglab/persistence/sqlite_backend.py` | HITL approve/reject、会话恢复案例 | PostgreSQL/托管 Checkpointer、备份与迁移 |
| 工具安全 | Tool Policy 分类、未声明工具 Fail-Closed；`raglab/agent/tool_policy.py` | Tool 路由与 HITL Benchmark | 管理后台、策略版本、租户级授权 |
| 人工审批 | Approval Gate 在 Tool 真正执行前中断；`raglab/control/approval_gate_tool_node.py` | approve/reject 案例 | 登录身份、RBAC、审批通知与审计留存 |
| 外部副作用 | External Effect Ledger、重放策略与补偿入口；`raglab/control/` | 错误恢复与安全案例 | 针对每个外部系统定义幂等键、补偿 SLA 和人工处置流程 |
| 并发隔离 | API 层按 `thread_id` 使用进程内锁；`raglab/api/app.py` | 会话级锁、线程隔离测试 | Redis/数据库分布式锁、多实例一致性 |
| 执行取消 | execution cancellation event 与状态记录；`raglab/api/app.py` | API 契约测试 | 可中断的任务队列、超时策略、强制终止隔离进程 |
| 事件与状态 | SSE、Conversation Event Store、Execution 状态查询 | API/SSE 契约测试 | Redis Streams/Kafka、断线续传、事件保留策略 |
| 上下文控制 | Token 预算、近期上下文、摘要和 Tool 证据配对 | Context Benchmark | 按模型动态预算、成本告警、线上分布监控 |
| 可观测性 | 运行时事件、execution 状态、Benchmark 报告；`raglab/observability/` | 本地日志与评测报告 | OpenTelemetry、Prometheus、Trace 后端和告警 |
| 交付 | Docker Compose、健康检查、GitHub Actions | CI 构建与契约测试 | HTTPS、Secret Manager、镜像扫描、灰度与回滚 |

## 面试官问“是否生产落地”时怎么回答

不要只回答“没有”，也不要虚构用户量。按四步回答：

1. **边界**：个人项目，目前是单机与自动化评测环境，没有真实商业流量。
2. **风险**：选择一个具体风险，例如 Tool 重复执行可能造成外部副作用。
3. **机制**：说明 Tool Policy、HITL、Checkpoint、Effect Ledger 如何协作。
4. **缺口**：说明多实例时进程内锁和 SSE 不够，需要分布式锁与事件总线。

这个回答体现的不是“假装有生产经历”，而是你能区分 Demo、生产化设计和真实生产。

## 当前不能声称的内容

- 不能声称服务过真实企业客户或承载过线上 QPS。
- 不能声称达到高可用、灾备或严格 SLA。
- 不能把本地 `user_id` 隔离说成完整鉴权/RBAC。
- 不能把 SQLite 单机持久化说成多实例一致性方案。
- 不能把 20 个 Benchmark 案例说成覆盖了所有真实输入。

