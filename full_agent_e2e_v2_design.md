# Full Agent E2E v2 测试集设计

## 目标

v2 面向 `SecureAgentRuntime`，用真实 Agent、当前持久化数据、Observation 和 Metrics 检查行为。它不再用 `RRF_TEST_FACT_731`、`GH_TEST_FACT_442`、`LATEST_REPOS` 等人工 marker，也不要求答案包含固定仓库排名或固定文本。

v1 实际位于 `raglab/evaluation/datasets/full_agent_e2e_v1.json`，因此 v2 放在同一加载目录：`raglab/evaluation/datasets/full_agent_e2e_v2.json`。

## 设计原则

1. **真实数据锺定**：知识问题来自现有 PDF chunks；GitHub 问题来自现有 repositories/snapshots/summaries；历史上下文使用现有 Event Store 行。
2. **无 fixture catalog**：v2 删除 v1 的 `fixture_catalog`。`setup` 只指向已存在的数据，或表示 Benchmark 运行时必须发生的真实状态转换。
3. **无固定答案字符串**：不使用 `answer_must_contain` / `ordered_answer_fragments`验证业务答案。
4. **行为与结构优先**：验证 Tool/capability 选择、结果非空、schema、排序、引用、HITL、effect ledger、事件持久化和隔离性。
5. **可变数据容忍**：GitHub 指标会随新采集变化；SQL case 验证形状和排序，不锁定当前 stars 数或前 5 名文本。
6. **空 Memory 库如实处理**：扫描时 `store` 表为 0 行。v2 测试“使用长期记忆路径后返回空结果且不幻觉”，不伪造偏好。

## 20 个 Case 与 Agent 能力

| Case | 类别 | 主要能力 | 真实数据依赖 |
|---|---|---|---|
| `e2e_v2_001` | direct_no_tool | 直答与 Tool 克制 | 无 |
| `e2e_v2_002` | knowledge_retrieval | PDF BM25、引用 | `DOC-RET-001-P002-C000` |
| `e2e_v2_003` | github_semantic_retrieval | GitHub BM25、metadata/evidence | `vitali87/code-graph-rag` 及持久化 summary chunks |
| `e2e_v2_004` | structured_query | Text-to-SQL、join、排序 | `repositories` + `repository_snapshots` |
| `e2e_v2_005` | previous_answer_reuse | 上一答案复用 | Event Store rows 9–10 |
| `e2e_v2_006` | previous_tool_evidence | 原始历史 Tool Evidence | Event Store rows 1–8 |
| `e2e_v2_007` | recent_context | 近期上下文 | Event Store rows 15–18 |
| `e2e_v2_008` | historical_context | Conversation Retriever | Event Store rows 1–3, 19–22 |
| `e2e_v2_009` | long_term_memory | LTM 查询、空结果安全 | SQLite Store 真实 0 行 |
| `e2e_v2_010` | retrieval_forbidden | 禁止检索约束 | 无 |
| `e2e_v2_011` | tool_minimality | 单一正确 Tool | `DOC-RAG-CHUNK-001-P002-C000` |
| `e2e_v2_012` | dynamic_capability | Skill 动态加载、审批前暂停 | 现有 `github-intelligence-update` Skill |
| `e2e_v2_013` | hitl_reject | HITL 拒绝、无副作用 | 真实 update Tool pending state |
| `e2e_v2_014` | hitl_approve | HITL 批准、幂等/effect ledger | 真实 update Tool；需授权 Benchmark sandbox |
| `e2e_v2_015` | event_persistence | Human/Assistant 事件持久化 | Conversation Event Store |
| `e2e_v2_016` | thread_isolation | Checkpoint/thread 隔离 | `test_user` 的两个现有 threads |
| `e2e_v2_017` | tool_error_recovery | 真实 schema 错误后修正 | GitHub metadata validation + `github/spec-kit` |
| `e2e_v2_018` | max_steps | 步数上限、去重 | PDF chunk `DOC-LG-001-P003-C000` |
| `e2e_v2_019` | archive_reconciliation | checkpoint/event 对账 | `adapter-test` 现有 checkpoint/events |
| `e2e_v2_020` | working_memory_safety | 压缩后恢复原始证据 | 22 条真实历史 events |

## Observation 要求

每个运行至少采集：

- `run_id`, `case_id`, `user_id`, `thread_id`
- 最终任务状态和非空答案标志
- 暴露的 Tool 列表、实际 Tool 调用序列、参数和状态
- capability groups 和 Context Planner 决策
- 检索结果数、返回字段、引用数，不必保存完整大段答案作为 assertion
- Context source：当前轮、recent、historical event、raw tool evidence、long-term memory
- HITL pending/decision、external effect ledger 和真实副作用计数
- Event Store 新增角色、序号、thread 归属
- 步数、Token、延迟、错误与重试

## Metrics

### 通用

- `task_completion_rate`
- `answer_nonempty_rate`
- `tool_selection_accuracy`
- `capability_group_precision`
- `unnecessary_tool_call_rate`
- `tool_error_recovery_rate`
- `max_steps_violation_rate`

### Retrieval / SQL

- `retrieval_nonempty_rate`
- `retrieval_result_schema_valid_rate`
- `citation_presence_rate`
- `sql_read_only_compliance_rate`
- `sql_result_shape_accuracy`
- `sql_ordering_accuracy`

### Context / Memory

- `previous_answer_reuse_success_rate`
- `historical_raw_evidence_recall_rate`
- `cross_thread_leak_rate`（目标 0）
- `unsupported_memory_claim_rate`（目标 0）
- `event_store_source_of_truth_preservation_rate`

### Security / HITL

- `approval_required_before_effect_rate`
- `rejected_effect_count`（目标 0）
- `duplicate_external_effect_rate`（目标 0）
- `approval_state_transition_accuracy`

## 执行前置和限制

- v2 JSON 是数据集定义，本任务没有修改 Adapter/Runner。当前 v1 校验脚本如果写死文件名，需在后续独立任务中让 Runner 显式选择 v2；本次按要求不改代码。
- HITL approve 会启动真实 GitHub 更新流水线，只能在明确授权、隔离且可观测的 Benchmark 环境运行。未授权时该 case 应标记 `blocked_by_environment`，不能假批准。
- 长期记忆正向召回暂无真实记录。当 `store` 未来自然产生真实记忆后，可以从 inventory 重新抽样升级 case；现在不应添加人工偏好。
- `e2e_v2_017` 的首次错误是真实 Tool schema 验证错误，不是 mock tool。
- 历史会话相关 case 引用的 row ID 只是扫描时定位键；Runner 应同时核对 `user_id` + `thread_id` + `sequence_no`，避免将 SQLite 自增 ID 当成语义标识。
