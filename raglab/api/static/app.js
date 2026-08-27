const API = "/api/v1";
const ACTIVE_THREAD_KEY = "raglab.web.active-thread.v1";
const USER_KEY = "raglab.web.user.v1";
const INTERNAL_EVENT_DISPLAY_MS = 500;

const elements = {
  sidebar: document.querySelector("#sidebar"),
  menuButton: document.querySelector("#menu-button"),
  newChat: document.querySelector("#new-chat"),
  threadList: document.querySelector("#thread-list"),
  threadLabel: document.querySelector("#thread-label"),
  userId: document.querySelector("#user-id"),
  userList: document.querySelector("#user-list"),
  messages: document.querySelector("#messages"),
  welcome: document.querySelector("#welcome"),
  form: document.querySelector("#chat-form"),
  question: document.querySelector("#question"),
  sendButton: document.querySelector("#send-button"),
  cancelButton: document.querySelector("#cancel-button"),
  approvalCard: document.querySelector("#approval-card"),
  approvalMessage: document.querySelector("#approval-message"),
  approvalDetail: document.querySelector("#approval-detail"),
  approveButton: document.querySelector("#approve-button"),
  rejectButton: document.querySelector("#reject-button"),
  statusDot: document.querySelector("#status-dot"),
  statusText: document.querySelector("#status-text"),
  runtimeButton: document.querySelector("#runtime-button"),
  runtimeDialog: document.querySelector("#runtime-dialog"),
  runtimeContent: document.querySelector("#runtime-content"),
  closeRuntime: document.querySelector("#close-runtime"),
  messageTemplate: document.querySelector("#message-template"),
};

let sessions = [];
// 当前会话只属于当前浏览器标签页：刷新仍能恢复同一会话；关闭页面后
// 再次打开或新开标签页时不沿用上一次选择，而是自动创建新会话。
localStorage.removeItem(ACTIVE_THREAD_KEY);
let activeThreadId = sessionStorage.getItem(ACTIVE_THREAD_KEY);
let busy = false;
const executionPollers = new Map();

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function saveActiveThread() {
  if (activeThreadId) sessionStorage.setItem(ACTIVE_THREAD_KEY, activeThreadId);
  else sessionStorage.removeItem(ACTIVE_THREAD_KEY);
}

function currentSession() {
  return sessions.find((session) => session.threadId === activeThreadId) || null;
}

function isExecutionActive(execution) {
  return ["RUNNING", "CANCELLING"].includes(execution?.status);
}

function updateInteractionState() {
  const session = currentSession();
  const pending = Boolean(session?.pending);
  const recoveredRunning = isExecutionActive(session?.execution);
  const inputLocked = busy || pending || recoveredRunning;

  // 同一个 thread 停在 LangGraph interrupt 时，只允许用户处理审批。
  // 如果继续发送普通消息，会与 Checkpoint 中尚未恢复的任务发生冲突。
  elements.sendButton.disabled = inputLocked;
  // 发送与停止占用编辑框右侧的同一个操作位，并且始终互斥显示。
  // RUNNING/CANCELLING 时发送按钮消失；终态落库后再切回发送按钮。
  elements.sendButton.classList.toggle("hidden", recoveredRunning);
  elements.cancelButton.classList.toggle("hidden", !recoveredRunning);
  // 第一次点击把状态改为 CANCELLING，随后立即禁用，避免重复取消。
  elements.cancelButton.disabled = session?.execution?.status === "CANCELLING";
  // disabled 是浏览器原生硬限制；readOnly 作为第二层限制，并让页面在
  // 某些浏览器恢复表单状态时也不能继续修改输入内容。
  elements.question.disabled = inputLocked;
  elements.question.readOnly = inputLocked;
  elements.form.classList.toggle("locked", inputLocked);
  elements.form.setAttribute("aria-busy", String(busy));
  elements.approveButton.disabled = busy || !pending;
  elements.rejectButton.disabled = busy || !pending;
  elements.question.placeholder = busy
    ? "Agent 正在执行，可点击停止按钮取消"
    : recoveredRunning
      ? "页面已恢复后台执行状态，请等待任务完成"
    : pending
      ? "当前会话正在等待工具审批，请先批准或拒绝"
      : "给 RAGLab Agent 发送消息";
}

function setBusy(value) {
  busy = value;
  updateInteractionState();
}

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let body;
  try {
    body = await response.json();
  } catch (_) {
    body = { detail: `HTTP ${response.status}` };
  }
  if (!response.ok) {
    const detail = body?.detail;
    const message = typeof detail === "string"
      ? detail
      : detail?.message || JSON.stringify(detail || body);
    throw new Error(message || `请求失败：HTTP ${response.status}`);
  }
  return body;
}

async function createSession() {
  setBusy(true);
  try {
    const userId = elements.userId.value.trim() || "local-user";
    const body = await request(`/threads?user_id=${encodeURIComponent(userId)}`, { method: "POST" });
    const session = {
      threadId: body.thread_id,
      title: "新会话",
      messages: [],
      pending: null,
      execution: null,
      createdAt: Date.now(),
    };
    sessions.unshift(session);
    activeThreadId = session.threadId;
    saveActiveThread();
    renderAll();
    await loadUsers();
    elements.question.focus();
  } catch (error) {
    addTransientError(error.message);
  } finally {
    setBusy(false);
  }
}

async function loadThreads() {
  for (const threadId of executionPollers.keys()) stopExecutionPolling(threadId);
  const userId = elements.userId.value.trim() || "local-user";
  const body = await request(`/threads?user_id=${encodeURIComponent(userId)}`, { headers: {} });
  sessions = body.threads.map((thread) => ({
    threadId: thread.thread_id,
    title: thread.title || "新会话",
    messages: [],
    pending: null,
    execution: null,
    createdAt: thread.created_at,
    updatedAt: thread.updated_at,
    messageCount: thread.message_count,
  }));
}

async function loadUsers() {
  const body = await request("/users", { headers: {} });
  elements.userList.replaceChildren();
  for (const userId of body.users) {
    const option = document.createElement("option");
    option.value = userId;
    elements.userList.append(option);
  }
}

async function loadMessages(session) {
  const userId = elements.userId.value.trim() || "local-user";
  const body = await request(
    `/threads/${encodeURIComponent(session.threadId)}/messages?user_id=${encodeURIComponent(userId)}`,
    { headers: {} },
  );
  session.messages = body.messages.map((message) => ({
    role: message.role,
    content: message.content,
    createdAt: message.created_at,
  }));
}

async function selectSession(threadId) {
  activeThreadId = threadId;
  saveActiveThread();
  const session = currentSession();
  renderAll();
  if (session) {
    try {
      // 恢复会话时同时读取消息历史和 LangGraph 最新中断状态。
      // 页面刷新最终也会进入 selectSession()，因此刷新与点击切换会话
      // 具有完全相同的 HITL 恢复行为。
      await Promise.all([
        loadMessages(session),
        refreshPending(session),
        refreshExecution(session),
      ]);
      // 只有当前仍选中这个 thread 时才整体重绘，避免快速切换时旧请求
      // 返回较晚，把另一个会话的页面状态覆盖掉。
      if (activeThreadId === session.threadId) renderAll();
    } catch (error) {
      addTransientError(error.message);
    }
  }
}

function renderThreadList() {
  elements.threadList.replaceChildren();
  for (const session of sessions) {
    const row = document.createElement("div");
    row.className = `thread-row${session.threadId === activeThreadId ? " active" : ""}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "thread-item";
    button.textContent = session.title || "新会话";
    button.title = session.threadId;
    button.addEventListener("click", () => {
      selectSession(session.threadId);
      elements.sidebar.classList.remove("open");
    });
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "thread-delete";
    deleteButton.textContent = "×";
    deleteButton.title = "删除会话";
    deleteButton.setAttribute("aria-label", `删除会话：${session.title || "新会话"}`);
    deleteButton.addEventListener("click", () => deleteSession(session));
    row.append(button, deleteButton);
    elements.threadList.append(row);
  }
}

async function deleteSession(session) {
  if (busy) return;
  const confirmed = window.confirm(`确定删除会话“${session.title || "新会话"}”吗？\n该会话的消息、工具记录和中断状态都会永久删除。`);
  if (!confirmed) return;
  setBusy(true);
  try {
    const userId = elements.userId.value.trim() || "local-user";
    await request(
      `/threads/${encodeURIComponent(session.threadId)}?user_id=${encodeURIComponent(userId)}`,
      { method: "DELETE" },
    );
    stopExecutionPolling(session.threadId);
    sessions = sessions.filter((item) => item.threadId !== session.threadId);
    if (activeThreadId === session.threadId) {
      activeThreadId = null;
      saveActiveThread();
      await createSession();
    } else {
      renderThreadList();
    }
    await loadUsers();
  } catch (error) {
    addTransientError(error.message);
  } finally {
    setBusy(false);
  }
}

function renderMessages() {
  const session = currentSession();
  elements.messages.replaceChildren();
  const displayMessages = [...(session?.messages || [])];
  // 正常发起请求时，sendQuestion() 已经创建了唯一的实时 progress 气泡。
  // 只有页面刷新导致该临时气泡丢失时，才根据持久化 execution 补一个，
  // 避免同一次执行在页面上同时出现两个“Agent 执行中”。
  const hasLiveProgress = displayMessages.some((message) => message.role === "status");
  if (isExecutionActive(session?.execution) && !hasLiveProgress) {
    const currentStep = executionStepLabel(session.execution.current_step);
    displayMessages.push({
      role: "status",
      content: `Agent 仍在后台运行：${currentStep}（${session.execution.execution_id}）`,
    });
  }
  const hasMessages = Boolean(displayMessages.length);

  if (!hasMessages) {
    const welcome = document.querySelector("#welcome")?.cloneNode(true) || buildWelcome();
    welcome.id = "welcome";
    elements.messages.append(welcome);
    bindSuggestions(welcome);
    return;
  }

  for (const message of displayMessages) {
    const fragment = elements.messageTemplate.content.cloneNode(true);
    const article = fragment.querySelector(".message");
    article.classList.add(message.role);
    fragment.querySelector(".avatar").textContent = message.role === "user" ? "你" : message.role === "error" ? "!" : message.role === "status" ? "…" : "R";
    fragment.querySelector(".message-role").textContent = message.role === "user" ? "你" : message.role === "error" ? "请求错误" : message.role === "status" ? "Agent 执行中" : "RAGLab Agent";
    fragment.querySelector(".message-content").textContent = message.content;
    if (message.trace) {
      const details = fragment.querySelector(".message-trace");
      details.classList.remove("hidden");
      details.querySelector("pre").textContent = JSON.stringify(message.trace, null, 2);
    }
    elements.messages.append(fragment);
  }
  requestAnimationFrame(() => { elements.messages.scrollTop = elements.messages.scrollHeight; });
}

function executionStepLabel(step) {
  const labels = {
    runtime_started: "正在启动 Agent Runtime…",
    runtime_resumed: "正在恢复中断后的执行…",
    runtime_waiting: "正在等待当前会话执行锁…",
    runtime_acquired: "已获得执行锁，正在运行 Agent…",
    context_pipeline_started: "正在规划并组装上下文…",
    context_pipeline_completed: "上下文准备完成…",
    model_started: "正在请求大模型…",
    model_completed: "模型响应完成，正在处理下一步…",
    model_failed: "模型调用失败，正在执行恢复流程…",
    graph_started: "正在执行 LangGraph 状态图…",
    graph_completed: "LangGraph 执行完成，正在整理结果…",
    tools_started: "正在调用工具…",
    tools_completed: "工具调用完成，正在整理结果…",
    tools_failed: "工具调用失败，正在执行恢复流程…",
    hitl_requested: "高风险操作正在等待人工审批…",
    runtime_blocked: "安全策略已阻止当前操作…",
    cancellation_requested: "已请求停止，正在等待当前步骤结束…",
    cancelled: "本次执行已取消。",
    context_pipeline_failed: "上下文处理失败，正在执行恢复流程…",
    hitl_resume_started: "正在恢复人工审批后的任务…",
    hitl_resume_completed: "人工审批后的任务已恢复…",
  };
  return labels[step] || "正在执行当前任务…";
}

function buildWelcome() {
  const wrapper = document.createElement("div");
  wrapper.className = "welcome";
  wrapper.innerHTML = `
    <div class="welcome-logo">R</div>
    <h2>今天想研究什么？</h2>
    <p>可以查询 PDF 知识库、GitHub 技术情报，或测试 Agent 的 Tool Calling 与 HITL 流程。</p>
    <div class="suggestions">
      <button type="button" data-prompt="请介绍这个知识库中关于 Agent 的主要内容。">查询知识库</button>
      <button type="button" data-prompt="请总结本地 GitHub Intelligence 中值得关注的项目。">查询技术情报</button>
      <button type="button" data-prompt="请列出你当前可以使用的 Skills。">查看 Skills</button>
    </div>`;
  return wrapper;
}

function bindSuggestions(root) {
  root.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.question.value = button.dataset.prompt;
      resizeTextarea();
      elements.question.focus();
    });
  });
}

function renderApproval() {
  const pending = currentSession()?.pending;
  elements.approvalCard.classList.toggle("hidden", !pending);
  if (pending) {
    const interrupts = pending.interrupts || [];
    const first = interrupts[0] || {};
    elements.approvalMessage.textContent = first.message || `工具 ${first.tool_name || "未知"} 等待审批。`;
    elements.approvalDetail.textContent = JSON.stringify(pending, null, 2);
  }
  updateInteractionState();
}

function renderAll() {
  renderThreadList();
  renderMessages();
  renderApproval();
  elements.threadLabel.textContent = activeThreadId || "尚未创建会话";
}

function addMessage(role, content, trace = null) {
  const session = currentSession();
  if (!session) return;
  session.messages.push({ role, content, trace, createdAt: Date.now() });
  if (role === "user" && session.title === "新会话") {
    session.title = content.replace(/\s+/g, " ").slice(0, 26) || "新会话";
  }
  renderAll();
  return session.messages[session.messages.length - 1];
}

function addTransientError(message) {
  if (currentSession()) addMessage("error", message);
  else alert(message);
}

function responseTrace(body, streamingEvents = []) {
  return {
    request_id: body.request_id,
    execution_status: body.execution_status,
    stats: body.stats,
    tool_trace: body.tool_trace,
    loaded_skills: body.runtime_trace?.loaded_skills || [],
    streaming_events: streamingEvents,
  };
}

async function sendQuestion(
  question,
  {
    addUserMessage = true,
    allowPending = false,
    initialStatus = null,
    executionId = null,
  } = {},
) {
  const session = currentSession();
  if (!session || busy || (session.pending && !allowPending)) return;
  if (addUserMessage) addMessage("user", question);
  const progress = addMessage(
    "status",
    initialStatus || "请求已发送，正在连接 Agent…",
  );
  const streamingEvents = [];
  setBusy(true);
  try {
    // 整个对话只发送这一次 HTTP 请求。后面的 reader.read() 都是在读取
    // 这一次请求的响应流，不是在反复调用 /chat/stream。
    const response = await fetch(`${API}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify({
        question,
        user_id: elements.userId.value.trim() || "local-user",
        thread_id: session.threadId,
        execution_id: executionId || session.pending?.execution_id || null,
        include_tool_trace: true,
      }),
    });
    if (!response.ok || !response.body) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        detail = typeof body.detail === "string" ? body.detail : body.detail?.message || detail;
      } catch (_) {}
      throw new Error(detail);
    }

    // response.body 是尚未全部到达的 HTTP 响应数据流。
    // reader 用来一段一段读取；TextDecoder 把网络字节转换成字符串。
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    // TCP/HTTP 不保证一次 read() 就是一条完整 SSE 消息。例如一条
    // "event: heartbeat" 可能被拆成两次到达，所以需要 buffer 累积。
    let buffer = "";
    let finalBody = null;
    let streamPending = null;
    let expectedExecutionId = null;
    while (true) {
      // 如果后端暂时没有 yield 新数据，这个 await 会等待，但不会重新
      // 发起 HTTP 请求。done=true 表示后端生成器结束、连接已关闭。
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

      // SSE 使用空行（\n\n）分隔事件。最后一段可能仍不完整，先放回
      // buffer，等下一批网络字节到达后再继续拼接。
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";
      for (const frame of frames) {
        let eventName = "message";
        const dataLines = [];
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) eventName = line.slice(6).trim();
          if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
        }
        if (!dataLines.length) continue;

        // 后端 data 行保存的是 JSON 字符串，这里恢复成 JavaScript 对象。
        const data = JSON.parse(dataLines.join("\n"));

        // accepted 确定本次连接的 execution_id。之后凡是带运行身份的事件，
        // 必须同时属于当前 thread 和当前 execution，否则直接忽略。
        if (eventName === "accepted") {
          expectedExecutionId = data.execution_id || null;
          session.execution = expectedExecutionId
            ? { execution_id: expectedExecutionId, status: "RUNNING" }
            : null;
          updateInteractionState();
        }
        if (data.thread_id && data.thread_id !== session.threadId) continue;
        if (data.execution_id && expectedExecutionId && data.execution_id !== expectedExecutionId) continue;

        // 以下事件来自 SecureAgentRuntime / SecureToolNode 的真实执行点。
        // 后端只公开节点、工具名称和状态，不发送 Prompt 或隐式推理。
        if ([
          "runtime_started", "graph_started", "tools_started",
          "tools_completed", "tools_failed", "hitl_requested",
          "runtime_blocked", "graph_completed",
          "runtime_waiting", "runtime_acquired",
          "context_pipeline_started", "context_pipeline_completed",
          "context_pipeline_failed", "model_started", "model_completed",
          "model_failed", "hitl_resume_started", "hitl_resume_completed",
        ].includes(eventName)) {
          streamingEvents.push({ event: eventName, ...data });
          progress.content = data.message || `Agent 内部事件：${eventName}`;
          renderMessages();
          // LangGraph 可能在极短时间内连续产生多个事件。如果立即处理下
          // 一条，页面文字会一闪而过。这里让每个内部步骤至少停留 0.5 秒。
          await wait(INTERNAL_EVENT_DISPLAY_MS);
          continue;
        }

        // accepted/status/heartbeat 都是 FastAPI 的外围执行状态；它们只
        // 说明请求已接收或后台 Worker 仍未结束，不代表 Agent 内部节点。
        if (["accepted", "status", "heartbeat"].includes(eventName)) {
          progress.content = eventName === "heartbeat"
            ? `${data.message}（${data.elapsed_seconds} 秒）`
            : data.message;
          renderMessages();
        } else if (eventName === "tool_trace") {
          // 当前 tool_trace 在 Agent 完成后获得，内容真实但不是工具开始时
          // 的实时通知。去重后用于显示本轮确实执行过哪些工具。
          const toolNames = (data.tools || []).map((tool) => tool.name || tool.tool_name).filter(Boolean);
          progress.content = toolNames.length
            ? `后端已执行：${[...new Set(toolNames)].join("、")}，正在整理结果…`
            : "工具执行完成，正在整理结果…";
          renderMessages();
        } else if (eventName === "pending_approval") {
          // 这里只暂存中断。SSE 尚未完全收尾时 busy 仍为 true，若提前
          // 显示审批卡，用户可能点击到仍被禁用的按钮而看不到任何反应。
          streamPending = data;
          progress.content = "检测到高风险工具调用，正在完成中断状态保存…";
          renderMessages();
        } else if (eventName === "result") {
          // result 是最终完整 ChatResponse，先保存，等流正常结束后统一
          // 将“执行中”气泡替换成最终回答。
          finalBody = data;
        } else if (eventName === "error") {
          throw new Error(data.message || "Agent 执行失败。");
        }
      }
      if (done) break;
    }
    if (!finalBody) throw new Error("流式连接结束，但没有收到最终结果。");
    session.pending = finalBody.pending_approval || streamPending || null;
    if (session.pending) {
      session.pending.execution_id = finalBody.execution_id || expectedExecutionId;
      session.execution = {
        execution_id: finalBody.execution_id || expectedExecutionId,
        status: "WAITING_HITL",
      };
    } else {
      session.execution = null;
    }
    progress.role = "assistant";
    progress.content = finalBody.answer || "Agent 没有返回文本答案。";
    progress.trace = responseTrace(finalBody, streamingEvents);
    renderAll();
    if (!session.pending) await refreshPending();
  } catch (error) {
    session.execution = null;
    progress.role = "error";
    progress.content = error.message;
    renderAll();
    await refreshPending();
  } finally {
    setBusy(false);
    if (!session.pending) elements.question.focus();
  }
}

async function refreshPending(targetSession = currentSession()) {
  const session = targetSession;
  if (!session) return;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);
  try {
    const body = await request(
      `/hitl/pending?thread_id=${encodeURIComponent(session.threadId)}`,
      { headers: {}, signal: controller.signal },
    );
    session.pending = body.pending_approval || null;
    if (session.pending) session.pending.execution_id = body.execution_id || null;
    if (activeThreadId === session.threadId) renderApproval();
  } catch (error) {
    console.warn("pending query failed or timed out", error);
  } finally {
    clearTimeout(timeoutId);
  }
}

function stopExecutionPolling(threadId) {
  const timer = executionPollers.get(threadId);
  if (timer) clearTimeout(timer);
  executionPollers.delete(threadId);
}

async function pollExecution(session) {
  const executionId = session.execution?.execution_id;
  if (!executionId || !isExecutionActive(session.execution)) {
    stopExecutionPolling(session.threadId);
    return;
  }
  const userId = elements.userId.value.trim() || "local-user";
  try {
    // 刷新后的页面不再拥有原 SSE 连接，因此按 sequence_no 从数据库
    // 续读新事件。即使一次轮询取回多条，也按原顺序逐条展示。
    const afterSequence = session.execution.last_sequence || 0;
    const eventBody = await request(
      `/executions/${encodeURIComponent(executionId)}/events?user_id=${encodeURIComponent(userId)}&after_sequence=${afterSequence}`,
      { headers: {} },
    );
    for (const event of eventBody.events || []) {
      session.execution.current_step = event.event_type;
      session.execution.last_sequence = event.sequence_no;
      if (activeThreadId === session.threadId) renderMessages();
      await wait(INTERNAL_EVENT_DISPLAY_MS);
    }

    const execution = await request(
      `/executions/${encodeURIComponent(executionId)}?user_id=${encodeURIComponent(userId)}`,
      { headers: {} },
    );
    session.execution = {
      ...execution,
      last_sequence: session.execution?.last_sequence || afterSequence,
    };
    if (execution.status === "WAITING_HITL") {
      stopExecutionPolling(session.threadId);
      await refreshPending(session);
    } else if (["SUCCEEDED", "FAILED", "CANCELLED"].includes(execution.status)) {
      stopExecutionPolling(session.threadId);
      session.execution = null;
      await Promise.all([loadMessages(session), refreshPending(session)]);
    }
    if (activeThreadId === session.threadId) renderAll();
  } catch (error) {
    console.warn("execution status query failed", error);
  }
  if (isExecutionActive(session.execution)) {
    const timer = setTimeout(() => pollExecution(session), 1500);
    executionPollers.set(session.threadId, timer);
  }
}

async function refreshExecution(targetSession = currentSession()) {
  const session = targetSession;
  if (!session) return;
  const userId = elements.userId.value.trim() || "local-user";
  const body = await request(
    `/threads/${encodeURIComponent(session.threadId)}/executions/active?user_id=${encodeURIComponent(userId)}`,
    { headers: {} },
  );
  session.execution = body.execution || null;
  stopExecutionPolling(session.threadId);
  if (isExecutionActive(session.execution)) {
    // 首次恢复时读取已有事件，只用最后一条还原当前画面；旧事件不逐条
    // 重播。之后 pollExecution 从这个序号继续，实时展示新增步骤。
    const executionId = session.execution.execution_id;
    const eventBody = await request(
      `/executions/${encodeURIComponent(executionId)}/events?user_id=${encodeURIComponent(userId)}&after_sequence=0`,
      { headers: {} },
    );
    const existingEvents = eventBody.events || [];
    const latestEvent = existingEvents.at(-1);
    if (latestEvent) {
      session.execution.current_step = latestEvent.event_type;
      session.execution.last_sequence = latestEvent.sequence_no;
    } else {
      session.execution.last_sequence = 0;
    }
    const timer = setTimeout(() => pollExecution(session), 1500);
    executionPollers.set(session.threadId, timer);
  }
  if (activeThreadId === session.threadId) renderAll();
}

async function cancelCurrentExecution() {
  const session = currentSession();
  const executionId = session?.execution?.execution_id;
  if (!session || !executionId || session.execution.status !== "RUNNING") return;
  const userId = elements.userId.value.trim() || "local-user";
  const previousExecution = { ...session.execution };
  // 在网络请求发出前先本地切换状态，立即禁用停止按钮，避免用户在
  // cancel 接口返回前连续点击产生重复请求。
  session.execution.status = "CANCELLING";
  session.execution.current_step = "cancellation_requested";
  renderAll();
  try {
    const execution = await request(
      `/executions/${encodeURIComponent(executionId)}/cancel?user_id=${encodeURIComponent(userId)}`,
      { method: "POST" },
    );
    session.execution = {
      ...execution,
      last_sequence: session.execution.last_sequence || 0,
    };
    const progress = [...session.messages].reverse().find((message) => message.role === "status");
    if (progress) progress.content = "已请求停止，正在等待当前模型或工具步骤结束…";
    renderAll();
  } catch (error) {
    session.execution = previousExecution;
    renderAll();
    addTransientError(error.message);
  }
}

async function decide(endpoint) {
  const session = currentSession();
  if (!session || busy || !session.pending) return;
  const actionText = endpoint === "approve" ? "批准" : "拒绝";
  const pendingExecutionId = session.pending.execution_id || null;
  // 审批决定一经提交，旧审批卡就不再是可操作状态，应立即从页面移除。
  // 如果后端恢复失败，sendQuestion 的错误分支会调用 refreshPending()，
  // 再根据 LangGraph checkpoint 恢复仍然有效的中断。
  session.pending = null;
  renderApproval();
  // 审批也复用 /chat/stream。它不是一条用户对话，所以不把 /approve
  // 或 /reject 显示成用户消息，但恢复后的节点事件会沿同一 SSE 返回。
  await sendQuestion(`/${endpoint}`, {
    addUserMessage: false,
    allowPending: true,
    initialStatus: `正在提交${actionText}决定，并从 LangGraph Checkpoint 恢复执行…`,
    executionId: pendingExecutionId,
  });
}

async function checkHealth() {
  try {
    await request("/health", { headers: {} });
    elements.statusDot.className = "status-dot online";
    elements.statusText.textContent = "服务已连接";
  } catch (_) {
    elements.statusDot.className = "status-dot offline";
    elements.statusText.textContent = "服务不可用";
  }
}

async function showRuntime() {
  elements.runtimeContent.textContent = "正在读取…";
  elements.runtimeDialog.showModal();
  try {
    const body = await request("/runtime", { headers: {} });
    elements.runtimeContent.textContent = JSON.stringify(body, null, 2);
  } catch (error) {
    elements.runtimeContent.textContent = error.message;
  }
}

function resizeTextarea() {
  elements.question.style.height = "auto";
  elements.question.style.height = `${Math.min(elements.question.scrollHeight, 180)}px`;
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = elements.question.value.trim();
  if (!question || busy || currentSession()?.pending) return;
  elements.question.value = "";
  resizeTextarea();
  sendQuestion(question);
});

elements.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});
elements.question.addEventListener("input", resizeTextarea);
elements.newChat.addEventListener("click", createSession);
elements.cancelButton.addEventListener("click", cancelCurrentExecution);
elements.approveButton.addEventListener("click", () => decide("approve"));
elements.rejectButton.addEventListener("click", () => decide("reject"));
elements.menuButton.addEventListener("click", () => elements.sidebar.classList.toggle("open"));
elements.runtimeButton.addEventListener("click", showRuntime);
elements.closeRuntime.addEventListener("click", () => elements.runtimeDialog.close());
elements.userId.value = localStorage.getItem(USER_KEY) || "local-user";
elements.userId.addEventListener("change", async () => {
  localStorage.setItem(USER_KEY, elements.userId.value.trim() || "local-user");
  activeThreadId = null;
  await initializeConversations();
});

async function initializeConversations() {
  try {
    await loadThreads();
    // 没有当前标签页保存的 thread，说明这是首次打开/新标签页，直接
    // 创建新会话。只有刷新当前标签页时 sessionStorage 才会保留 ID。
    if (
      !activeThreadId
      || !sessions.some((session) => session.threadId === activeThreadId)
    ) {
      await createSession();
      return;
    }
    saveActiveThread();
    renderAll();
    await selectSession(activeThreadId);
  } catch (error) {
    sessions = [];
    activeThreadId = null;
    renderAll();
    addTransientError(error.message);
  }
}

async function init() {
  await checkHealth();
  await loadUsers();
  await initializeConversations();
  setInterval(checkHealth, 30000);
}

init();
