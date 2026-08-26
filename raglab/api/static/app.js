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
let activeThreadId = localStorage.getItem(ACTIVE_THREAD_KEY);
let busy = false;

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function saveActiveThread() {
  if (activeThreadId) localStorage.setItem(ACTIVE_THREAD_KEY, activeThreadId);
  else localStorage.removeItem(ACTIVE_THREAD_KEY);
}

function currentSession() {
  return sessions.find((session) => session.threadId === activeThreadId) || null;
}

function updateInteractionState() {
  const pending = Boolean(currentSession()?.pending);
  const inputLocked = busy || pending;

  // 同一个 thread 停在 LangGraph interrupt 时，只允许用户处理审批。
  // 如果继续发送普通消息，会与 Checkpoint 中尚未恢复的任务发生冲突。
  elements.sendButton.disabled = inputLocked;
  // disabled 是浏览器原生硬限制；readOnly 作为第二层限制，并让页面在
  // 某些浏览器恢复表单状态时也不能继续修改输入内容。
  elements.question.disabled = inputLocked;
  elements.question.readOnly = inputLocked;
  elements.form.classList.toggle("locked", inputLocked);
  elements.form.setAttribute("aria-busy", String(busy));
  elements.approveButton.disabled = busy || !pending;
  elements.rejectButton.disabled = busy || !pending;
  elements.question.placeholder = busy
    ? "Agent 正在执行，请等待本轮完成"
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
  const userId = elements.userId.value.trim() || "local-user";
  const body = await request(`/threads?user_id=${encodeURIComponent(userId)}`, { headers: {} });
  sessions = body.threads.map((thread) => ({
    threadId: thread.thread_id,
    title: thread.title || "新会话",
    messages: [],
    pending: null,
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
  const hasMessages = Boolean(session?.messages?.length);

  if (!hasMessages) {
    const welcome = document.querySelector("#welcome")?.cloneNode(true) || buildWelcome();
    welcome.id = "welcome";
    elements.messages.append(welcome);
    bindSuggestions(welcome);
    return;
  }

  for (const message of session.messages) {
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
  { addUserMessage = true, allowPending = false, initialStatus = null } = {},
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
    let expectedRunId = null;
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

        // accepted 确定本次连接的 run_id。之后凡是带运行身份的事件，
        // 必须同时属于当前 thread 和当前 run，否则直接忽略。
        if (eventName === "accepted") expectedRunId = data.run_id || null;
        if (data.thread_id && data.thread_id !== session.threadId) continue;
        if (data.run_id && expectedRunId && data.run_id !== expectedRunId) continue;

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
    progress.role = "assistant";
    progress.content = finalBody.answer || "Agent 没有返回文本答案。";
    progress.trace = responseTrace(finalBody, streamingEvents);
    renderAll();
    if (!session.pending) await refreshPending();
  } catch (error) {
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
    if (activeThreadId === session.threadId) renderApproval();
  } catch (error) {
    console.warn("pending query failed or timed out", error);
  } finally {
    clearTimeout(timeoutId);
  }
}

async function decide(endpoint) {
  const session = currentSession();
  if (!session || busy || !session.pending) return;
  const actionText = endpoint === "approve" ? "批准" : "拒绝";
  // 审批也复用 /chat/stream。它不是一条用户对话，所以不把 /approve
  // 或 /reject 显示成用户消息，但恢复后的节点事件会沿同一 SSE 返回。
  await sendQuestion(`/${endpoint}`, {
    addUserMessage: false,
    allowPending: true,
    initialStatus: `正在提交${actionText}决定，并从 LangGraph Checkpoint 恢复执行…`,
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
    if (!sessions.length) {
      await createSession();
      return;
    }
    if (!sessions.some((session) => session.threadId === activeThreadId)) {
      activeThreadId = sessions[0].threadId;
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
