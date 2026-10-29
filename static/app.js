const SENTIMENT_LABELS = { positive: "Positivo", negative: "Negativo", neutral: "Neutro" };
const CATEGORY_LABELS = {
  insult: "insulto",
  threat: "amenaza",
  identity_hate: "odio por identidad",
  sexual_harassment: "acoso sexual",
  profanity: "puteada",
};
const STAT_KEYS = ["positive", "negative", "neutral", "flagged"];

const els = {
  thread: document.getElementById("thread"),
  messages: document.getElementById("messages"),
  empty: document.getElementById("empty"),
  form: document.getElementById("composer"),
  text: document.getElementById("text"),
  send: document.getElementById("send"),
  error: document.getElementById("error"),
  total: document.getElementById("stat-total"),
};

class ApiError extends Error {
  constructor(status, body) {
    super(`HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (!response.ok) {
    throw new ApiError(response.status, await response.json().catch(() => null));
  }
  return response.json();
}

const pct = (x) => `${Math.round(x * 100)}%`;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const capitalize = (text) => text.charAt(0).toUpperCase() + text.slice(1);

function hiddenLabel(flag) {
  return `Mensaje oculto por posible ${CATEGORY_LABELS[flag.category]}. Activalo para mostrarlo.`;
}

function formatTime(iso) {
  return new Date(iso).toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit" });
}

function renderMessage(message) {
  const item = el("li", message.flagged ? "msg is-flagged" : "msg");
  const bubble = el("div", "bubble");
  const text = el("p", "text", message.text);
  // What could get the message flagged: the most likely of insult, threat and identity hate.
  const flag = message.flag;

  if (message.flagged) {
    const note = el("p", "flag-note", `⚠ Posible ${CATEGORY_LABELS[flag.category]} (${pct(flag.score)}) `);
    note.append(el("span", "flag-hint", "Pasá el mouse o tocá para verlo"));
    bubble.append(note);

    text.classList.add("spoiler");
    text.tabIndex = 0;
    text.setAttribute("role", "button");
    text.setAttribute("aria-pressed", "false");
    text.setAttribute("aria-label", hiddenLabel(flag));
    text.dataset.hiddenLabel = hiddenLabel(flag);
  }
  bubble.append(text);

  const sentiment = message.sentiment;
  const meta = el("div", "meta");
  const sentimentChip = el("span", "sentiment", `${SENTIMENT_LABELS[sentiment.label]} ${pct(sentiment.confidence)}`);
  sentimentChip.dataset.sentiment = sentiment.label;
  sentimentChip.title = Object.entries(sentiment.probabilities)
    .map(([label, p]) => `${SENTIMENT_LABELS[label] ?? label}: ${pct(p)}`)
    .join("\n");

  const time = el("time", "", formatTime(message.created_at));
  time.dateTime = message.created_at;

  const flagChip = el("span", "", `${capitalize(CATEGORY_LABELS[flag.category])} ${pct(flag.score)}`);
  flagChip.title = Object.entries(message.categories)
    .map(([category, p]) => `${capitalize(CATEGORY_LABELS[category] ?? category)}: ${pct(p)}`)
    .join("\n");

  meta.append(
    sentimentChip,
    flagChip,
    el("span", "model", message.model),
    el("span", "", `${message.latency_ms} ms`),
    time,
  );

  item.append(bubble, meta, renderLabeler(message));
  return item;
}

// Two buttons that record the right answer for this message as a fine-tuning example.
function renderLabeler(message) {
  const row = el("div", "labeler");
  row.append(el("span", "", "¿Es insulto?"));
  for (const [label, text] of [["insult", "Sí"], ["clean", "No"]]) {
    const button = el("button", "label-btn", text);
    button.type = "button";
    button.dataset.messageId = message.id;
    button.dataset.label = label;
    button.setAttribute("aria-pressed", String(message.label === label));
    row.append(button);
  }
  return row;
}

async function labelMessage(button) {
  const row = button.closest(".labeler");
  const buttons = row.querySelectorAll(".label-btn");
  buttons.forEach((b) => (b.disabled = true));
  try {
    const example = await api(`/api/messages/${button.dataset.messageId}/label`, {
      method: "POST",
      body: JSON.stringify({ label: button.dataset.label }),
    });
    buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.label === example.label)));
  } catch (error) {
    showError(describeError(error));
  } finally {
    buttons.forEach((b) => (b.disabled = false));
  }
}

function renderPending() {
  const item = el("li", "msg is-pending");
  const bubble = el("div", "bubble");
  bubble.append(el("p", "text pending-note", "Evaluando…"));
  item.append(bubble);
  return item;
}

function renderStats(stats) {
  els.total.textContent = stats.total === 1 ? "1 mensaje" : `${stats.total} mensajes`;
  for (const key of STAT_KEYS) {
    document.getElementById(`stat-${key}`).textContent = stats[key];
    document.getElementById(`share-${key}`).textContent =
      stats.total ? `${pct(stats[key] / stats.total)} del total` : "";
  }
}

function syncEmptyState() {
  els.empty.hidden = els.messages.children.length > 0;
}

function scrollToBottom() {
  els.thread.scrollTop = els.thread.scrollHeight;
}

function showError(message) {
  els.error.textContent = message;
  els.error.hidden = false;
}

function describeError(error) {
  if (error instanceof ApiError) {
    if (error.status === 422) return "El mensaje tiene que tener entre 1 y 1000 caracteres.";
    return `El servidor respondió con un error (${error.status}). Probá de nuevo.`;
  }
  return "No hay conexión con el servidor. Revisá que esté corriendo y probá de nuevo.";
}

function toggleSpoiler(target) {
  const revealed = target.classList.toggle("revealed");
  target.setAttribute("aria-pressed", String(revealed));
  if (revealed) target.removeAttribute("aria-label");
  else target.setAttribute("aria-label", target.dataset.hiddenLabel);
}

function autoGrow() {
  const textarea = els.text;
  textarea.style.height = "auto";
  // scrollHeight leaves out the border, which border-box sizing counts in the height.
  const border = textarea.offsetHeight - textarea.clientHeight;
  textarea.style.height = `${textarea.scrollHeight + border}px`;
}

async function sendMessage() {
  const text = els.text.value.trim();
  if (!text) return;

  els.error.hidden = true;
  els.send.disabled = true;
  const pending = renderPending();
  els.messages.append(pending);
  syncEmptyState();
  scrollToBottom();

  try {
    const message = await api("/api/messages", { method: "POST", body: JSON.stringify({ text }) });
    pending.replaceWith(renderMessage(message));
    els.text.value = "";
    autoGrow();
    renderStats(await api("/api/stats"));
  } catch (error) {
    pending.remove();
    syncEmptyState();
    showError(describeError(error));
  } finally {
    els.send.disabled = false;
    els.text.focus();
    scrollToBottom();
  }
}

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});

els.text.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!els.send.disabled) sendMessage();
  }
});

els.text.addEventListener("input", autoGrow);

els.messages.addEventListener("click", (event) => {
  const labelButton = event.target.closest(".label-btn");
  if (labelButton) {
    labelMessage(labelButton);
    return;
  }
  const spoiler = event.target.closest(".spoiler");
  if (spoiler) toggleSpoiler(spoiler);
});

els.messages.addEventListener("keydown", (event) => {
  const spoiler = event.target.closest(".spoiler");
  if (spoiler && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    toggleSpoiler(spoiler);
  }
});

async function init() {
  try {
    const [messages, stats] = await Promise.all([api("/api/messages"), api("/api/stats")]);
    els.messages.replaceChildren(...messages.map(renderMessage));
    renderStats(stats);
  } catch (error) {
    showError(describeError(error));
  }
  syncEmptyState();
  scrollToBottom();
}

init();
