const LABEL_TEXT = { insult: "insulto", clean: "no insulto" };
const CATEGORY_TEXT = { insult: "insulto", threat: "amenaza", identity_hate: "odio por identidad" };
const SOURCE_TEXT = { chat: "Del chat", generated: "Generado", imported: "Importado" };
const KEY_LABELS = { 1: "insult", 2: "clean", 3: "discard" };

const els = {
  labeledTotal: document.getElementById("labeled-total"),
  target: document.getElementById("target"),
  meter: document.getElementById("meter"),
  meterFill: document.getElementById("meter-fill"),
  counts: {
    insult: document.getElementById("count-insult"),
    clean: document.getElementById("count-clean"),
    pending: document.getElementById("count-pending"),
    discarded: document.getElementById("count-discarded"),
  },
  card: document.getElementById("review-card"),
  text: document.getElementById("example-text"),
  meta: document.getElementById("example-meta"),
  empty: document.getElementById("review-empty"),
  error: document.getElementById("review-error"),
  modelStatus: document.getElementById("model-status"),
  buttons: document.querySelectorAll(".review-btn"),
};

let current = null;
let busy = false;

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

const pct = (x) => `${Math.round(x * 100)}%`;

function span(text) {
  const node = document.createElement("span");
  node.textContent = text;
  return node;
}

function showError(message) {
  els.error.textContent = message;
  els.error.hidden = false;
}

function describeModel(active) {
  if (!active) return "Todavía no hay un modelo activo. Entrenalo con el comando de abajo.";
  const date = new Date(active.activated_at).toLocaleString("es-AR", { dateStyle: "medium", timeStyle: "short" });
  const auroc = active.metrics.flag_auroc?.["suite-es"];
  const quality = auroc == null ? "" : ` En el chat en español que no vio al entrenar separa con AUROC ${auroc.toFixed(2)} (1.00 es perfecto).`;
  // The date ends in "p. m." or "a. m.", so it goes last and closes the sentence.
  return `mmBERT afinado (${active.name}).${quality} Activado el ${date}`;
}

async function refreshSummary() {
  const s = await api("/api/training/summary");
  const total = s.labeled.insult + s.labeled.clean;
  els.labeledTotal.textContent = total;
  els.target.textContent = s.target;
  els.meter.setAttribute("aria-valuemax", s.target);
  els.meter.setAttribute("aria-valuenow", total);
  els.meterFill.style.width = `${Math.min(100, (total / s.target) * 100)}%`;
  els.counts.insult.textContent = s.labeled.insult;
  els.counts.clean.textContent = s.labeled.clean;
  els.counts.pending.textContent = s.pending;
  els.counts.discarded.textContent = s.discarded;
  els.modelStatus.textContent = describeModel(s.active_model);
}

async function loadNext() {
  const { example, prediction } = await api("/api/training/next");
  current = example;
  els.card.hidden = !example;
  els.empty.hidden = Boolean(example);
  if (!example) return;

  els.text.textContent = example.text;
  const parts = [SOURCE_TEXT[example.source] ?? example.source];
  if (example.category) parts.push(example.category);
  if (example.suggested) parts.push(`Sugerido: ${LABEL_TEXT[example.suggested]}`);
  parts.push(`Modelo hoy: ${pct(prediction.p_flag)} ${CATEGORY_TEXT[prediction.category]} (${prediction.model})`);
  els.meta.replaceChildren(...parts.map(span));
}

async function review(label) {
  if (!current || busy) return;
  busy = true;
  els.error.hidden = true;
  els.buttons.forEach((b) => (b.disabled = true));
  try {
    await api(`/api/training/examples/${current.id}/label`, { method: "POST", body: JSON.stringify({ label }) });
    await Promise.all([loadNext(), refreshSummary()]);
  } catch {
    showError("No se pudo guardar la etiqueta. Revisá que el server esté corriendo y probá de nuevo.");
  } finally {
    busy = false;
    els.buttons.forEach((b) => (b.disabled = false));
  }
}

document.querySelector(".review-actions").addEventListener("click", (event) => {
  const button = event.target.closest(".review-btn");
  if (button) review(button.dataset.label);
});

document.addEventListener("keydown", (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const label = KEY_LABELS[event.key];
  if (label) {
    event.preventDefault();
    review(label);
  }
});

Promise.all([refreshSummary(), loadNext()]).catch(() =>
  showError("No hay conexión con el servidor. Revisá que esté corriendo y recargá la página."),
);
