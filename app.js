let documentId = null;
let lastAnswer = null;
let documentImages = [];

const pdfInput = document.getElementById("pdfInput");
const docPanel = document.getElementById("docPanel");
const messages = document.getElementById("messages");
const workspaceIntro = document.getElementById("workspaceIntro");
const workspaceTitle = document.getElementById("workspaceTitle");
const form = document.getElementById("askForm");
const question = document.getElementById("question");
const depth = document.getElementById("depth");
const askBtn = document.getElementById("askBtn");
const composer = document.getElementById("askForm");
const modeButtons = document.querySelectorAll(".mode-toggle button");
const imagePreview = document.getElementById("imagePreview");
const imagePreviewImage = document.getElementById("imagePreviewImage");
const imagePreviewCaption = document.getElementById("imagePreviewCaption");
const imagePreviewOpen = document.getElementById("imagePreviewOpen");
const imagePreviewClose = document.getElementById("imagePreviewClose");
const imagePreviewBackdrop = document.querySelector(".image-preview-backdrop");

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function apiToken() {
  return localStorage.getItem("neurodocs_api_token") || "";
}

function authHeaders(headers = {}) {
  const token = apiToken();
  return token ? { ...headers, Authorization: `Bearer ${token}` } : headers;
}

function withAuthToken(url) {
  const token = apiToken();
  if (!token) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}token=${encodeURIComponent(token)}`;
}

async function readError(res, fallback) {
  try {
    const data = await res.json();
    return data.detail || fallback;
  } catch {
    return fallback;
  }
}

function clearIntro() {
  if (workspaceIntro?.parentElement === messages) workspaceIntro.remove();
}

function addMessage(role, html, tone = "") {
  if (role !== "system") clearIntro();
  const item = document.createElement("article");
  item.className = `msg ${role}${tone ? ` ${tone}` : ""}`;
  item.innerHTML = `<div class="bubble">${html}</div>`;
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
  return item;
}

function renderWorkspaceIntro(title) {
  if (!workspaceIntro) return;
  workspaceTitle.textContent = title;
  if (workspaceIntro.parentElement !== messages) {
    messages.appendChild(workspaceIntro);
  }
}

function renderDocument(meta) {
  docPanel.className = "doc-panel";
  docPanel.innerHTML = `
    <strong>${escapeHtml(meta.filename)}</strong>
    <div class="doc-meta">
      <span>${meta.pages} pages</span>
      <span>${meta.chunks} chunks</span>
      <span>${meta.image_count} images</span>
      <span>${meta.processing_time}s</span>
    </div>`;
}

function setComposerEnabled(enabled) {
  question.disabled = !enabled;
  askBtn.disabled = !enabled;

  modeButtons.forEach((button) => {
    button.disabled = !enabled;
  });

  const micBtn = document.getElementById("mic-btn");
  if (micBtn) {
    micBtn.disabled = !enabled;
  }

  composer.classList.toggle("locked", !enabled);

  question.placeholder = enabled
    ? "Ask anything in the document..."
    : "Upload a PDF to start asking...";
}

async function loadImages() {
  documentImages = [];
  if (!documentId) return;
  const res = await fetch(`/api/images/${documentId}`, { headers: authHeaders() });
  if (!res.ok) return;
  const data = await res.json();
  documentImages = Array.isArray(data.images)
    ? data.images.map((img) => ({ ...img, url: withAuthToken(img.url) }))
    : [];
}

async function loadExistingDocuments() {
  const res = await fetch("/api/documents", { headers: authHeaders() });
  if (!res.ok) return;

  const data = await res.json();
  const latest = Array.isArray(data.documents) ? data.documents[0] : null;
  if (!latest) return;

  documentId = latest.document_id;
  renderDocument(latest);
  await loadImages();
  renderWorkspaceIntro(latest.filename);
  setComposerEnabled(true);
}

function imagePage(filename) {
  const match = String(filename || "").match(/page_(\d+)_/i);
  return match ? Number(match[1]) : null;
}

function renderRelevantImages(pages) {
  const pageSet = new Set((pages || []).map(Number));
  if (!documentImages.length || !pageSet.size) return "";

  const relevant = documentImages
    .filter((img) => pageSet.has(imagePage(img.filename)))
    .slice(0, 4);

  if (!relevant.length) return "";

  const cards = relevant.map((img) => `
    <figure class="answer-image">
      <button type="button" class="image-open" data-url="${escapeHtml(img.url)}" data-label="${escapeHtml(img.filename)}">
        <img src="${img.url}" alt="${escapeHtml(img.filename)}" title="${escapeHtml(img.filename)}" loading="lazy" />
      </button>
      <span>Page ${imagePage(img.filename)}</span>
    </figure>
  `).join("");

  return `
    <details class="evidence-rail">
      <summary>
        <span>Evidence images</span>
        <small>${relevant.length} matched from cited pages</small>
      </summary>
      <div class="evidence-strip">${cards}</div>
    </details>
  `;
}

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    depth.value = button.dataset.depth;
    modeButtons.forEach((item) => item.classList.toggle("active", item === button));
  });
});

function openImagePreview(url, label) {
  if (!imagePreview || !imagePreviewImage || !url) return;
  imagePreviewImage.src = url;
  imagePreviewImage.alt = label || "Document image";
  imagePreviewCaption.textContent = label || "Document image";
  imagePreviewOpen.href = url;
  imagePreview.classList.remove("hidden");
  imagePreview.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
}

function closeImagePreview() {
  if (!imagePreview || !imagePreviewImage) return;
  imagePreview.classList.add("hidden");
  imagePreview.setAttribute("aria-hidden", "true");
  imagePreviewImage.removeAttribute("src");
  document.body.classList.remove("modal-open");
}

messages.addEventListener("click", (event) => {
  const trigger = event.target.closest(".image-open");
  if (!trigger) return;
  openImagePreview(trigger.dataset.url, trigger.dataset.label);
});

imagePreviewClose?.addEventListener("click", closeImagePreview);
imagePreviewBackdrop?.addEventListener("click", closeImagePreview);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeImagePreview();
});

pdfInput.addEventListener("change", async () => {
  const file = pdfInput.files?.[0];
  if (!file) return;

  setComposerEnabled(false);
  documentId = null;
  lastAnswer = null;
  documentImages = [];
  messages.innerHTML = "";
  renderWorkspaceIntro(file.name);
  addMessage("assistant", `Indexing <strong>${escapeHtml(file.name)}</strong>... Large PDFs can take a minute.`, "system");

  const body = new FormData();
  body.append("file", file);

  const res = await fetch("/api/documents", { method: "POST", headers: authHeaders(), body });
  if (!res.ok) {
    const message = await readError(res, "Upload failed. Check the PDF and backend logs.");
    addMessage("assistant", escapeHtml(message), "system");
    return;
  }

  const meta = await res.json();
  documentId = meta.document_id;
  messages.innerHTML = "";
  renderDocument(meta);
  await loadImages();
  renderWorkspaceIntro(meta.filename);
  setComposerEnabled(true);
  question.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const q = question.value.trim();
  if (!q) return;
  if (!documentId) {
    addMessage("assistant", "Upload a PDF first.", "system");
    return;
  }

  addMessage("user", escapeHtml(q));
  question.value = "";
  askBtn.disabled = true;
  askBtn.textContent = "Wait";
  const pending = addMessage("assistant", "Searching cited evidence...", "system");

  const res = await fetch("/api/ask", {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ document_id: documentId, question: q, top_k: Number(depth.value) }),
  });

  askBtn.disabled = false;
  askBtn.textContent = "Ask";
  if (!res.ok) {
    pending.querySelector(".bubble").innerHTML = escapeHtml(await readError(res, "Answer failed. Backend returned an error."));
    return;
  }

  const data = await res.json();
  lastAnswer = data;
  const noMatch = data.confidence === "none" || !data.sources?.length;
  const pages = data.pages?.length ? data.pages.map((p) => `<span class="cite">p.${p}</span>`).join("") : "";
  const relevantImages = renderRelevantImages(data.pages || []);
  const sources = (data.sources || []).map((source) => `
    <details class="source">
      <summary><span>Source ${source.id}</span><small>p.${source.page ?? "?"}</small></summary>
      <p>${escapeHtml(source.snippet)}</p>
    </details>
  `).join("");

  if (noMatch) {
    pending.classList.add("no-match");
    pending.querySelector(".bubble").innerHTML = `
      <div class="answer-head">
        <strong>No matching evidence</strong>
        <span>${data.answer_time}s</span>
      </div>
      <div class="answer-body">
        <div class="answer-text">${escapeHtml(data.answer)}</div>
      </div>`;
    return;
  }

  pending.querySelector(".bubble").innerHTML = `
    <div class="answer-head">
      <strong>Answer</strong>
      <span>${escapeHtml(data.confidence)} confidence | ${data.answer_time}s</span>
    </div>
    <div class="answer-body">
      <div class="answer-text">${escapeHtml(data.answer)}</div>
      ${pages ? `<div class="citations">${pages}</div>` : ""}
    </div>
    ${relevantImages}
    ${sources ? `<div class="source-list">${sources}</div>` : ""}
    <div class="actions"><button type="button" class="export-btn">Export PDF</button><button type="button" class="speak-btn">Speak</button></div>`;

  pending.querySelector(".export-btn").onclick = () => exportPdf(q, data);
  pending.querySelector(".speak-btn").onclick = () => speak(data.answer);
});

async function exportPdf(q, data) {
  const res = await fetch("/api/export/pdf", {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ question: q, answer: data.answer, sources: data.sources || [] }),
  });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "neurodocs-answer.pdf";
  a.click();
  URL.revokeObjectURL(url);
}

function speak(text) {
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.02;
  window.speechSynthesis.speak(utterance);
}

document.getElementById("newChat").onclick = () => {
  messages.innerHTML = "";
  renderWorkspaceIntro(documentId ? "New question" : "PDF Workspace");
};

setComposerEnabled(false);
loadExistingDocuments();

const micBtn = document.getElementById("mic-btn");

if (micBtn) {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    micBtn.addEventListener("click", () => {
      recognition.start();
      micBtn.textContent = "🎙️";
    });

    recognition.onresult = (event) => {
      question.value = event.results[0][0].transcript;
      question.focus();
      micBtn.textContent = "🎤";
    };

    recognition.onerror = () => {
      micBtn.textContent = "🎤";
    };

    recognition.onend = () => {
      micBtn.textContent = "🎤";
    };
  } else {
    micBtn.style.display = "none";
  }
}