/* PrivacyShield Tactile Security Lab — Client Controller */

const state = {
  documentId: null,
  filename: null,
  sampleKind: "indian_id",
  style: "pixelate",
  showTags: true,
  viewMode: "split", // split | overlay | redacted
  auditTab: "json", // json | ocr
  splitPct: 50,
  originalImage: null,
  redactedImage: null,
  annotatedImage: null,
  entities: [],
  auditLog: [],
  ocrLines: [],
  metrics: null,
  localBytesExchanged: 0,
  externalBytesSent: 0,
};

const els = {
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  generateSampleBtn: document.getElementById("generateSampleBtn"),
  sampleChips: document.querySelectorAll(".chip-btn"),
  modeButtons: document.querySelectorAll(".mode-option"),
  showTagsToggle: document.getElementById("showTagsToggle"),
  viewTabs: document.querySelectorAll(".view-tab"),
  splitPresets: document.getElementById("splitPresets"),
  splitPresetBtns: document.querySelectorAll("#splitPresets .mini-btn"),
  splitPercentLabel: document.getElementById("splitPercentLabel"),
  emptyStage: document.getElementById("emptyStage"),
  compareWrapper: document.getElementById("compareWrapper"),
  compareStage: document.getElementById("compareStage"),
  singleWrapper: document.getElementById("singleWrapper"),
  imgOriginal: document.getElementById("imgOriginal"),
  imgRedacted: document.getElementById("imgRedacted"),
  imgSingle: document.getElementById("imgSingle"),
  redactedLayer: document.getElementById("redactedLayer"),
  sliderDivider: document.getElementById("sliderDivider"),
  processingVeil: document.getElementById("processingVeil"),
  processingLabel: document.getElementById("processingLabel"),
  entityList: document.getElementById("entityList"),
  redactAllBtn: document.getElementById("redactAllBtn"),
  unmaskAllBtn: document.getElementById("unmaskAllBtn"),
  downloadRedactedBtn: document.getElementById("downloadRedactedBtn"),
  downloadAuditBtn: document.getElementById("downloadAuditBtn"),
  auditTabs: document.querySelectorAll(".audit-tab"),
  auditOutput: document.getElementById("auditOutput"),
  docFilenameLabel: document.getElementById("docFilenameLabel"),
  metricLatency: document.getElementById("metricLatency"),
  metricLatencyBreakdown: document.getElementById("metricLatencyBreakdown"),
  metricMemory: document.getElementById("metricMemory"),
  metricEngines: document.getElementById("metricEngines"),
  metricEntities: document.getElementById("metricEntities"),
  metricRedactedState: document.getElementById("metricRedactedState"),
  networkAuditText: document.getElementById("networkAuditText"),
};

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function updateNetworkMonitor(addedLocalBytes = 0) {
  state.localBytesExchanged += addedLocalBytes;
  els.networkAuditText.textContent = `LOCAL IPC: ${formatBytes(state.localBytesExchanged)} · EXTERNAL: ${state.externalBytesSent} B`;
}

async function localApiFetch(url, options = {}) {
  // Strictly enforce same-origin local requests
  const target = new URL(url, window.location.origin);
  if (target.origin !== window.location.origin) {
    throw new Error("Blocked non-local request.");
  }

  let bodyBytes = 0;
  if (typeof options.body === "string") {
    bodyBytes = new Blob([options.body]).size;
  } else if (options.body instanceof FormData) {
    for (const val of options.body.values()) {
      if (val instanceof File || val instanceof Blob) {
        bodyBytes += val.size;
      } else {
        bodyBytes += String(val).length;
      }
    }
  }

  updateNetworkMonitor(bodyBytes);
  const response = await fetch(target.pathname, options);
  if (!response.ok) {
    const errText = await response.text();
    throw new Error(errText || `HTTP ${response.status}`);
  }
  return response.json();
}

function setProcessing(active, label = "RUNNING LOCAL OCR + PII PIPELINE...") {
  els.processingLabel.textContent = label;
  els.processingVeil.classList.toggle("hidden", !active);
}

function setSplitPercent(pct) {
  const clamped = Math.max(0, Math.min(100, pct));
  state.splitPct = clamped;
  els.redactedLayer.style.clipPath = `inset(0 0 0 ${clamped}%)`;
  els.sliderDivider.style.left = `${clamped}%`;
  els.sliderDivider.setAttribute("aria-valuenow", String(Math.round(clamped)));
  els.splitPercentLabel.textContent = `SPLIT ${Math.round(clamped)}%`;
}

function renderViewport() {
  if (!state.originalImage) {
    els.emptyStage.classList.remove("hidden");
    els.compareWrapper.classList.add("hidden");
    els.singleWrapper.classList.add("hidden");
    return;
  }

  els.emptyStage.classList.add("hidden");
  els.splitPresets.classList.toggle("hidden", state.viewMode !== "split");

  if (state.viewMode === "split") {
    els.compareWrapper.classList.remove("hidden");
    els.singleWrapper.classList.add("hidden");
    els.imgOriginal.src = state.originalImage;
    els.imgRedacted.src = state.redactedImage;
    setSplitPercent(state.splitPct);
  } else if (state.viewMode === "overlay") {
    els.compareWrapper.classList.add("hidden");
    els.singleWrapper.classList.remove("hidden");
    els.imgSingle.src = state.annotatedImage;
  } else {
    els.compareWrapper.classList.add("hidden");
    els.singleWrapper.classList.remove("hidden");
    els.imgSingle.src = state.redactedImage;
  }
}

function renderMetrics() {
  if (!state.metrics) return;
  const m = state.metrics;
  els.metricLatency.textContent = `${m.inference_latency_ms} ms`;
  els.metricLatencyBreakdown.textContent = `OCR ${m.ocr_ms} · NLP ${m.nlp_regex_ms} · CV ${m.redaction_ms}`;
  els.metricMemory.textContent = `${m.cpu_memory_mb} MB`;
  els.metricEngines.textContent = `${String(m.ocr_engine || "OCR").toUpperCase()} + SPACY`;
  els.metricEntities.textContent = `${m.active_redaction_count} / ${m.entity_count}`;
  els.metricRedactedState.textContent =
    m.entity_count === 0
      ? "NO PII DETECTED"
      : m.active_redaction_count === m.entity_count
      ? "ALL ENTITIES SHIELDED"
      : `${m.entity_count - m.active_redaction_count} ITEM(S) EXEMPTED`;
}

function renderAuditPanel() {
  els.docFilenameLabel.textContent = state.filename
    ? `${state.filename.toUpperCase()} · ID:${state.documentId}`
    : "NO ACTIVE SESSION";

  if (!state.documentId) return;

  if (state.auditTab === "json") {
    const payload = {
      document_id: state.documentId,
      filename: state.filename,
      redaction_style: state.style,
      metrics: state.metrics,
      privacy_guarantee: {
        mode: "LOCAL_ON_DEVICE",
        cloud_bytes_sent: 0,
      },
      entities: state.auditLog,
    };
    els.auditOutput.textContent = JSON.stringify(payload, null, 2);
  } else {
    els.auditOutput.textContent = state.ocrLines.length
      ? state.ocrLines.map((ln, idx) => `[${String(idx + 1).padStart(2, "0")}] ${ln}`).join("\n")
      : "(No text lines extracted)";
  }
}

function renderEntityInspector() {
  const hasEntities = state.entities.length > 0;
  els.redactAllBtn.disabled = !hasEntities;
  els.unmaskAllBtn.disabled = !hasEntities;
  els.downloadRedactedBtn.disabled = !state.redactedImage;
  els.downloadAuditBtn.disabled = !state.documentId;

  if (!hasEntities) {
    els.entityList.innerHTML = `<div class="entity-empty">${
      state.documentId
        ? "Zero sensitive PII patterns found in this image."
        : "Detected PII bounding boxes (Aadhaar, PAN, UPI, Card, Phone, Email, Person, Address) will appear here with per-item override switches."
    }</div>`;
    return;
  }

  els.entityList.innerHTML = "";
  for (const ent of state.entities) {
    const item = document.createElement("div");
    item.className = `entity-item${ent.enabled ? "" : " exempted"}`;

    const confPct = Math.round(ent.confidence * 100);
    const [x1, y1, x2, y2] = ent.bbox;

    item.innerHTML = `
      <div class="entity-row-top">
        <div class="entity-badges">
          <span class="type-pill">${ent.type}</span>
          <span class="conf-pill">${confPct}% · ${ent.source.toUpperCase()}</span>
        </div>
        <label class="entity-toggle" title="Toggle redaction for ${ent.id}">
          <input type="checkbox" ${ent.enabled ? "checked" : ""} data-entity-id="${ent.id}" />
          <span class="switch-track"><span class="switch-thumb"></span></span>
        </label>
      </div>
      <div class="entity-value"></div>
      <div class="entity-coords">
        <span>${ent.id.toUpperCase()}</span>
        <span>BBOX [${x1}, ${y1}, ${x2}, ${y2}]</span>
      </div>
    `;

    // Safely assign textContent to avoid XSS on arbitrary OCR strings
    item.querySelector(".entity-value").textContent = ent.enabled
      ? `${ent.masked_preview}  (SHIELDED)`
      : `${ent.text}  (VISIBLE)`;

    const checkbox = item.querySelector("input[type='checkbox']");
    checkbox.addEventListener("change", () => {
      ent.enabled = checkbox.checked;
      applyRedactionUpdate();
    });

    els.entityList.appendChild(item);
  }
}

function ingestAnalysisResponse(data) {
  state.documentId = data.document_id;
  if (data.filename) state.filename = data.filename;
  if (data.original_image) state.originalImage = data.original_image;
  state.redactedImage = data.redacted_image;
  state.annotatedImage = data.annotated_image;
  state.entities = data.entities || [];
  state.auditLog = data.audit_log || [];
  if (data.ocr_lines) state.ocrLines = data.ocr_lines;
  state.metrics = data.metrics;

  renderViewport();
  renderMetrics();
  renderEntityInspector();
  renderAuditPanel();
}

async function analyzeFile(file) {
  if (!file) return;
  setProcessing(true, `SCANNING ${file.name.toUpperCase()} LOCALLY...`);
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("style", state.style);
    form.append("show_tags", String(state.showTags));
    const data = await localApiFetch("/api/analyze", { method: "POST", body: form });
    ingestAnalysisResponse(data);
  } catch (err) {
    alert(`Analysis error: ${err.message}`);
  } finally {
    setProcessing(false);
  }
}

async function generateAndAnalyzeSample() {
  setProcessing(true, "GENERATING SYNTHETIC FIXTURE + SCANNING...");
  try {
    const data = await localApiFetch("/api/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        kind: state.sampleKind,
        style: state.style,
        show_tags: state.showTags,
      }),
    });
    ingestAnalysisResponse(data);
  } catch (err) {
    alert(`Sample generation failed: ${err.message}`);
  } finally {
    setProcessing(false);
  }
}

async function applyRedactionUpdate() {
  if (!state.documentId) return;
  const enabledIds = state.entities.filter((e) => e.enabled).map((e) => e.id);
  try {
    const data = await localApiFetch("/api/redact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        document_id: state.documentId,
        enabled_entity_ids: enabledIds,
        style: state.style,
        show_tags: state.showTags,
      }),
    });
    ingestAnalysisResponse(data);
  } catch (err) {
    alert(`Redaction update failed: ${err.message}`);
  }
}

/* EVENT WIRING */
els.dropzone.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    els.fileInput.click();
  }
});

els.fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    analyzeFile(e.target.files[0]);
    e.target.value = "";
  }
});

["dragenter", "dragover"].forEach((evt) => {
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((evt) => {
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("dragover");
  });
});

els.dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) analyzeFile(file);
});

els.sampleChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    els.sampleChips.forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    state.sampleKind = chip.dataset.sample;
  });
});

els.generateSampleBtn.addEventListener("click", generateAndAnalyzeSample);

els.modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    els.modeButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.style = btn.dataset.style;
    applyRedactionUpdate();
  });
});

els.showTagsToggle.addEventListener("change", () => {
  state.showTags = els.showTagsToggle.checked;
  if (state.style === "matte_bar") {
    applyRedactionUpdate();
  }
});

els.viewTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    els.viewTabs.forEach((t) => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    state.viewMode = tab.dataset.view;
    renderViewport();
  });
});

els.splitPresetBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    setSplitPercent(Number(btn.dataset.split));
  });
});

els.auditTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    els.auditTabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    state.auditTab = tab.dataset.auditTab;
    renderAuditPanel();
  });
});

els.redactAllBtn.addEventListener("click", () => {
  state.entities.forEach((e) => (e.enabled = true));
  applyRedactionUpdate();
});

els.unmaskAllBtn.addEventListener("click", () => {
  state.entities.forEach((e) => (e.enabled = false));
  applyRedactionUpdate();
});

els.downloadRedactedBtn.addEventListener("click", () => {
  if (!state.redactedImage) return;
  const a = document.createElement("a");
  a.href = state.redactedImage;
  const base = (state.filename || "document").replace(/\.[^/.]+$/, "");
  a.download = `${base}_shielded.png`;
  a.click();
});

els.downloadAuditBtn.addEventListener("click", () => {
  if (!state.documentId) return;
  const blob = new Blob(
    [
      JSON.stringify(
        {
          document_id: state.documentId,
          filename: state.filename,
          style: state.style,
          metrics: state.metrics,
          audit_log: state.auditLog,
        },
        null,
        2
      ),
    ],
    { type: "application/json" }
  );
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `privacyshield_audit_${state.documentId}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

/* DRAGGABLE BEFORE / AFTER COMPARISON SLIDER */
let draggingSlider = false;

function updateSliderFromPointer(clientX) {
  const rect = els.compareStage.getBoundingClientRect();
  if (rect.width <= 0) return;
  const pct = ((clientX - rect.left) / rect.width) * 100;
  setSplitPercent(pct);
}

els.compareStage.addEventListener("pointerdown", (e) => {
  if (!state.originalImage || state.viewMode !== "split") return;
  draggingSlider = true;
  els.compareStage.setPointerCapture(e.pointerId);
  updateSliderFromPointer(e.clientX);
});

els.compareStage.addEventListener("pointermove", (e) => {
  if (!draggingSlider) return;
  updateSliderFromPointer(e.clientX);
});

els.compareStage.addEventListener("pointerup", (e) => {
  if (!draggingSlider) return;
  draggingSlider = false;
  if (els.compareStage.hasPointerCapture(e.pointerId)) {
    els.compareStage.releasePointerCapture(e.pointerId);
  }
});

els.sliderDivider.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    setSplitPercent(state.splitPct - 5);
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    setSplitPercent(state.splitPct + 5);
  }
});

/* INITIAL LOCAL HEALTH PING */
(async function initHealthCheck() {
  try {
    const res = await localApiFetch("/api/health");
    els.metricMemory.textContent = `${res.cpu_memory_mb} MB`;
    els.metricEngines.textContent = `${String(res.ocr_engine || "OCR").toUpperCase()} + SPACY`;
  } catch {
    els.metricEngines.textContent = "SERVER DISCONNECTED";
  }
})();
