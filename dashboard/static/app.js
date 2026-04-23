// Dashboard frontend logic. Talks to FastAPI backend.

// Tab switching — wired first via event delegation so the nav works
// regardless of any later runtime error in this file.
document.addEventListener('click', (e) => {
  const btn = e.target.closest('.tab-btn');
  if (!btn) return;
  const target = btn.dataset.tab;
  if (!target) return;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  const panel = document.querySelector(`.tab-panel[data-tab="${target}"]`);
  if (panel) panel.classList.add('active');
  if (target === 'lmstudio' && typeof lmsRefresh === 'function') lmsRefresh();
});

const $ = (sel) => document.querySelector(sel);

const els = {
  models: $('#models-list'),
  datasets: $('#datasets-list'),
  runs: $('#runs-list'),
  trainModel: $('#train-model'),
  trainDataset: $('#train-dataset'),
  log: $('#log-pane'),
  status: $('#job-status'),
  progress: $('#job-progress'),
  cancel: $('#btn-cancel'),
  snackbar: $('#snackbar'),
  gpuChecks: $('#gpu-checkboxes'),
  gpuSummary: $('#gpu-summary'),
  gpuNote: $('#gpu-profile-note'),
  gpuHeteroWarn: $('#gpu-hetero-warn'),
  gpuHeteroDetail: $('#gpu-hetero-detail'),
  numGpus: $('#train-num-gpus'),
};

let gpuStateApplied = false;
let lastGpuState = null;
let selectedGpuIds = null;        // null = use all visible
let excludeSmallest = false;

let currentJobRunning = false;

function toast(msg, opts = {}) {
  const sb = els.snackbar;
  if (sb && typeof sb.show === 'function') {
    sb.label = msg;
    sb.show();
  } else {
    console.log('[toast]', msg);
  }
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    let detail;
    try { detail = (await res.json()).detail; } catch {}
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

function makeListItem(text, supporting) {
  const item = document.createElement('md-list-item');
  item.setAttribute('type', 'text');
  const headline = document.createElement('div');
  headline.slot = 'headline';
  headline.textContent = text;
  item.appendChild(headline);
  if (supporting) {
    const sup = document.createElement('div');
    sup.slot = 'supporting-text';
    sup.textContent = supporting;
    item.appendChild(sup);
  }
  return item;
}

function makeOption(value, label) {
  const opt = document.createElement('md-select-option');
  opt.value = value;
  const headline = document.createElement('div');
  headline.slot = 'headline';
  headline.textContent = label;
  opt.appendChild(headline);
  return opt;
}

function renderModels(models) {
  els.models.innerHTML = '';
  els.trainModel.innerHTML = '';
  if (!models.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No models yet — download one above.';
    els.models.appendChild(empty);
    return;
  }
  const blank = makeOption('', '— select a model —');
  els.trainModel.appendChild(blank);
  for (const m of models) {
    els.models.appendChild(makeListItem(m.name, m.has_config ? m.path : `${m.path} (no config.json)`));
    els.trainModel.appendChild(makeOption(m.name, m.name));
  }
}

function renderDatasets(datasets) {
  els.datasets.innerHTML = '';
  els.trainDataset.innerHTML = '';
  if (!datasets.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No datasets uploaded yet.';
    els.datasets.appendChild(empty);
    return;
  }
  const blank = makeOption('', '— select a dataset —');
  els.trainDataset.appendChild(blank);
  for (const d of datasets) {
    els.datasets.appendChild(
      makeListItem(d.name, `${d.lines} records · ${(d.size / 1024).toFixed(1)} KiB`)
    );
    els.trainDataset.appendChild(makeOption(d.name, `${d.name} (${d.lines} rec)`));
  }
}

function renderRuns(runs) {
  els.runs.innerHTML = '';
  if (!runs.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No completed runs yet.';
    els.runs.appendChild(empty);
    return;
  }
  for (const r of runs) {
    const when = new Date(r.modified * 1000).toLocaleString();
    const tag = r.has_adapter ? 'adapter saved' : 'in-progress / no adapter';
    els.runs.appendChild(makeListItem(r.name, `${tag} · ${when} · ${r.path}`));
  }
}

function setStatus(jobStatus) {
  const running = jobStatus.running;
  currentJobRunning = running;
  els.cancel.disabled = !running;
  els.progress.style.display = running ? 'block' : 'none';

  els.status.classList.remove('running', 'error', 'done');
  if (running) {
    els.status.textContent = `${jobStatus.kind || 'job'} running`;
    els.status.classList.add('running');
  } else if (jobStatus.return_code === 0) {
    els.status.textContent = `${jobStatus.label || 'last job'} done`;
    els.status.classList.add('done');
  } else if (jobStatus.return_code != null && jobStatus.return_code !== 0) {
    els.status.textContent = `failed (exit ${jobStatus.return_code})`;
    els.status.classList.add('error');
  } else {
    els.status.textContent = 'idle';
  }
}

function renderGPUs(gpu) {
  if (!gpu) return;
  lastGpuState = gpu;
  els.gpuChecks.innerHTML = '';
  if (!gpu.gpus || gpu.gpus.length === 0) {
    els.gpuSummary.textContent = 'CPU only';
    els.gpuSummary.classList.remove('running', 'done');
    els.gpuSummary.classList.add('error');
    els.gpuHeteroWarn.hidden = true;
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No NVIDIA GPU detected — training falls back to CPU (slow).';
    els.gpuChecks.appendChild(empty);
  } else {
    els.gpuSummary.textContent = `${gpu.num_gpus}× GPU · ${gpu.total_vram_gb} GB total`;
    els.gpuSummary.classList.remove('error');
    els.gpuSummary.classList.add('done');

    const inMajority = new Set(gpu.majority_subset_indices || []);
    if (selectedGpuIds === null) {
      // First render: select all by default.
      selectedGpuIds = gpu.gpus.map(g => g.index);
    }
    const selectedSet = new Set(selectedGpuIds);

    for (const g of gpu.gpus) {
      const wrap = document.createElement('label');
      wrap.className = 'gpu-check' + (gpu.heterogeneous && !inMajority.has(g.index) ? ' outlier' : '');
      const cb = document.createElement('md-checkbox');
      cb.checked = selectedSet.has(g.index);
      cb.addEventListener('change', () => {
        if (cb.checked) selectedSet.add(g.index); else selectedSet.delete(g.index);
        selectedGpuIds = Array.from(selectedSet).sort((a, b) => a - b);
        excludeSmallest = false;     // explicit click overrides the auto flag
      });
      const text = document.createElement('span');
      text.innerHTML = `<strong>#${g.index}</strong> ${g.name} · ${g.vram_gb} GB`;
      wrap.appendChild(cb);
      wrap.appendChild(text);
      els.gpuChecks.appendChild(wrap);
    }

    if (gpu.heterogeneous) {
      els.gpuHeteroWarn.hidden = false;
      const out = gpu.gpus.filter(g => !inMajority.has(g.index)).map(g => `#${g.index} (${g.vram_gb} GB)`);
      els.gpuHeteroDetail.textContent =
        `Profile clamps to the smallest GPU (${gpu.per_gpu_vram_gb} GB). ` +
        `Excluding ${out.join(', ')} would unlock the ` +
        `${gpu.majority_subset_vram_gb} GB tier across ${(gpu.majority_subset_indices || []).length} GPUs.`;
    } else {
      els.gpuHeteroWarn.hidden = true;
    }
  }
  if (gpu.profile) {
    els.gpuNote.textContent =
      `Auto-tune profile: ${gpu.profile.name} — ` +
      `batch=${gpu.profile.batch_size}, grad_accum=${gpu.profile.grad_accum}, ` +
      `max_len=${gpu.profile.max_length}, lora_r=${gpu.profile.lora_r}, ` +
      `4bit=${gpu.profile.use_4bit}, grad_ckpt=${gpu.profile.gradient_checkpointing}. ` +
      `Recommended max model: ~${gpu.profile.max_recommended_params_b}B params.`;
  }
  if (!gpuStateApplied && els.numGpus) {
    els.numGpus.value = String(gpu.num_gpus || 1);
    els.numGpus.max = String(Math.max(1, gpu.num_gpus || 1));
    gpuStateApplied = true;
  }
}

document.addEventListener('click', (e) => {
  if (e.target && e.target.id === 'btn-exclude-smallest') {
    excludeSmallest = true;
    if (lastGpuState && lastGpuState.majority_subset_indices) {
      selectedGpuIds = [...lastGpuState.majority_subset_indices];
      renderGPUs(lastGpuState);
    }
    toast('Will exclude smallest GPU(s) on next training run.');
  }
});

async function refreshState() {
  try {
    const state = await api('/api/state');
    renderModels(state.models);
    renderDatasets(state.datasets);
    renderRuns(state.runs);
    renderGPUs(state.gpu);
    setStatus(state.job);
    if (state.job.log_tail && state.job.log_tail.length && !window.__sseConnected) {
      els.log.textContent = state.job.log_tail.join('\n');
    }
  } catch (e) {
    toast(`refresh failed: ${e.message}`);
  }
}

function openLogStream() {
  const src = new EventSource('/api/logs/stream');
  window.__sseConnected = true;
  src.addEventListener('log', (ev) => {
    const wasAtBottom = els.log.scrollTop + els.log.clientHeight >= els.log.scrollHeight - 20;
    if (els.log.textContent === 'No job running.') els.log.textContent = '';
    els.log.textContent += (els.log.textContent ? '\n' : '') + ev.data;
    if (wasAtBottom) els.log.scrollTop = els.log.scrollHeight;
  });
  src.addEventListener('status', (ev) => {
    try { setStatus(JSON.parse(ev.data)); } catch {}
  });
  src.onerror = () => {
    window.__sseConnected = false;
    setTimeout(openLogStream, 2000);
  };
}

// ---- wire up controls ----

$('#btn-download').addEventListener('click', async () => {
  const repo = $('#dl-repo').value.trim();
  const token = $('#dl-token').value.trim() || null;
  if (!repo) return toast('Enter a HF repo id');
  try {
    await api('/api/download', {
      method: 'POST',
      body: JSON.stringify({ repo_id: repo, token }),
    });
    els.log.textContent = '';
    toast(`Downloading ${repo}...`);
    setTimeout(refreshState, 1500);
  } catch (e) {
    toast(e.message);
  }
});

const fileInput = $('#ds-file');
$('#btn-pick-file').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  const f = fileInput.files?.[0];
  $('#ds-filename').textContent = f ? f.name : '';
  $('#btn-upload').disabled = !f;
});

$('#btn-upload').addEventListener('click', async () => {
  const f = fileInput.files?.[0];
  if (!f) return;
  const fd = new FormData();
  fd.append('file', f);
  try {
    const res = await fetch('/api/upload-dataset', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    toast(`Uploaded ${f.name}`);
    fileInput.value = '';
    $('#ds-filename').textContent = '';
    $('#btn-upload').disabled = true;
    refreshState();
  } catch (e) {
    toast(`upload failed: ${e.message}`);
  }
});

$('#btn-hf-fetch').addEventListener('click', async () => {
  const repo = $('#ds-hf-repo').value.trim();
  const split = $('#ds-hf-split').value.trim() || 'train';
  const text_field = $('#ds-hf-textfield').value.trim() || null;
  if (!repo) return toast('Enter a HF dataset id');
  try {
    await api('/api/download-dataset', {
      method: 'POST',
      body: JSON.stringify({ repo_id: repo, split, text_field }),
    });
    els.log.textContent = '';
    toast(`Fetching ${repo}...`);
    setTimeout(refreshState, 1500);
  } catch (e) {
    toast(e.message);
  }
});

function intOrNull(s) {
  const n = parseInt(s, 10);
  return Number.isFinite(n) ? n : null;
}

$('#btn-train').addEventListener('click', async () => {
  // If the user touched the checkboxes, send the explicit list. Otherwise let
  // the backend use all visible GPUs (or honor "exclude smallest" if clicked).
  const allSelected =
    selectedGpuIds && lastGpuState &&
    lastGpuState.gpus && selectedGpuIds.length === lastGpuState.gpus.length;

  const payload = {
    model: els.trainModel.value,
    dataset: els.trainDataset.value,
    output: $('#train-output').value.trim() || 'run',
    epochs: parseFloat($('#train-epochs').value) || 3,
    auto_tune: $('#train-autotune').selected,
    num_gpus: intOrNull($('#train-num-gpus').value),
    gpu_ids: (allSelected || !selectedGpuIds) ? null : selectedGpuIds,
    exclude_smallest: excludeSmallest,
    strategy: $('#train-strategy').value || 'auto',
    mixed_precision: $('#train-precision').value || 'bf16',
    batch_size: intOrNull($('#train-batch').value),
    grad_accum: intOrNull($('#train-grad').value),
    lr: parseFloat($('#train-lr').value) || 2e-4,
    max_length: intOrNull($('#train-maxlen').value),
    lora_r: intOrNull($('#train-r').value),
    lora_alpha: intOrNull($('#train-alpha').value),
    no_4bit: $('#train-no4bit').selected,
    gradient_checkpointing: $('#train-gc').selected ? true : null,
    target_modules: $('#train-targets').value.trim(),
    attn_impl: $('#train-attn').value || null,
    no_packing: !$('#train-packing').selected,
    no_group_by_length: !$('#train-groupbylen').selected,
    compile: $('#train-compile').selected,
    num_workers: intOrNull($('#train-workers').value) || 4,
  };
  if (!payload.model) return toast('Pick a base model');
  if (!payload.dataset) return toast('Pick a dataset');
  try {
    const res = await api('/api/train', { method: 'POST', body: JSON.stringify(payload) });
    els.log.textContent = '';
    toast(`Training started · ${res.num_gpus}× ${res.strategy}`);
    setTimeout(refreshState, 1500);
  } catch (e) {
    toast(e.message);
  }
});

$('#btn-cancel').addEventListener('click', async () => {
  try {
    await api('/api/cancel', { method: 'POST' });
    toast('Cancel requested');
  } catch (e) {
    toast(e.message);
  }
});

// ---- agent ----

const agentEls = {
  pane: $('#agent-pane'),
  status: $('#agent-status'),
  pending: $('#agent-pending'),
  pendingCmd: $('#agent-pending-cmd'),
  model: $('#agent-model'),
  adapter: $('#agent-adapter'),
};

function renderAgentChoices(state) {
  agentEls.model.innerHTML = '';
  for (const m of state.models || []) {
    agentEls.model.appendChild(makeOption(m.name, m.name));
  }
  agentEls.adapter.innerHTML = '';
  agentEls.adapter.appendChild(makeOption('', '— none (use base only) —'));
  for (const r of state.runs || []) {
    if (r.has_adapter) {
      agentEls.adapter.appendChild(makeOption(r.name, r.name));
    }
  }
}

function appendAgentEvent(ev) {
  let line = '';
  switch (ev.kind) {
    case 'start':       line = `▶ ${ev.text}`; break;
    case 'model':       line = `[model #${ev.iteration}] ${ev.text}`; break;
    case 'tool_call':   line = `[tool_call #${ev.iteration}] $ ${ev.cmd}`; break;
    case 'tool_result': line = `[tool_result rc=${ev.return_code}]\n${ev.stdout || ''}${ev.stderr ? '\n[stderr]\n'+ev.stderr : ''}`; break;
    case 'blocked':     line = `[BLOCKED] ${ev.cmd} (${ev.text})`; break;
    case 'skipped':     line = `[skipped] ${ev.cmd}`; break;
    case 'done':        line = `■ ${ev.text}`; break;
    case 'error':       line = `[ERROR] ${ev.text}`; break;
    default:            line = JSON.stringify(ev);
  }
  if (agentEls.pane.textContent === 'No agent session yet.') agentEls.pane.textContent = '';
  agentEls.pane.textContent += (agentEls.pane.textContent ? '\n\n' : '') + line;
  agentEls.pane.scrollTop = agentEls.pane.scrollHeight;
}

function setAgentStatus(running, pending) {
  agentEls.status.classList.remove('running', 'done', 'error');
  if (running) {
    agentEls.status.textContent = pending ? 'awaiting approval' : 'running';
    agentEls.status.classList.add(pending ? 'error' : 'running');
  } else {
    agentEls.status.textContent = 'idle';
  }
  $('#btn-agent-cancel').disabled = !running;
  if (pending) {
    agentEls.pendingCmd.textContent = pending;
    agentEls.pending.hidden = false;
  } else {
    agentEls.pending.hidden = true;
  }
}

function openAgentStream() {
  const src = new EventSource('/api/agent/stream');
  src.addEventListener('agent', (ev) => {
    try { appendAgentEvent(JSON.parse(ev.data)); } catch {}
  });
  src.addEventListener('status', (ev) => {
    try {
      const s = JSON.parse(ev.data);
      setAgentStatus(s.running, s.pending_command);
    } catch {}
  });
  src.onerror = () => setTimeout(openAgentStream, 2000);
}

$('#btn-agent-run').addEventListener('click', async () => {
  const payload = {
    model: agentEls.model.value,
    adapter: agentEls.adapter.value || null,
    goal: $('#agent-goal').value.trim(),
    mode: $('#agent-mode').value || 'approve',
    max_iters: parseInt($('#agent-iters').value, 10) || 6,
    timeout: parseInt($('#agent-timeout').value, 10) || 30,
  };
  if (!payload.model) return toast('Pick a base model');
  if (!payload.goal) return toast('Set a goal');
  agentEls.pane.textContent = '';
  try {
    await api('/api/agent/run', { method: 'POST', body: JSON.stringify(payload) });
    toast('Agent started');
  } catch (e) {
    toast(e.message);
  }
});

$('#btn-agent-approve').addEventListener('click', async () => {
  await api('/api/agent/approve', { method: 'POST', body: JSON.stringify({ approve: true }) });
});
$('#btn-agent-deny').addEventListener('click', async () => {
  await api('/api/agent/approve', { method: 'POST', body: JSON.stringify({ approve: false }) });
});
$('#btn-agent-cancel').addEventListener('click', async () => {
  await api('/api/agent/approve', { method: 'POST', body: JSON.stringify({ approve: false }) });
  toast('Sent deny — agent will exit on next tool call');
});

// Populate agent dropdowns whenever state refreshes.
const _origRender = refreshState;
refreshState = async function() {  // eslint-disable-line no-func-assign
  await _origRender();
  try {
    const s = await api('/api/state');
    renderAgentChoices(s);
  } catch {}
};

// ---- bootstrap ----

refreshState();
openLogStream();
openAgentStream();
setInterval(refreshState, 5000);

// ════════════════════════════════════════════════════════════
// LM Studio tab
// ════════════════════════════════════════════════════════════

const lms = {
  serverChip:   document.getElementById('lms-server-chip'),
  notInstalled: document.getElementById('lms-not-installed'),
  loadedLabel:  document.getElementById('lms-loaded-label'),
  modelSelect:  document.getElementById('lms-model-select'),
  vramBars:     document.getElementById('lms-vram-bars'),
  dlLog:        document.getElementById('lms-dl-log'),
};

function lmsSetServerChip(running) {
  lms.serverChip.classList.remove('running', 'error', 'done');
  if (running === null) {
    lms.serverChip.textContent = 'checking…';
  } else if (running) {
    lms.serverChip.textContent = 'running';
    lms.serverChip.classList.add('done');
  } else {
    lms.serverChip.textContent = 'stopped';
    lms.serverChip.classList.add('error');
  }
}

function lmsRenderModels(models) {
  lms.modelSelect.innerHTML = '';
  if (!models || !models.length) {
    const o = document.createElement('md-select-option');
    o.value = '';
    const h = document.createElement('div'); h.slot = 'headline';
    h.textContent = '— no GGUF models found —';
    o.appendChild(h);
    lms.modelSelect.appendChild(o);
    return;
  }
  for (const m of models) {
    const o = document.createElement('md-select-option');
    o.value = m.rel_path;
    const h = document.createElement('div'); h.slot = 'headline';
    h.textContent = `${m.name}  (${m.size_gb} GB)`;
    o.appendChild(h);
    lms.modelSelect.appendChild(o);
  }
}

function lmsRenderVram(gpus) {
  if (!gpus || !gpus.length) {
    lms.vramBars.innerHTML = '<div class="empty">No NVIDIA GPU detected or nvidia-smi not available.</div>';
    return;
  }
  lms.vramBars.innerHTML = '';
  for (const g of gpus) {
    const usedGb  = (g.used_mb  / 1024).toFixed(1);
    const totalGb = (g.total_mb / 1024).toFixed(1);
    const pct = g.pct;
    const fillClass = pct >= 90 ? 'crit' : pct >= 70 ? 'warn' : '';

    const row = document.createElement('div');
    row.className = 'vram-row';
    row.innerHTML = `
      <div class="vram-label">
        <span><strong>GPU ${g.index}</strong> · ${g.name}</span>
        <span>${usedGb} / ${totalGb} GB &nbsp;(${pct}%)</span>
      </div>
      <div class="vram-track">
        <div class="vram-fill ${fillClass}" style="width:${pct}%"></div>
      </div>`;
    lms.vramBars.appendChild(row);
  }
}

async function lmsRefresh() {
  try {
    const s = await api('/api/lms/status');
    lms.notInstalled.hidden = s.installed;
    lmsSetServerChip(s.server_running);

    const isLlama = s.backend === 'llama-server';
    const loadedText = s.loaded_model
      ? `Loaded: ${s.loaded_model}`
      : 'No model loaded.';
    lms.loadedLabel.textContent = isLlama
      ? `${loadedText}  ·  backend: llama-server (pinned)`
      : loadedText;

    // When backend is llama-server the model is pinned by the systemd
    // unit's Environment=MODEL=... Swapping it works via a drop-in
    // override written by a root helper — so Load is allowed; we just
    // relabel the buttons so the user understands the model swap means
    // a service restart, not a hot-reload.
    const loadBtn   = document.getElementById('btn-lms-load');
    const unloadBtn = document.getElementById('btn-lms-unload');
    if (loadBtn) {
      loadBtn.disabled = false;
      loadBtn.textContent = isLlama ? 'Swap model (restart server)' : 'Load';
      loadBtn.title = isLlama
        ? 'Rewrites llama-server.service drop-in Environment=MODEL=<path> and restarts the unit.'
        : '';
    }
    if (unloadBtn) {
      unloadBtn.textContent = isLlama ? 'Stop server (free VRAM)' : 'Unload';
      unloadBtn.title = isLlama
        ? 'Stops llama-server.service; frees VRAM. Start Server reloads it.'
        : '';
    }

    lmsRenderModels(s.models);
    lmsRenderVram(s.vram);
  } catch (e) {
    lmsSetServerChip(null);
  }
}

async function lmsRefreshVram() {
  try {
    const gpus = await api('/api/lms/vram');
    lmsRenderVram(gpus);
  } catch {}
}

// Live auto-refresh: while the LM Studio tab is visible, poll the full
// status every 2.5 s so loaded model, backend, server state, and the VRAM
// chart all move together. When the tab isn't active we skip the call so
// we don't burn the network / GPU query bandwidth for a view nobody's
// looking at. A faster 1 s VRAM-only poll runs on top of that for the
// smoothest chart motion during active inference.
let _lmsPollInflight = false;
async function _lmsTick(fullStatus) {
  if (_lmsPollInflight) return;
  const active = document.querySelector('.tab-panel.active');
  if (!active || active.dataset.tab !== 'lmstudio') return;
  _lmsPollInflight = true;
  try {
    if (fullStatus) await lmsRefresh();
    else           await lmsRefreshVram();
  } finally {
    _lmsPollInflight = false;
  }
}
setInterval(() => _lmsTick(true),  2500);   // full status (covers VRAM)
setInterval(() => _lmsTick(false), 1000);   // VRAM-only between full ticks

// Also refresh immediately whenever the tab becomes visible again after the
// user had another app in focus — no waiting for the next interval tick.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') _lmsTick(true);
});

document.getElementById('btn-lms-start').addEventListener('click', async () => {
  const port       = parseInt(document.getElementById('lms-port').value, 10) || 1234;
  const gpu_layers = parseInt(document.getElementById('lms-gpu-layers').value, 10);
  const context_length = parseInt(document.getElementById('lms-ctx').value, 10) || 4096;
  lmsSetServerChip(null);
  try {
    await api('/api/lms/server/start', {
      method: 'POST',
      body: JSON.stringify({ port, gpu_layers, context_length }),
    });
    toast('LM Studio server starting…');
    setTimeout(lmsRefresh, 2000);
  } catch (e) { toast(e.message); lmsRefresh(); }
});

document.getElementById('btn-lms-stop').addEventListener('click', async () => {
  try {
    await api('/api/lms/server/stop', { method: 'POST' });
    toast('Server stopped.');
    lmsRefresh();
  } catch (e) { toast(e.message); }
});

document.getElementById('btn-lms-load').addEventListener('click', async () => {
  const rel_path = lms.modelSelect.value;
  if (!rel_path) return toast('Select a model first.');
  const gpu_layers = parseInt(document.getElementById('lms-gpu-layers').value, 10);
  const context_length = parseInt(document.getElementById('lms-ctx').value, 10) || 4096;
  try {
    const r = await api('/api/lms/models/load', {
      method: 'POST',
      body: JSON.stringify({ rel_path, gpu_layers, context_length }),
    });
    if (r && r.backend === 'llama-server') {
      toast(`Restarting llama-server with ${rel_path}…`);
      // Unit restart + model load takes a few seconds; give it time to
      // come back up before we refresh (otherwise the status says 'none'
      // briefly and the UI flickers).
      setTimeout(lmsRefresh, 6000);
    } else {
      toast(`Loading ${rel_path}…`);
      setTimeout(lmsRefresh, 2000);
    }
  } catch (e) { toast(e.message); }
});

document.getElementById('btn-lms-unload').addEventListener('click', async () => {
  try {
    const r = await api('/api/lms/models/unload', { method: 'POST' });
    // Different backends report different outcomes; reflect what actually
    // happened instead of the previous hardcoded "Model unloaded" that made
    // a silent no-op look successful.
    if (r && r.backend === 'llama-server') {
      toast('llama-server stopped — VRAM freed.');
    } else if (r && r.still_loaded) {
      toast(`Unload reported ok, but model still listed: ${r.still_loaded}`);
    } else {
      toast('Model unloaded.');
    }
    lmsRefresh();
  } catch (e) { toast(e.message); }
});

document.getElementById('btn-lms-download').addEventListener('click', async () => {
  const repo_id = document.getElementById('lms-dl-repo').value.trim();
  const filename = document.getElementById('lms-dl-file').value.trim();
  const token   = document.getElementById('lms-dl-token').value.trim();
  if (!repo_id) return toast('Enter a HF repo id.');

  lms.dlLog.style.display = 'block';
  lms.dlLog.textContent = '';

  const es = new EventSource(
    `/api/lms/models/download?` +
    new URLSearchParams({ repo_id, filename, token }).toString()
  );

  // Can't POST via EventSource; use fetch + ReadableStream instead
  es.close();
  lms.dlLog.textContent = 'Starting download…\n';

  try {
    const res = await fetch('/api/lms/models/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id, filename, token }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data:')) {
          try {
            const ev = JSON.parse(line.slice(5).trim());
            if (ev.data) {
              lms.dlLog.textContent += ev.data + '\n';
              lms.dlLog.scrollTop = lms.dlLog.scrollHeight;
            }
            if (ev.event === 'done') { toast('Download complete.'); lmsRefresh(); }
            if (ev.event === 'error') toast(`Download error: ${ev.data}`);
          } catch {}
        }
      }
    }
  } catch (e) {
    lms.dlLog.textContent += `\nError: ${e.message}`;
    toast(e.message);
  }
});
