// Dashboard frontend logic. Talks to FastAPI backend.

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

$('#btn-launch-lmstudio').addEventListener('click', async () => {
  try {
    await api('/api/launch-lmstudio', { method: 'POST' });
    toast('LM Studio launched');
  } catch (e) {
    toast(e.message);
  }
});

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
