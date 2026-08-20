import { MODELS, getModel } from './model-data.js';
import { calculateCache, calculateCommunication, formatBytes } from './calculators.js';

const $ = (selector, root = document) => root?.querySelector(selector) ?? null;
const $$ = (selector, root = document) => root ? [...root.querySelectorAll(selector)] : [];
const enableControls = (root) => $$('button, input, select', root).forEach((control) => { control.disabled = false; });

const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* Storage is optional. */ } },
};

function setupTheme() {
  const button = $('#theme-toggle');
  if (!button) return;
  const saved = storage.get('transformer-course-theme');
  const systemLight = typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: light)').matches;
  const initial = saved === 'light' || saved === 'dark' ? saved : systemLight ? 'light' : 'dark';
  const apply = (theme) => {
    document.documentElement.dataset.theme = theme;
    button.textContent = `主题：${theme === 'dark' ? '暗色' : '亮色'}（点击切换）`;
  };
  apply(initial);
  button.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    apply(next);
    storage.set('transformer-course-theme', next);
  });
  enableControls(button.parentElement);
}

function setupNavigation() {
  const sections = [...$$('[data-lesson]'), $('#models'), $('#systems'), $('#comparison'), $('#sources')].filter(Boolean);
  const nav = $('.site-header nav');
  const links = $$('a', nav);
  const progress = $('#course-progress');
  if (!sections.length || !nav || !links.length || !progress) return;

  const navigationTarget = (id) => {
    if (id.startsWith('system-')) return 'systems';
    if (id === 'comparison' || id === 'sources' || id.startsWith('model-') || id.startsWith('source-')) return 'models';
    return id;
  };
  const setCurrent = (id) => {
    let activeLink = null;
    links.forEach((link) => {
      if (link.hash === `#${navigationTarget(id)}`) {
        link.setAttribute('aria-current', 'location');
        activeLink = link;
      } else link.removeAttribute('aria-current');
    });
    if (activeLink && nav.scrollWidth > nav.clientWidth) {
      nav.scrollTo({ left: Math.max(0, activeLink.offsetLeft - (nav.clientWidth - activeLink.offsetWidth) / 2), behavior: 'auto' });
    }
  };
  const alignHashTarget = () => {
    const target = document.getElementById(location.hash.slice(1));
    if (target) target.scrollIntoView({ block: 'start', behavior: 'auto' });
  };
  const syncHash = () => {
    const id = location.hash.slice(1);
    if (id && links.some((link) => link.hash === `#${navigationTarget(id)}`)) setCurrent(id);
    if (id) requestAnimationFrame(alignHashTarget);
  };
  syncHash();
  addEventListener('load', alignHashTarget, { once: true });
  addEventListener('hashchange', syncHash);

  let scheduled = false;
  const updateProgress = () => {
    scheduled = false;
    const root = document.documentElement;
    const distance = Math.max(1, root.scrollHeight - innerHeight);
    const value = Math.round(Math.min(100, Math.max(0, scrollY / distance * 100)));
    progress.value = value;
    progress.textContent = `${value}%`;
  };
  const scheduleProgress = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(updateProgress);
  };
  addEventListener('scroll', scheduleProgress, { passive: true });
  addEventListener('resize', scheduleProgress, { passive: true });
  scheduleProgress();

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible) setCurrent(visible.target.id);
    }, { rootMargin: '-25% 0px -60%', threshold: [0, 0.25, 0.6] });
    sections.forEach((section) => observer.observe(section));
  }
}

const FLOW_DESCRIPTIONS = [
  'Token ids：只有整数索引，shape 是 (B,T)，还没有连续语义向量。',
  'Embedding：新增 D 维表示；位置方法可能是绝对 embedding 或在 attention 内注入 RoPE。',
  'Attention：保持 (B,T,D)，沿 token 维交换信息；内部会拆成多个 head。',
  'FFN：保持 (B,T,D)，内部先升维并做非线性，再投影回 D。',
  'Logits：把最后一维 D 映射到词表 V；训练标签的 shift 可以在数据层完成。',
];

function setupDataflow() {
  const lab = $('#dataflow-lab');
  const steps = $$('[data-flow-step]', lab);
  const next = $('#flow-next', lab);
  const description = $('#flow-description', lab);
  if (!lab || steps.length !== FLOW_DESCRIPTIONS.length || !next || !description) return;
  let active = 0;
  const select = (index) => {
    active = index;
    steps.forEach((step, stepIndex) => {
      const selected = stepIndex === active;
      step.classList.toggle('active', selected);
      step.setAttribute('aria-pressed', String(selected));
    });
    description.textContent = FLOW_DESCRIPTIONS[active];
  };
  next.addEventListener('click', () => select((active + 1) % steps.length));
  steps.forEach((step, index) => step.addEventListener('click', () => select(index)));
  select(0);
  enableControls(lab);
}

const TOPOLOGIES = Object.freeze({
  mha: { memory: 'K₁ V₁ · K₂ V₂ · K₃ V₃', description: 'MHA：每个 Q 头有独立 K/V 头，表达自由度高，KV cache 最大。' },
  gqa: { memory: 'KV 组 A · KV 组 B', description: 'GQA：一组 Q 头共享一组 K/V；保留多个 KV 子空间，同时减少 cache。' },
  mqa: { memory: '共享 K · 共享 V', description: 'MQA：全部 Q 头共享单组 K/V；cache 最小，但 KV 表达共享最多。' },
  linear: { memory: '递归状态 Sₜ', description: '线性注意力：把历史压入固定或有限状态；保留什么取决于更新规则。' },
  sparse: { memory: '窗口 + 选中历史', description: '稀疏注意力：只计算窗口或 top-k 位置；cache 是否压缩必须查看写入路径。' },
});

function setupTopology() {
  const lab = $('#topology-lab');
  const memory = $('#topology-memory', lab);
  const description = $('#topology-description', lab);
  const inputs = $$('input[name="topology"]', lab);
  if (!lab || !memory || !description || inputs.length !== Object.keys(TOPOLOGIES).length) return;
  inputs.forEach((input) => input.addEventListener('change', () => {
    if (!input.checked || !TOPOLOGIES[input.value]) return;
    memory.textContent = TOPOLOGIES[input.value].memory;
    description.textContent = TOPOLOGIES[input.value].description;
  }));
  enableControls(lab);
}

const scopeLabel = (scope) => ({ reference: '[reference shape + derived bytes]', theoretical: '[optimized / theoretical]', 'deployment-assumption': '[deployment]' })[scope] ?? scope;

function setupCacheCalculator() {
  const form = $('#cache-calculator');
  const output = $('#cache-result', form);
  const modelInput = $('#cache-model', form);
  const tokenInput = $('#cache-tokens', form);
  const batchInput = $('#cache-batch', form);
  const dtypeInput = $('#cache-dtype', form);
  const modeInput = $('#cache-mode', form);
  if (!form || !output || !modelInput || !tokenInput || !batchInput || !dtypeInput || !modeInput) return;

  const syncDtypes = () => {
    const deployment = modelInput.value === 'deepseek' && modeInput.value === 'deployment';
    const deepReference = modelInput.value === 'deepseek' && modeInput.value === 'reference';
    [...dtypeInput.options].forEach((option) => {
      option.disabled = deployment ? option.value !== 'mixed' : deepReference ? option.value !== 'bf16' : option.value === 'mixed';
    });
    if (deployment) dtypeInput.value = 'mixed';
    else if (dtypeInput.value === 'mixed' || deepReference) dtypeInput.value = 'bf16';
  };
  const syncModes = () => {
    const supportedModes = getModel(modelInput.value).supportedModes;
    [...modeInput.options].forEach((option) => { option.disabled = !supportedModes.includes(option.value); });
    if (!supportedModes.includes(modeInput.value)) modeInput.value = supportedModes[0];
    syncDtypes();
  };
  modelInput.addEventListener('change', syncModes);
  modeInput.addEventListener('change', syncDtypes);
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    try {
      const result = calculateCache({ modelId: modelInput.value, tokens: Number(tokenInput.value), batch: Number(batchInput.value), dtype: dtypeInput.value, mode: modeInput.value });
      const title = result.complete ? '总量' : '已计入小计';
      const lines = [
        `${result.modelName} · ${scopeLabel(result.scope)} · B=${result.batch}`,
        `${title}：${formatBytes(result.quantifiedBytes, 'GB')} / ${formatBytes(result.quantifiedBytes, 'GiB')}`,
        ...result.components.map((part) => `• ${part.label}: ${formatBytes(part.bytes, 'GB')} (${part.dtype.toUpperCase()})\n  ${part.formula} · shape[${part.shapeEvidence}] bytes[${part.byteEvidence}]`),
      ];
      if (!result.complete) lines.push('未计项：', ...result.unquantifiedComponents.map((item) => `• ${item}`));
      lines.push(`配置范围：${result.withinConfiguredContext ? '在配置上限内' : `超出 max=${result.maxConfiguredContext.toLocaleString('en-US')}`}`, `Decode：${result.decodeComplexity}`);
      if (result.note) lines.push(`说明：${result.note}`);
      output.textContent = lines.join('\n');
    } catch (error) {
      output.textContent = `无法计算：${error instanceof Error ? error.message : '未知错误'}`;
    }
  });
  syncModes();
  enableControls(form);
}

function setupCommunicationCalculator() {
  const form = $('#communication-calculator');
  const output = $('#communication-result', form);
  const ids = ['comm-batch', 'comm-tokens', 'comm-hidden', 'comm-topk', 'comm-dtype', 'comm-bandwidth', 'comm-unit', 'comm-efficiency', 'comm-remote-fraction'];
  const controls = Object.fromEntries(ids.map((id) => [id, $(`#${id}`, form)]));
  if (!form || !output || Object.values(controls).some((control) => !control)) return;
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    try {
      const result = calculateCommunication({
        batch: Number(controls['comm-batch'].value), tokens: Number(controls['comm-tokens'].value), hiddenSize: Number(controls['comm-hidden'].value),
        topK: Number(controls['comm-topk'].value), dtype: controls['comm-dtype'].value, bandwidth: Number(controls['comm-bandwidth'].value),
        bandwidthUnit: controls['comm-unit'].value, efficiency: Number(controls['comm-efficiency'].value), remoteRouteFraction: Number(controls['comm-remote-fraction'].value),
      });
      output.textContent = [
        '[推导] 单向远端 dispatch 理论下限',
        `top-k routed payload：${formatBytes(result.routedPayloadBytes, 'GB')} / ${formatBytes(result.routedPayloadBytes, 'GiB')}`,
        `按远端比例后的 payload：${formatBytes(result.remoteBytes, 'GB')} / ${formatBytes(result.remoteBytes, 'GiB')}`,
        `有效带宽：${(result.effectiveBytesPerSecond / 1e9).toFixed(3)} GB/s`,
        `时间下限：${(result.seconds * 1000).toFixed(6)} ms`,
        `假设：top-k=${result.assumptions.activationCopies}，remote-route fraction=${result.assumptions.remoteRouteFraction}；未含 combine、协议、拥塞与负载不均。`,
        'FP4 专家权重不会自动减少 BF16/FP8 激活通信。',
      ].join('\n');
    } catch (error) {
      output.textContent = `无法计算：${error instanceof Error ? error.message : '未知错误'}`;
    }
  });
  enableControls(form);
}

function setupChecks() {
  $$('.knowledge-check').forEach((form, index) => {
    const output = $('output', form);
    const controls = $$('input, button', form);
    if (!output || !controls.length) return;
    const completion = document.createElement('span');
    completion.className = 'completion-state';
    completion.setAttribute('aria-live', 'polite');
    form.append(completion);
    const key = `transformer-course-check-${index + 1}`;
    const markComplete = () => {
      form.classList.add('complete');
      completion.textContent = '✓ 已完成';
    };
    if (storage.get(key) === 'done') markComplete();
    else completion.textContent = '未完成';
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const selected = $('input:checked', form);
      if (!selected) { output.textContent = '请先选择一个答案。'; return; }
      const correct = selected.value === form.dataset.answer;
      output.textContent = `${correct ? '正确。' : '再想一步。'}${form.dataset.feedback}`;
      if (correct) { markComplete(); storage.set(key, 'done'); }
    });
    enableControls(form);
  });
}

function verifyDataBinding() {
  if (!document.documentElement || MODELS.length !== 6) return;
  document.documentElement.dataset.modelCount = String(MODELS.length);
  document.documentElement.dataset.qwenFamily = getModel('qwen').implementationFamily;
}

[setupTheme, setupNavigation, setupDataflow, setupTopology, setupCacheCalculator, setupCommunicationCalculator, setupChecks, verifyDataBinding].forEach((setup) => {
  try { setup(); } catch (error) { console.error(`Enhancement failed in ${setup.name}`, error); }
});
