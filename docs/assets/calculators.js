import { getModel } from './model-data.js';

export const DTYPE_BYTES = Object.freeze({ fp32: 4, bf16: 2, fp16: 2, fp8: 1, fp4: 0.5 });
const UNITS = Object.freeze({ B: 1, KB: 1e3, MB: 1e6, GB: 1e9, TB: 1e12, KiB: 2 ** 10, MiB: 2 ** 20, GiB: 2 ** 30, TiB: 2 ** 40 });
const MAX = Number.MAX_SAFE_INTEGER;

const positiveInteger = (value, label) => {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number <= 0) throw new RangeError(`${label} must be a positive safe integer`);
  return number;
};

const positiveFinite = (value, label) => {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) throw new RangeError(`${label} must be a positive finite number`);
  return number;
};

const fraction = (value, label) => {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0 || number > 1) throw new RangeError(`${label} must be in [0, 1]`);
  return number;
};

const safeProduct = (label, ...values) => {
  let result = 1;
  for (const value of values) {
    if (!Number.isFinite(value) || value < 0 || result > MAX / Math.max(value, 1)) throw new RangeError(`${label} exceeds safe numeric range`);
    result *= value;
  }
  if (!Number.isFinite(result) || result > MAX) throw new RangeError(`${label} exceeds safe numeric range`);
  return result;
};

const safeSum = (label, values) => {
  let result = 0;
  for (const value of values) {
    if (!Number.isFinite(value) || value < 0 || result > MAX - value) throw new RangeError(`${label} exceeds safe numeric range`);
    result += value;
  }
  return result;
};

const dtypeSize = (dtype) => {
  const size = DTYPE_BYTES[dtype];
  if (!size) throw new RangeError(`Unsupported dtype: ${dtype}`);
  return size;
};

const component = (id, label, bytes, dtype, formula, { shapeEvidence = 'source', byteEvidence = 'derived' } = {}) => {
  if (!Number.isFinite(bytes) || bytes < 0 || bytes > MAX) throw new RangeError(`${id} bytes exceed safe numeric range`);
  return Object.freeze({ id, label, bytes, dtype, formula, shapeEvidence, byteEvidence });
};

export function convertBytes(bytes, unit = 'GB') {
  const number = Number(bytes);
  if (!Number.isFinite(number) || number < 0 || number > MAX) throw new RangeError('bytes must be finite and within safe numeric range');
  if (!(unit in UNITS)) throw new RangeError(`Unsupported unit: ${unit}`);
  return number / UNITS[unit];
}

export function formatBytes(bytes, unit = 'GB') {
  return `${convertBytes(bytes, unit).toFixed(3)} ${unit}`;
}

const cacheBytes = (label, dimensions, bytesPerElement) => safeProduct(label, ...dimensions, bytesPerElement);

function deepseekCompressorState(batch, counts, headDim) {
  const bytesFor = (layers, ratio, dim) => {
    const coefficient = ratio === 4 ? 2 : 1;
    return cacheBytes('DeepSeek compressor state', [batch, layers, 2, coefficient, ratio, coefficient, dim], DTYPE_BYTES.fp32);
  };
  return safeSum('DeepSeek compressor states', [bytesFor(counts[4], 4, headDim), bytesFor(counts[128], 128, headDim)]);
}

export function calculateCache({ modelId, tokens, batch = 1, dtype = 'bf16', mode = 'reference' }) {
  const model = getModel(modelId);
  const length = positiveInteger(tokens, 'tokens');
  const batches = positiveInteger(batch, 'batch');
  if (!model.supportedModes.includes(mode)) throw new RangeError(`Unsupported mode '${mode}' for ${model.id}`);
  if (model.id === 'deepseek' && mode === 'reference' && dtype !== 'bf16') throw new RangeError('DeepSeek reference layout is BF16');
  if (model.id === 'deepseek' && mode === 'deployment' && dtype !== 'mixed') throw new RangeError("DeepSeek deployment mode requires dtype='mixed' (FP8 core + raw FP4 indexer)");
  const bytesPerElement = model.id === 'deepseek' && mode === 'deployment' ? null : dtypeSize(dtype);
  const parts = [];
  const unquantified = [];
  let complete = true;
  let scope = mode === 'reference' ? 'reference' : mode === 'optimized' ? 'theoretical' : 'deployment-assumption';
  let note = '';
  let allocationModel = 'logical-used-tokens';

  if (model.id === 'nanogpt') {
    if (mode === 'reference') parts.push(component('no-cache', '当前实现不提供 KV cache', 0, dtype, 'B × 0 bytes'));
    else {
      const spec = model.cache.theoretical;
      parts.push(component('theoretical-kv', '假设加入标准 MHA cache', cacheBytes('nanoGPT theoretical KV', [batches, spec.layers, length, spec.kvHeads, 2, spec.headDim], bytesPerElement), dtype, 'B × L × T × Hkv × (Dk + Dv) × bytes'));
    }
  } else if (model.id === 'minimind') {
    const cache = model.cache;
    parts.push(component('attention-core', '展开 GQA K/V', cacheBytes('MiniMind KV', [batches, cache.layers, length, cache.kvHeads, cache.keyDim + cache.valueDim], bytesPerElement), dtype, 'B × L × T × Hkv × (Dk + Dv) × bytes'));
  } else if (model.id === 'qwen') {
    const { fullAttention: full, recurrent, convolution } = model.cache;
    parts.push(component('full-attention-kv', '16 层 GQA K/V', cacheBytes('Qwen full KV', [batches, full.layers, length, full.kvHeads, full.keyDim + full.valueDim], bytesPerElement), dtype, 'B × Lfull × T × Hkv × (Dk + Dv) × bytes'));
    parts.push(component('linear-recurrent-state', 'fallback Gated DeltaNet recurrent state', cacheBytes('Qwen recurrent state', [batches, recurrent.layers, recurrent.heads, recurrent.keyDim, recurrent.valueDim], DTYPE_BYTES.fp32), 'fp32', 'B × Llinear × Hv × Dk × Dv × 4 bytes'));
    parts.push(component('linear-convolution-state', '48 层 combined QKV convolution state', cacheBytes('Qwen convolution state', [batches, convolution.layers, convolution.combinedQkvDim, convolution.kernel], bytesPerElement), dtype, 'B × Llinear × combined_QKV_dim × kernel × bytes'));
  } else if (model.id === 'glm') {
    const cache = model.cache;
    if (mode === 'optimized') parts.push(component('attention-core', 'MLA latent cache', cacheBytes('GLM latent cache', [batches, cache.layers, length, cache.optimized.latentDim], bytesPerElement), dtype, 'B × L × T × 576 × bytes'));
    else {
      const expanded = cache.reference;
      parts.push(component('attention-core', '当前 HF/reference 展开 K/V', cacheBytes('GLM expanded KV', [batches, cache.layers, length, expanded.heads, expanded.keyDim + expanded.valueDim], bytesPerElement), dtype, 'B × L × T × H × (Dk + Dv) × bytes'));
    }
    parts.push(component('indexer-cache', '21 层实例化 Indexer key cache', cacheBytes('GLM indexer cache', [batches, cache.indexer.layers, length, cache.indexer.dim], bytesPerElement), dtype, 'B × Lindexer × T × Dindex × bytes'));
  } else if (model.id === 'kimi') {
    const cache = model.cache;
    if (mode === 'optimized') {
      const storedDim = cache.optimized.latentDim + cache.optimized.keyRemainderDim;
      parts.push(component('latent-cache', 'MLA 512 latent + 64 key remainder', cacheBytes('Kimi latent cache', [batches, cache.fullAttention.layers, length, storedDim], bytesPerElement), dtype, 'B × Lfull × T × (512 + 64) × bytes'));
    } else {
      const full = cache.fullAttention;
      parts.push(component('full-attention-kv', '当前 HF/reference 展开 K/V', cacheBytes('Kimi expanded KV', [batches, full.layers, length, full.heads, full.keyDim + full.valueDim], bytesPerElement), dtype, 'B × Lfull × T × H × (Dk + Dv) × bytes'));
    }
    complete = false;
    unquantified.push(...cache.unquantified);
  } else if (model.id === 'deepseek') {
    const cache = model.cache;
    const counts = cache.compressRatioCounts;
    const ratio4Slots = Math.floor(length / 4);
    const ratio128Slots = Math.floor(length / 128);
    const allocatedSlots = safeSum('DeepSeek allocated slots', [counts[0] * cache.window, counts[4] * (cache.window + ratio4Slots), counts[128] * (cache.window + ratio128Slots)]);
    allocationModel = 'preallocated-max-seq-len';
    note = 'reference 源码 buffer 按运行时 max_seq_len/max_batch_size 预分配；输入 B/T 在此代表该次分配配置，不是当前已用 token 的动态增长值。';
    const coreDtype = mode === 'reference' ? 'bf16' : 'fp8';
    const coreEvidence = mode === 'reference' ? 'derived' : 'deployment';
    parts.push(component('attention-core', '43 层 attention core 预分配 buffer', cacheBytes('DeepSeek attention core', [batches, allocatedSlots, cache.headDim], dtypeSize(coreDtype)), coreDtype, 'B × [Σ Lr × (window + floor(T/r))] × D × bytes', { byteEvidence: coreEvidence }));
    const indexDtype = mode === 'reference' ? 'bf16' : 'fp4-raw';
    const indexBytes = mode === 'reference' ? DTYPE_BYTES.bf16 : DTYPE_BYTES.fp4;
    const indexEvidence = mode === 'reference' ? 'derived' : 'deployment';
    parts.push(component('indexer-cache', mode === 'reference' ? 'ratio=4 的 Indexer 预分配 cache' : 'Indexer raw FP4 payload 下限', cacheBytes('DeepSeek indexer cache', [batches, cache.indexer.layers, ratio4Slots, cache.indexer.dim], indexBytes), indexDtype, 'B × Lindex × floor(T/4) × Dindex × bytes', { byteEvidence: indexEvidence }));
    parts.push(component('attention-compressor-fp32-state', 'Attention Compressor kv_state + score_state', deepseekCompressorState(batches, counts, cache.headDim), 'fp32', 'B × layers × 2 buffers × state_shape × 4 bytes'));
    const indexCoefficient = 2;
    const indexState = cacheBytes('DeepSeek indexer compressor state', [batches, cache.indexer.layers, 2, indexCoefficient, cache.indexer.ratio, indexCoefficient, cache.indexer.dim], DTYPE_BYTES.fp32);
    parts.push(component('indexer-compressor-fp32-state', 'Indexer Compressor kv_state + score_state', indexState, 'fp32', 'B × Lindex × 2 buffers × (2r) × (2Dindex) × 4 bytes'));
    if (mode === 'deployment') {
      complete = false;
      unquantified.push('FP8 attention core 与 FP4 Indexer cache 的 scale/metadata、packing/alignment 开销未计；raw payload 仅为下限');
      note += ' 部署组合依据官方模型卡的 --kv-cache-dtype fp8 与 use_fp4_indexer_cache:true；量化组件只给 raw payload 下限。';
    }
  }

  if (model.id !== 'deepseek' && ['fp8', 'fp4'].includes(dtype) && parts.some((entry) => entry.bytes > 0)) {
    complete = false;
    unquantified.push(`${dtype.toUpperCase()} storage 的 scale/metadata、packing/alignment 开销未计`);
    note = `${note}${note ? ' ' : ''}所选 ${dtype.toUpperCase()} 仅是 raw payload 存储假设，不证明当前 reference runtime 支持该量化布局。`;
  }

  const quantifiedBytes = safeSum('cache quantified subtotal', parts.map((entry) => entry.bytes));
  return Object.freeze({
    modelId, modelName: model.name, tokens: length, batch: batches, dtype, mode, scope,
    quantifiedBytes, totalBytes: complete ? quantifiedBytes : undefined, complete,
    components: Object.freeze(parts), unquantifiedComponents: Object.freeze(unquantified),
    withinConfiguredContext: length <= model.specs.maxContext, maxConfiguredContext: model.specs.maxContext,
    decodeComplexity: model.cache.decodeComplexity, allocationModel, note,
  });
}

export function calculateCommunication({ batch = 1, tokens = 1, hiddenSize, topK = 1, dtype = 'bf16', bandwidth, bandwidthUnit = 'Gb/s', efficiency = 1, remoteRouteFraction = 1 }) {
  const batches = positiveInteger(batch, 'batch');
  const tokenCount = positiveInteger(tokens, 'tokens');
  const hidden = positiveInteger(hiddenSize, 'hiddenSize');
  const copies = positiveInteger(topK, 'topK');
  const link = positiveFinite(bandwidth, 'bandwidth');
  const utilization = fraction(efficiency, 'efficiency');
  if (utilization === 0) throw new RangeError('efficiency must be in (0, 1]');
  const remoteFraction = fraction(remoteRouteFraction, 'remoteRouteFraction');
  if (!['Gb/s', 'GB/s'].includes(bandwidthUnit)) throw new RangeError(`Unsupported bandwidth unit: ${bandwidthUnit}`);

  const routedPayloadBytes = cacheBytes('routed activation payload', [batches, tokenCount, hidden, copies], dtypeSize(dtype));
  const remoteBytes = safeProduct('remote routed payload', routedPayloadBytes, remoteFraction);
  const rawBytesPerSecond = link * 1e9 / (bandwidthUnit === 'Gb/s' ? 8 : 1);
  const effectiveBytesPerSecond = rawBytesPerSecond * utilization;
  if (!Number.isFinite(effectiveBytesPerSecond) || effectiveBytesPerSecond <= 0) throw new RangeError('effective bandwidth is outside numeric range');
  return Object.freeze({
    bytes: remoteBytes, routedPayloadBytes, remoteBytes,
    decimalGB: convertBytes(remoteBytes, 'GB'), binaryGiB: convertBytes(remoteBytes, 'GiB'),
    rawBytesPerSecond, effectiveBytesPerSecond, seconds: remoteBytes / effectiveBytesPerSecond,
    theoreticalLowerBound: true,
    assumptions: Object.freeze({ activationCopies: copies, remoteRouteFraction: remoteFraction, oneWayDispatch: true, protocolOverheadIncluded: false, contentionIncluded: false }),
    note: '条件性远端 payload = top-k routed activation × remote-route fraction；FP4 expert weights do not automatically reduce activation communication.',
  });
}
