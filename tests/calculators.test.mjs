import test from 'node:test';
import assert from 'node:assert/strict';

import { MODELS, EVIDENCE_SNAPSHOT, getModel } from '../docs/assets/model-data.js';
import { calculateCache, calculateCommunication, convertBytes, formatBytes } from '../docs/assets/calculators.js';

const part = (result, id) => result.components.find((entry) => entry.id === id);
const near = (actual, expected, tolerance = 0.01) => {
  assert.ok(Math.abs(actual - expected) / expected <= tolerance, `${actual} not within ${tolerance * 100}% of ${expected}`);
};

test('six models declare exact supported modes and scoped evidence', () => {
  assert.equal(MODELS.length, 6);
  assert.equal(EVIDENCE_SNAPSHOT.verifiedAt, '2026-08-20');
  assert.doesNotMatch(EVIDENCE_SNAPSHOT.statement, /所有|逐文件/);
  const modes = Object.fromEntries(MODELS.map((model) => [model.id, model.supportedModes]));
  assert.deepEqual(modes, {
    nanogpt: ['reference', 'optimized'], minimind: ['reference'], qwen: ['reference'],
    glm: ['reference', 'optimized'], kimi: ['reference', 'optimized'], deepseek: ['reference', 'deployment'],
  });
  for (const model of MODELS) {
    for (const source of model.sources) {
      assert.ok(['official', 'source', 'config', 'deployment'].includes(source.level));
      assert.match(source.url, /^https:\/\//);
      assert.ok(source.revision);
    }
  }
});

test('Qwen and GLM source evidence use byte-matched Transformers commits', () => {
  const qwenSource = getModel('qwen').sources.find((entry) => entry.level === 'source');
  const glmSource = getModel('glm').sources.find((entry) => entry.level === 'source');
  assert.equal(qwenSource.url, 'https://github.com/huggingface/transformers/blob/b9090ae58cdaf9e7a195f38d3b2574dd408acaa6/src/transformers/models/qwen3_5/modeling_qwen3_5.py');
  assert.equal(glmSource.url, 'https://github.com/huggingface/transformers/blob/6b8369fe9f89312dc4b523c4ad449640bd3e68f3/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py');
});

test('unsupported and unknown cache modes are rejected instead of falling through', () => {
  assert.throws(() => calculateCache({ modelId: 'minimind', tokens: 1, mode: 'optimized' }), RangeError);
  assert.throws(() => calculateCache({ modelId: 'qwen', tokens: 1, mode: 'deployment' }), RangeError);
  assert.throws(() => calculateCache({ modelId: 'glm', tokens: 1, mode: 'mystery' }), RangeError);
});

test('count inputs require positive safe integers and unsafe products are rejected', () => {
  for (const value of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1, Infinity, NaN]) {
    assert.throws(() => calculateCache({ modelId: 'minimind', tokens: value }), RangeError);
  }
  assert.throws(() => calculateCache({ modelId: 'kimi', tokens: Number.MAX_SAFE_INTEGER, batch: 1 }), RangeError);
  assert.throws(() => calculateCache({ modelId: 'minimind', tokens: 1, batch: 1.2 }), RangeError);
  assert.throws(() => calculateCommunication({ batch: 1, tokens: 1.5, hiddenSize: 8, topK: 1, bandwidth: 1 }), RangeError);
});

test('batch defaults to one, multiplies cache/state, and formulas include B', () => {
  const one = calculateCache({ modelId: 'minimind', tokens: 32768 });
  const two = calculateCache({ modelId: 'minimind', tokens: 32768, batch: 2 });
  assert.equal(one.batch, 1);
  assert.equal(two.quantifiedBytes, 2 * one.quantifiedBytes);
  assert.match(part(two, 'attention-core').formula, /B ×/);
  assert.equal(one.totalBytes, 402_653_184);
  assert.equal(convertBytes(one.totalBytes, 'MiB'), 384);
});

test('source proves shape/path while byte arithmetic is derived', () => {
  const result = calculateCache({ modelId: 'glm', tokens: 1024, mode: 'reference' });
  assert.equal(part(result, 'attention-core').shapeEvidence, 'source');
  assert.equal(part(result, 'attention-core').byteEvidence, 'derived');
  assert.equal(part(result, 'indexer-cache').byteEvidence, 'derived');
});

test('nanoGPT reference has no cache and states O(T²) attention recomputation', () => {
  const result = calculateCache({ modelId: 'nanogpt', tokens: 1024, mode: 'reference' });
  assert.equal(result.quantifiedBytes, 0);
  assert.match(result.decodeComplexity, /O\(T²\)/);
  assert.match(result.decodeComplexity, /线性层 O\(T\)/);
});

test('Qwen includes full KV, FP32 recurrent, and selected-dtype combined QKV convolution state', () => {
  const result = calculateCache({ modelId: 'qwen', tokens: 1_000_000, batch: 2, dtype: 'bf16', mode: 'reference' });
  assert.equal(part(result, 'full-attention-kv').bytes, 131_072_000_000);
  assert.equal(part(result, 'linear-recurrent-state').bytes, 301_989_888);
  assert.equal(part(result, 'linear-convolution-state').bytes, 7_864_320);
  assert.equal(part(result, 'linear-convolution-state').dtype, 'bf16');
  assert.equal(result.complete, true);
});

test('GLM reference expanded and optimized latent remain separate', () => {
  const reference = calculateCache({ modelId: 'glm', tokens: 1_000_000, mode: 'reference' });
  const optimized = calculateCache({ modelId: 'glm', tokens: 1_000_000, mode: 'optimized' });
  near(part(reference, 'attention-core').bytes, 5_111_808_000_000);
  near(part(optimized, 'attention-core').bytes, 89_856_000_000);
  assert.equal(optimized.scope, 'theoretical');
  assert.equal(getModel('glm').specs.instantiatedIndexers, 21);
});

test('Kimi reference is expanded; optimized MLA stores 512+64 and result remains incomplete', () => {
  const reference = calculateCache({ modelId: 'kimi', tokens: 1_000_000, mode: 'reference' });
  const optimized = calculateCache({ modelId: 'kimi', tokens: 1_000_000, mode: 'optimized' });
  assert.equal(part(reference, 'full-attention-kv').bytes, 1_474_560_000_000);
  assert.equal(part(optimized, 'latent-cache').bytes, 27_648_000_000);
  assert.equal(getModel('kimi').cache.optimized.latentDim, 512);
  assert.equal(getModel('kimi').cache.optimized.keyRemainderDim, 64);
  for (const result of [reference, optimized]) {
    assert.equal(result.complete, false);
    assert.equal(result.totalBytes, undefined);
    assert.match(result.unquantifiedComponents.join(' '), /KDA recurrent/);
    assert.match(result.unquantifiedComponents.join(' '), /conv/i);
  }
});

test('DeepSeek reference models BF16 preallocated buffers with floor and FP32 states', () => {
  const model = getModel('deepseek');
  assert.deepEqual(model.cache.compressRatioCounts, { 0: 2, 4: 21, 128: 20 });
  const result = calculateCache({ modelId: 'deepseek', tokens: 1_000_000, batch: 1, dtype: 'bf16', mode: 'reference' });
  assert.equal(result.allocationModel, 'preallocated-max-seq-len');
  assert.equal(part(result, 'attention-core').bytes, 5_541_625_856);
  assert.equal(part(result, 'indexer-cache').bytes, 1_344_000_000);
  assert.equal(part(result, 'attention-compressor-fp32-state').dtype, 'fp32');
  assert.equal(part(result, 'indexer-compressor-fp32-state').dtype, 'fp32');
  assert.match(result.note, /预分配/);
});

test('DeepSeek deployment is fixed mixed FP8 core + raw FP4 indexer lower bound', () => {
  const result = calculateCache({ modelId: 'deepseek', tokens: 1_000_000, dtype: 'mixed', mode: 'deployment' });
  assert.equal(part(result, 'attention-core').dtype, 'fp8');
  assert.equal(part(result, 'indexer-cache').dtype, 'fp4-raw');
  assert.equal(part(result, 'indexer-cache').byteEvidence, 'deployment');
  assert.equal(part(result, 'indexer-cache').bytes, 336_000_000);
  assert.equal(result.complete, false);
  assert.match(result.unquantifiedComponents.join(' '), /scale|metadata/i);
  assert.throws(() => calculateCache({ modelId: 'deepseek', tokens: 1, dtype: 'fp4', mode: 'deployment' }), RangeError);
});

test('generic quantized dtype results are raw payload subtotals, never complete totals', () => {
  for (const dtype of ['fp8', 'fp4']) {
    for (const modelId of ['minimind', 'qwen', 'glm']) {
      const result = calculateCache({ modelId, tokens: 1024, dtype, mode: 'reference' });
      assert.equal(result.complete, false, `${modelId}/${dtype} must be incomplete`);
      assert.equal(result.totalBytes, undefined);
      assert.match(result.unquantifiedComponents.join(' '), /scale|metadata|packing/i);
      assert.match(result.note, /raw payload|存储假设/i);
    }
  }
});

test('DeepSeek version boundaries distinguish HF config, inference config, and vLLM example', () => {
  const model = getModel('deepseek');
  const specs = model.specs;
  assert.equal(specs.hfNextnPredictLayers, 1);
  assert.equal(specs.inferenceMtpLayers, 3);
  assert.equal(specs.dsparkBlockSize, 5);
  assert.deepEqual(specs.dsparkInjectedMainLayers, [40, 41, 42]);
  assert.equal(specs.officialVllmSpeculativeTokens, 7);
  const deployment = model.sources.find((entry) => entry.level === 'deployment');
  assert.equal(deployment.url, 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md');
});

test('communication applies remote-route fraction and validates all inputs', () => {
  const base = { batch: 1, tokens: 2, hiddenSize: 4096, topK: 6, dtype: 'bf16', efficiency: 0.5, remoteRouteFraction: 0.25 };
  const gbits = calculateCommunication({ ...base, bandwidth: 400, bandwidthUnit: 'Gb/s' });
  const gbytes = calculateCommunication({ ...base, bandwidth: 50, bandwidthUnit: 'GB/s' });
  assert.equal(gbits.routedPayloadBytes, 98_304);
  assert.equal(gbits.remoteBytes, 24_576);
  assert.equal(gbits.bytes, gbits.remoteBytes);
  assert.equal(gbits.seconds, gbytes.seconds);
  assert.equal(gbits.assumptions.remoteRouteFraction, 0.25);
  for (const fraction of [-0.1, 1.1, NaN, Infinity]) {
    assert.throws(() => calculateCommunication({ ...base, bandwidth: 1, remoteRouteFraction: fraction }), RangeError);
  }
});

test('byte formatter reports decimal and binary units without conflation', () => {
  assert.equal(formatBytes(402_653_184, 'MB'), '402.653 MB');
  assert.equal(formatBytes(402_653_184, 'MiB'), '384.000 MiB');
});
