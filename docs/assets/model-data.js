export const EVIDENCE_LEVELS = Object.freeze({
  official: '官方材料', source: '源码事实', config: '配置值', derived: '推导', deployment: '部署假设',
});

export const EVIDENCE_SNAPSHOT = Object.freeze({
  verifiedAt: '2026-08-20',
  statement: '各来源条目分别说明 config、源码路径或部署示例的核验对象；只对明确列出的本地文件与 revision 作匹配声明。推导值不是官方性能实测。',
});

const source = (level, label, url, revision, local) => Object.freeze({ level, label, url, revision, local });
const modes = (...values) => Object.freeze(values);

export const MODELS = Object.freeze([
  Object.freeze({
    id: 'nanogpt', name: 'nanoGPT', role: '最小 GPT 教学坐标系', implementationFamily: 'gpt2',
    supportedModes: modes('reference', 'optimized'),
    specs: Object.freeze({ parameters: 124_000_000, layers: 12, hiddenSize: 768, attentionHeads: 12, kvHeads: 12, headDim: 64, maxContext: 1024, tieWordEmbeddings: true, targetShift: 'data-loader' }),
    cache: Object.freeze({ kind: 'none', theoretical: Object.freeze({ layers: 12, kvHeads: 12, headDim: 64 }), decodeComplexity: '每步重算整个窗口：attention 主项 O(T²)，线性层 O(T)' }),
    sources: Object.freeze([
      source('source', 'nanoGPT upstream 与本地 model.py', 'https://github.com/karpathy/nanoGPT/commit/3adf61e154c3fe3fca428ad6bc3818b27a3b8291', '3adf61e154c3fe3fca428ad6bc3818b27a3b8291', 'nanoGPT/model.py'),
    ]),
  }),
  Object.freeze({
    id: 'minimind', name: 'MiniMind', role: '现代小模型组件替换', implementationFamily: 'minimind',
    supportedModes: modes('reference'),
    specs: Object.freeze({ parameters: 64_000_000, layers: 8, hiddenSize: 768, attentionHeads: 8, kvHeads: 4, headDim: 96, maxContext: 32768, yarnDefault: false, tieWordEmbeddings: true }),
    cache: Object.freeze({ kind: 'expanded-kv', layers: 8, kvHeads: 4, keyDim: 96, valueDim: 96, decodeComplexity: 'KV cache 避免历史 K/V 重投影；full attention decode 仍随历史长度线性增长' }),
    sources: Object.freeze([
      source('source', 'MiniMind upstream 与本地实现', 'https://github.com/jingyaogong/minimind/commit/393e387e9ad99f0f04c296e4c5e7353f4444629f', '393e387e9ad99f0f04c296e4c5e7353f4444629f', 'minimind/model/model_minimind.py'),
    ]),
  }),
  Object.freeze({
    id: 'qwen', name: 'Qwen3.6-27B', role: '线性注意力与 GQA 混合', implementationFamily: 'qwen3_5',
    supportedModes: modes('reference'),
    specs: Object.freeze({ parameters: 27_000_000_000, layers: 64, linearLayers: 48, fullAttentionLayers: 16, hiddenSize: 5120, attentionHeads: 24, kvHeads: 4, headDim: 256, maxContext: 262144, tieWordEmbeddings: false, lmHead: 'independent' }),
    cache: Object.freeze({
      kind: 'hybrid',
      fullAttention: Object.freeze({ layers: 16, kvHeads: 4, keyDim: 256, valueDim: 256 }),
      recurrent: Object.freeze({ layers: 48, heads: 48, keyDim: 128, valueDim: 128, dtype: 'fp32' }),
      convolution: Object.freeze({ layers: 48, combinedQkvDim: 10240, kernel: 4 }),
      decodeComplexity: '线性层状态恒定；16 个 full-attention 层仍随历史长度线性增长',
    }),
    sources: Object.freeze([
      source('config', 'Qwen/Qwen3.6-27B 官方 config', 'https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/config.json', '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9', 'configs/qwen3.6-27b-config.json'),
      source('source', '与本地 qwen3_5 源码逐字节匹配的 Transformers commit', 'https://github.com/huggingface/transformers/blob/b9090ae58cdaf9e7a195f38d3b2574dd408acaa6/src/transformers/models/qwen3_5/modeling_qwen3_5.py', 'b9090ae58cdaf9e7a195f38d3b2574dd408acaa6', 'research/qwen3.6/modeling_qwen3_5.py'),
    ]),
  }),
  Object.freeze({
    id: 'glm', name: 'GLM-5.2', role: 'MLA、稀疏索引与跨层共享', implementationFamily: 'glm_moe_dsa',
    supportedModes: modes('reference', 'optimized'),
    specs: Object.freeze({ parameters: 753_000_000_000, activeParameters: 40_000_000_000, layers: 78, hiddenSize: 6144, attentionHeads: 64, kvHeads: 64, keyDim: 256, valueDim: 256, latentDim: 576, instantiatedIndexers: 21, indexerDim: 128, maxContext: 1048576 }),
    cache: Object.freeze({ kind: 'mla-reference-expanded', layers: 78, reference: Object.freeze({ heads: 64, keyDim: 256, valueDim: 256 }), optimized: Object.freeze({ latentDim: 576 }), indexer: Object.freeze({ layers: 21, dim: 128 }), decodeComplexity: 'top-k 稀疏计算减少参与位置，但 cache 仍随历史增长' }),
    sources: Object.freeze([
      source('config', 'zai-org/GLM-5.2 官方 config', 'https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/config.json', 'b4734de4facf877f85769a911abafc5283eab3d9', 'configs/glm-5.2-config.json'),
      source('source', '与本地源码逐字节匹配的 Transformers commit', 'https://github.com/huggingface/transformers/blob/6b8369fe9f89312dc4b523c4ad449640bd3e68f3/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py', '6b8369fe9f89312dc4b523c4ad449640bd3e68f3', 'research/glm-5.2/modeling_glm_moe_dsa.py'),
    ]),
  }),
  Object.freeze({
    id: 'kimi', name: 'Kimi-K3', role: 'KDA、Gated MLA 与超大 MoE 多模态', implementationFamily: 'kimi_linear',
    supportedModes: modes('reference', 'optimized'),
    specs: Object.freeze({ parameters: 2_800_000_000_000, activeParameters: 104_000_000_000, layers: 93, kdaLayers: 69, fullAttentionLayers: 24, hiddenSize: 7168, attentionHeads: 96, keyDim: 192, valueDim: 128, latentDim: 512, maxContext: 1048576, visionParameters: 401_000_000, firstDenseLayers: 1, moeLayers: 92, experts: 896, expertsPerToken: 16, sharedExperts: 2 }),
    cache: Object.freeze({
      kind: 'hybrid-kda-mla',
      fullAttention: Object.freeze({ layers: 24, heads: 96, keyDim: 192, valueDim: 128 }),
      optimized: Object.freeze({ latentDim: 512, keyRemainderDim: 64 }),
      unquantified: Object.freeze(['69 层 KDA recurrent state：外部 FLA state 的精确 dtype 无本地证据', '69 层三组 KDA conv state：运行时布局/dtype 未在本地证据中完整固定']),
      decodeComplexity: 'KDA 状态恒定；24 个 full-attention 层仍随历史增长',
    }),
    sources: Object.freeze([
      source('config', 'moonshotai/Kimi-K3 官方 config', 'https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/config.json', '9f62e4e9fffbd0a83ddd60e1c209d828994b3569', 'configs/kimi-k3-config.json'),
      source('source', '当前 KimiDynamicCache 拼接展开 K/V', 'https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py', '9f62e4e9fffbd0a83ddd60e1c209d828994b3569', 'research/kimi-k3/modeling_kimi_linear.py'),
    ]),
  }),
  Object.freeze({
    id: 'deepseek', name: 'DeepSeek-V4-Flash-0731', role: '分层压缩稀疏注意力与 DSpark', implementationFamily: 'deepseek_v4',
    supportedModes: modes('reference', 'deployment'),
    specs: Object.freeze({ parameters: 304_000_000_000, activeParameters: 13_000_000_000, layers: 43, hiddenSize: 4096, attentionHeads: 64, kvHeads: 1, headDim: 512, window: 128, maxContext: 1048576, hfNextnPredictLayers: 1, inferenceMtpLayers: 3, dsparkBlockSize: 5, dsparkInjectedMainLayers: Object.freeze([40, 41, 42]), officialVllmSpeculativeTokens: 7 }),
    cache: Object.freeze({
      kind: 'layered-compression', headDim: 512, window: 128,
      compressRatioCounts: Object.freeze({ 0: 2, 4: 21, 128: 20 }),
      indexer: Object.freeze({ ratio: 4, layers: 21, dim: 128 }),
      decodeComplexity: '滑窗与压缩历史共同决定稀疏 attention 参与量；reference buffer 按运行时 max_seq_len/max_batch_size 预分配',
    }),
    sources: Object.freeze([
      source('config', '官方 HF 主 config：num_nextn_predict_layers=1', 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json', '7872f01b1d1fe23eabc4c98b48bffcef5a386062', 'configs/deepseek-v4-flash-config.json'),
      source('source', '官方 inference 源码与本地快照', 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py', '7872f01b1d1fe23eabc4c98b48bffcef5a386062', 'research/deepseek-v4/model.py'),
      source('source', '本地 inference_config：n_mtp_layers=3、block_size=5、注入 40/41/42', 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/config.json', '7872f01b1d1fe23eabc4c98b48bffcef5a386062', 'research/deepseek-v4/inference_config.json'),
      source('deployment', '官方 vLLM 示例：KV FP8、FP4 Indexer cache、7 speculative tokens', 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md', '7872f01b1d1fe23eabc4c98b48bffcef5a386062', '--kv-cache-dtype fp8; use_fp4_indexer_cache:true; num_speculative_tokens=7'),
    ]),
  }),
]);

export function getModel(id) {
  const model = MODELS.find((entry) => entry.id === id);
  if (!model) throw new RangeError(`Unknown model: ${id}`);
  return model;
}
