# Qwen3.6-27B 架构审计 & 推理框架实现差异

> 审计依据: HF repo `Qwen/Qwen3.6-27B`(config.json / README / safetensors index)、transformers 内置 `models/qwen3_5/`(modeling_qwen3_5.py 2075 行 / configuration_qwen3_5.py 200 行)、arXiv:2412.06464(Gated DeltaNet, ICLR 2025)、vLLM 源码(qwen3_5.py / qwen3_next.py)、vLLM recipes、SGLang docs、llama.cpp(convert_hf_to_gguf.py / docs/speculative.md)、ggml-org GGUF repos。
> 标签: [源码事实]=transformers/vLLM/llama.cpp 源码直读; [官方材料]=HF README/recipe/blog; [推导]=基于源码与论文的工程推断; [未知]=未能获取。

---

## 第一部分: Qwen3.6-27B 架构分析

### 0. 工程事实
- HF repo **无任何自定义 .py**(无 modeling_/configuration_ *.py),`model_type=qwen3_5` 内置于 transformers(`transformers_version: 4.57.1`)。**无需 `trust_remote_code`**。`Qwen3_5ForConditionalGeneration` 由 `modular_qwen3_5.py` 自动生成。 [源码事实]
- 27B 为 **dense** 模型(`base_model_ep_plan = None`,configuration_qwen3_5.py:80);家族另有 MoE 变体(35B-A3B / 397B-A17B)。 [源码事实] [官方材料]
- 总 55.6 GB / 15 shard / apache-2.0;vision_config.model_type 在 text 层标 `qwen3_5`、vision 实为 `qwen3_5_vision`(configuration_qwen3_5.py:186 做了 legacy 重映射)。 [官方材料] [源码事实]
- 已下载 transformers 主干源码至 `research/qwen3.6/`: `modeling_qwen3_5.py`、`configuration_qwen3_5.py`、`vllm_qwen3_5.py`、`vllm_qwen3_next.py`、`index.json`。 [源码事实]

---

### 1. Gated DeltaNet(线性注意力层)

#### 1.1 数据流与投影
`Qwen3_5GatedDeltaNet`(modeling_qwen3_5.py:389)有 **4 个输入投影** [源码事实]:
| 投影 | 输出维度 | 作用 |
|---|---|---|
| `in_proj_qkv` | key_dim·2 + value_dim = 16·128·2 + 48·128 = 4096+6144 | 合并的 Q,K,V |
| `in_proj_z` | value_dim = 6144 | **输出门控** z(逐 value head) |
| `in_proj_b` | num_v_heads = 48 | delta 系数 β |
| `in_proj_a` | num_v_heads = 48 | 输入/遗忘门 α(Mamba2 式) |

> 注意 num_v_heads(48) // num_k_heads(16) = 3,即每个 key head 被 3 个 value head 共享(`repeat_interleave`,modeling_qwen3_5.py:521-523)。 [源码事实]

#### 1.2 Short Conv(linear_conv_kernel_dim=4)
- 对 QKV 联合做 **depthwise 因果 1D 卷积**,kernel=4,`groups=conv_dim`,激活 silu(modeling_qken3_5.py:407-414, 491-497)。 [源码事实]
- 调用 `causal_conv1d_fn`(训练/prefill)与 `causal_conv1d_update`(单步解码,原地更新 conv_state),来自 **Dao-AILab/causal-conv1d**;缺库则回退 torch 实现。 [源码事实]
- 这是 Mamba 系的 short conv:提供局部位移不变性 + 改善线性注意力表达能力。 [推导]

#### 1.3 Delta Rule + Mamba2 式门控(核心)
源码(modeling_qwen3_5.py:518-520) [源码事实]:
```
beta = sigmoid(b)                                  # delta 写入强度
g    = -A_log.float().exp() * softplus(a + dt_bias) # 输入/遗忘门 α(Mamba2 离散化)
```
- `A_log`(初始化 `uniform(0,16)` 取 log)、`dt_bias`(ones)是 **Mamba2 的离散化参数**:门 g = -exp(A_log)·softplus(a+dt_bias),作为递推状态的对角衰减。 [源码事实]
- 递推更新由 `flash-linear-attention` 库的两个 kernel 完成(modeling_qwen3_5.py:527-549) [源码事实]:
  - `chunk_gated_delta_rule`(分块并行,训练/prefill)
  - `recurrent_gated_delta_rule`(逐 token,单步 decode)
  - 两者均 `use_qk_l2norm_in_kernel=True`(q,k 在 kernel 内做 L2 归一化)
- **gated delta rule 语义**(论文 arXiv:2412.06464):状态更新 `S_t = diag(g_t)·S_{t-1} + β_t·(v_t − S_{t-1}·k_t)·k_tᵀ`;输出 `o_t = S_t·q_t`。括号项 `(v − S·k)` 即 **delta rule**:写入前先减去"被检索出的旧值",实现"定向覆盖"而非纯累加。 [官方材料] [推导]

#### 1.4 输出门控(RMSNormGated = swish)
- `core_attn_out = self.norm(core_attn_out, z)`(modeling_qwen3_5.py:558),`Qwen3_5RMSNormGated`(modeling_qwen3_5.py:184-199):先 RMSNorm,再 `× F.silu(z)`(silu = swish)。 [源码事实]
- **这就是 `output_gate_type=swish` 的落点**:线性注意力输出门控用 swish(而非 sigmoid)。 [源码事实] [推导]

#### 1.5 与同族架构的关系
- **vs Mamba2**: 论文标题即"Improving Mamba2 with Delta Rule"。同一套门控机制(`A_log`/`dt_bias`/softplus),但 Mamba2 是**加性写入** `S_t = A·S_{t-1} + B`,Gated DeltaNet 改用 **delta 写入**(矩阵值记忆 + 减法项),检索/长上下文更强。源码直接复用 Mamba2 的 A_log/dt_bias 命名与 causal-conv1d。 [官方材料] [源码事实]
- **vs DeltaNet**: 在 DeltaNet(delta rule + short conv)基础上 **加门** α。 [官方材料]
- **vs GLA(Gated Linear Attention)**: GLA 有门控但加性写入;Gated DeltaNet = GLA 的门控 + DeltaNet 的 delta 写入,二者互补(论文:门控负责快速遗忘,delta 负责精准更新)。 [官方材料]
- **vs RetNet**: RetNet 用固定指数衰减、无输入门、无 delta;属更早期线性注意力。 [推导]
- **vs RWKV**: RWKV 时间混合(WKV)是线性 RNN 式衰减;Gated DeltaNet 用矩阵值记忆 + delta,表达力更强。同属"门控线性 RNN"族。 [推导]
- **vs KDA**: KDA = **Kimi Delta Attention**(Moonshot Kimi-K3 的架构)。SGLang 官方文档对 Kimi-K3 列出 "fused KDA decode kernels / KDA-aware prefix caching"。KDA 与 Gated DeltaNet 同为 **delta-rule 线性注意力**,机制同源(短 conv + 门控 delta 递推 + 矩阵状态),实现细节(头划分、门控参数化)不同。 [官方材料] [推导]

> 输入门控小结: `in_proj_a` → α(经 softplus+A_log,作 Mamba2 式对角衰减,控制遗忘)是**输入门**;`in_proj_z` → silu(z) 是**输出门**。 [源码事实]

---

### 2. 混合布局:48 linear + 16 full

#### 2.1 布局规则
- `layer_types` 在 config.json 中**显式列出 64 项**;若缺省则由 `full_attention_interval=4` 生成(configuration_qwen3_5.py:112-117) [源码事实]:
  ```python
  "linear_attention" if bool((i+1) % 4) else "full_attention"
  ```
  即 0-based 第 3,7,11,…,63 层为 full(每 4 层一组的**最后一层**),其余为 linear。README 概括为 `16 × (3 × Gated DeltaNet → FFN, 1 × Gated Attention → FFN)`。 [源码事实] [官方材料]
- `Qwen3_5DecoderLayer`(modeling_qwen3_5.py:759-812)按 `block_type` 分支实例化 `linear_attn`(GatedDeltaNet)或 `self_attn`(Gated Attention);二者都接同一 `Qwen3_5MLP`(silu SwiGLU)与残差。 [源码事实]

#### 2.2 为什么 3:1
- 论文实验:Gated DeltaNet + sliding-window/full attention 的**混合**在训练效率与任务性能上均优于纯线性或纯全注意力;少量全注意力层恢复精确检索能力,线性层压低 KV 与算力。 [官方材料]
- 3:1 比例 + 每组末层 full:保证每段局部上下文都有一层"精确召回"锚点,且周期性刷新线性注意力的状态漂移。 [推导]
- 与 Kimi-K3(亦为 hybrid linear+full)同思路,但具体比例/位置不同。 [推导]

---

### 3. MRoPE(mrope_section=[11,11,10], interleaved)

- **旋转维度**: `dim = head_dim · partial_rotary_factor = 256 · 0.25 = 64`,即每头仅前 64 维参与旋转;`inv_freq` 长度 = 32(compute_default_rope_parameters, modeling_qwen3_5.py:137, 141)。 [源码事实]
- **三轴分配**: `mrope_section=[11,11,10]` 对应 **(Time, Height, Width)**,和 = 11+11+10 = 32 = 旋转维度/2。即 32 个频率按 11/11/10 切给 T/H/W。 [源码事实]
- **位置生成** `get_rope_index`(modeling_qwen3_5.py:1304):返回 `position_ids` 形状 `(3, batch, seq)` [源码事实]:
  - 文本(modality 0):三轴相同(`arange + current_pos` expand 到 3)→ 退化为标准 1D RoPE。
  - 图像/视频:用 `(T,H,W)` meshgrid 生成真正 3D 位置(modeling_qwen3_5.py:1295-1300);视频用时间戳分隔逐帧 grid。
- **交错(interleaved=true)**: `apply_interleaved_mrope`(modeling_qwen3_5.py:166-181)把分块布局 `[TTT…HHH…WWW]` 重排为交错 `[THWTHW…TT]`,注释明言"preserving frequency continuity"。相比 Qwen2.5-VL 的 `[16,24,24]`(偏空间),此处 `[11,11,10]` **更均衡且给 time 更多权重**,反映视频优先。 [源码事实] [推导]
- `rope_theta=1e7`;长文本用 YaRN 扩到 ~1.01M(vLLM recipe 给出 `rope_type=yarn, factor=4.0`)。 [官方材料]

---

### 4. 输出门控: attn_output_gate + output_gate_type=swish

两套**不同**的门控,易混淆 [源码事实]:

| 位置 | config 字段 | 实现 | 激活 | 源码 |
|---|---|---|---|---|
| **全注意力(Gated Attention)** | `attn_output_gate=true` | `q_proj` 输出翻倍,一半作 gate;`attn_output = attn_output * sigmoid(gate)` | **sigmoid** | modeling_qwen3_5.py:660-661, 686-689, 717 |
| **线性注意力(Gated DeltaNet)** | `output_gate_type=swish` | `RMSNormGated(core_attn_out, z) = RMSNorm(x)·silu(z)` | **swish(silu)** | modeling_qwen3_5.py:184-199, 558 |

- ⚠️ 注意命名误导: `output_gate_type=swish` 描述的是**线性注意力**的 swish 门;**全注意力实际用 sigmoid**,与字段名 `attn_output_gate`(无 _type)配套。 [源码事实] [推导]
- HF 参考实现将两者**硬编码**;`attn_output_gate` / `output_gate_type` 这两个 config 字段**在 HF modeling 代码中未被读取**(grep 无引用),由服务框架(vLLM/SGLang)消费。 [源码事实]
- 全注意力还带 `q_norm`/`k_norm`(RMSNorm,仅作用 head_dim,注释 "unlike olmo, only on the head dim",modeling_qwen3_5.py:672-673)+ GQA(24 Q / 4 KV, head_dim 256)。 [源码事实]

---

### 5. MTP 头: mtp_use_dedicated_embeddings=false

#### 5.1 权重结构(从 index.json 直读,15 个 mtp.* 键) [源码事实]
```
mtp.pre_fc_norm_embedding.weight   # 对 [embedding 输入] 的 RMSNorm
mtp.pre_fc_norm_hidden.weight      # 对 [hidden 输入] 的 RMSNorm
mtp.fc.weight                      # 拼接 [hidden, embedding] → hidden 的线性
mtp.layers.0.{input_layernorm, post_attention_layernorm,
              self_attn.{q,k,v,o}_proj, q_norm, k_norm,
              mlp.{gate,up,down}_proj}
mtp.norm.weight                    # 末层 norm
```
- **无 `mtp.embed_tokens`**;全文仅 `model.language_model.embed_tokens.weight` 一个嵌入表。 [源码事实]
- **`mtp_use_dedicated_embeddings=false` 的含义**: MTP 头**复用主模型的共享 embed_tokens**(既作输入嵌入,也作输出 lm_head 投影),不另设专属嵌入表。参数更省、与 DeepSeek-V3 MTP 一致。 [源码事实] [推导]
- MTP 单层(`mtp_num_hidden_layers=1`)用的是**全注意力**(self_attn,非 linear_attn):draft 需精确,故用贵但准的全注意力。 [源码事实]
- 输入 = 上一 token 的 hidden_state 与下一 token 的 embedding 拼接 → 双 `pre_fc_norm` → `fc` → 1 层 transformer → 共享 embed 投影出 logits(DeepSeek-V3 范式)。 [源码事实] [推导]

#### 5.2 HF 与服务框架的差异
- **HF transformers 参考实现完全不实现 MTP 推理**:仅在加载时 `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`(modeling_qwen3_5.py:823, 1614)忽略 MTP 权重。 [源码事实]
- MTP 仅用于**训练**与**服务端投机解码**(vLLM / SGLang / llama.cpp)。 [官方材料] [源码事实]

---

## 第二部分: 推理框架实现差异

### 6. vLLM(vllm>=0.19.0)

#### 6.1 架构继承与状态缓存
- `Qwen3_5DecoderLayer(Qwen3NextDecoderLayer)`(vllm_qwen3_5.py:114):Qwen3.5/3.6 复用 **Qwen3Next 基类**(vllm_qwen3_next.py,883 行,`Qwen3NextModel` 混入 `EagleModelMixin`)。 [源码事实]
- **Gated DeltaNet 状态走 Mamba 通道**: 用 `MambaStateDtypeCalculator.gated_delta_net_state_dtype` / `MambaStateShapeCalculator.gated_delta_net_state_shape` / `MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func`(vllm_qwen3_5.py:370-402, 574-604)。即线性注意力层用**递推状态缓存**(类 Mamba2),全注意力层用 **paged KV cache**——同一模型**两种缓存类型并存**。 [源码事实]
- 由此带来工程坑: prefix caching 对 Mamba cache 走 "align" 模式,**实验性**;CUDA graph capture size > mamba cache size 会 assert 失败(报错路径 `vllm/model_executor/models/qwen3_next.py:585` → `causal_conv1d_update`),需降 `--max-cudagraph-capture-size`(默认 512)。 [官方材料] [源码事实]

#### 6.2 融合内核与门控
- `attn_output_gate = getattr(config, "attn_output_gate", True)`(vllm_qwen3_next.py:263)**从 config 读**;q_proj 输出按 `(1+gate)` 翻倍。 [源码事实]
- **融合**: `use_fused_qk_norm_rope_gate` 把 "门控拆分 + QK-RMSNorm + (partial) NeoX-RoPE + gate 拷贝" 融进单 kernel `fused_qk_rmsnorm_rope_gate`(vllm_qwen3_next.py:324-351);门为 **"pre-sigmoid gate"**(注释, vllm_qwen3_next.py:336),即最终 `attn_output·sigmoid(gate)`,与 HF 一致。 [源码事实]
- `gqa_interleaved_layout=False`(vllm_qwen3_5.py:143)。 [源码事实]

#### 6.3 MTP 投机解码
- 命令(README/recipe): `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`(README) 或 `{"method":"mtp","num_speculative_tokens":1}`(recipe,版本差异)。 [官方材料]
- 不启用 MTP 时: `loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])`(vllm_qwen3_next.py:882)跳过 MTP 权重;启用时 MTP 模块作为 draft 加载,读 `vllm_config.speculative_config.num_speculative_tokens`(vllm_qwen3_next.py:857)。 [源码事实]
- `num_speculative_tokens` 可 1–5;MTP-1 降 TPOT、高接受率,但高并发下吞吐下降(draft 占 KV 容量)。**AMD GPU 的 MTP 仍在开发**。 [官方材料]

#### 6.4 并行与部署
- TP/EP/DP: 文本-only 用 `-dp 8 --enable-expert-parallel --language-model-only`(跳过 vision,腾 KV,开 EP);多模态用 `--mm-encoder-tp-mode data`(vision encoder 数据并行)+ `--mm-processor-cache-type shm`(共享内存缓存预处理多模态输入)。 [官方材料]
- `--language-model-only` 标志仅加载语言模型(vLLM 专有,跳过 vision encoder 与多模态 profiling)。 [官方材料]
- 长文本: YaRN override `rope_parameters` + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` → 1.01M。 [官方材料]
- 量化: FP8 / NVFP4(nvidia 官方量化版)/ Blackwell 用 `cu130-nightly`。 [官方材料]

---

### 7. SGLang(sglang>=0.5.10)

- MTP 命令(README): `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`。 [官方材料]
  - `NEXTN` = DeepSeek 式 next-token MTP;`speculative-eagle-topk` 借 EAGLE 式 top-k 采样;`num_steps 3 / draft_tokens 4`。
- SGLang 对**同族 delta-attention** 的优化可直接迁移到 Qwen3.6(二者机制同源): 官方对 Kimi-K3(KDA)列出 "fused KDA decode kernels / KDA-aware prefix caching / DP attention / MTP / PD disaggregation"。 [官方材料] [推导]
  - 关键: 线性注意力的**前缀缓存必须感知递推状态**(state-aware),否则前缀复用会破坏 delta 状态——这是 SGLang "KDA-aware prefix caching" 的工程要点,对 Gated DeltaNet 同样必要。 [推导]
- SGLang docs 的 Qwen3.5 cookbook 链接(`lmsysorg.mintlify.app/cookbook/llm/Qwen/Qwen3.5`)已 404,新 cookbook 迁移至 `docs.sglang.io/cookbook/autoregressive/...`(如 Kimi-K3 路径)。具体 SGLang 对 qwen3_5 的 kernel 融合细节**未直读源码**。 [未知]
- 文档结构: RadixAttention + prefix caching + 多 GPU 并行;跨 NVIDIA/AMD/Intel/TPU/Ascend。 [官方材料]

---

### 8. llama.cpp(GGUF + MTP-GGUF)

#### 8.1 GGUF 产物
- ggml-org 官方发布**分离的两类 GGUF** [官方材料]:
  - `ggml-org/Qwen3.6-27B-GGUF`(trunk)
  - `ggml-org/Qwen3.6-27B-MTP-GGUF`(MTP 头,单独文件)
  - 另有 35B-A3B 的对应 GGUF/MTP-GGUF
- README 示例: `llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF`(确认 Qwen3.5/3.6 arch 已注册)。 [官方材料]

#### 8.2 转换器 MTP 机制(convert_hf_to_gguf.py:121-126, 257-268) [源码事实]
- `--mtp`: **仅导出 MTP 头为独立 GGUF**(作投机 draft),输出加 `mtp-` 前缀。
- `--no-mtp`: 从 trunk GGUF 中**剔除** MTP;与 `--mtp` 配合分两次跑,产出 trunk + MTP 两文件。
- 需 `model_class.supports_mtp_export`;默认(不加标志)则把 MTP **打包进 trunk**(更省空间,因分离形式会**复制嵌入表**)。
- 注释明确: 分离形式允许 trunk 与 MTP **用不同量化**,可能更优。

#### 8.3 投机解码(docs/speculative.md) [源码事实]
- `llama-server` 支持: `draft`(独立小模型)、`draft-eagle3`(EAGLE-3,单层 transformer draft 读目标隐状态)。
- MTP 头 = 1 层全注意力 transformer(见 §5),正契合 EAGLE-3 式"单层 draft"形态;用法: `llama-server -m trunk.gguf -md mtp-draft.gguf --spec-type draft`(具体 spec-type 名对 MTP 未在文档直证)。 [源码事实] [推导]
- EAGLE-3 示例已含 `AngelSlim/Qwen3-4B_eagle3` → `Qwen/Qwen3-4B`,说明 Qwen3 系投机路径成熟。 [官方材料]
- llama.cpp 对 **Gated DeltaNet 递推内核** 的支持情况**未直证**(arch 注册源码路径未取到);CPU/Metal 后端跑混合线性注意力通常需自写递推 kernel,成熟度低于 vLLM/SGLang 的 GPU 路径。 [未知] [推导]

---

### 9. 框架横向对比(聚焦工程差异)

| 维度 | vLLM | SGLang | llama.cpp |
|---|---|---|---|
| **连续批处理** | 原生(continuous batching) | 原生 + RadixAttention | 有限(serve 端连续批,弱于前两者) |
| **分页注意力** | PagedAttention(KV block pool) | RadixAttention(前缀树复用) | KV cache 线性/统一管理,无 paged |
| **线性注意力状态** | 走 MambaState 通道,**混合缓存** | KDA-aware(state-aware)prefix cache | [未知] 递推内核成熟度存疑 |
| **CUDA Graph** | 支持;与 mamba cache 有 capture-size 坑 | 支持 | 支持(ggml graph) |
| **TP/EP/DP** | TP/EP/DP 全套,`--enable-expert-parallel`,`mm-encoder-tp-mode data` | TP/DP attention,PD 分离 | 单机多卡 rpc,无 EP |
| **FlashAttention** | FA2/FA3,全注意力层用 | FA,全注意力层用 | 无 FA,CPU/Metal 自实现 |
| **MTP 投机** | `speculative-config` method=`mtp`/`qwen3_next_mtp`,1–5 token | `--speculative-algo NEXTN`,steps=3/draft=4 | `--mtp` 导出独立 MTP-GGUF,`-md` 作 draft |
| **输出门控** | 读 `attn_output_gate`,融合 `fused_qk_rmsnorm_rope_gate`,sigmoid | [未知] 未直读源码 | [未知] |
| **长上下文** | YaRN→1.01M,`VLLM_ALLOW_LONG_MAX_MODEL_LEN` | YaRN(同) | YaRN/NTK 上下文扩展 |
| **量化** | BF16/FP8/NVFP4 | BF16/FP8/NVFP4/MXFP8 | Q4_K_M … Q8 全系 GGUF 量化 |
| **AMD** | 支持(MTP 仍开发中) | 支持(MI300X/MI355X) | ROCm/Vulkan |

#### 9.1 对 MLA / NSA / 线性注意力的 kernel 支持
- **MLA(DeepSeek)**: vLLM 原生(权重吸收 + decoupled RoPE);SGLang 原生(矩阵吸收 kernel);llama.cpp 部分支持(MLA 吸收路径)。 [推导]
- **NSA(Native Sparse Attention)**: 三框架均偏新,支持有限/跟进中。 [未知]
- **线性注意力(Gated DeltaNet/KDA)**: vLLM 复用 **Mamba/SSM 内核栈**(`gated_delta_net_state_*` + `causal_conv1d`);SGLang 有 **fused KDA decode kernels**(同族);llama.cpp 需 CPU/Metal 递推内核,**成熟度最低**。 [源码事实] [官方材料] [推导]

---

### 10. 关键工程结论

1. **27B 是 dense 混合注意力**(48 Gated DeltaNet + 16 Gated Attention,3:1),无 MoE;家族 MoE 变体共享同一 `qwen3_5` 架构。 [源码事实] [官方材料]
2. **Gated DeltaNet = Mamba2 门控 + DeltaNet delta 写入 + short conv**,递推状态使线性层**无 KV、O(1) decode**;全注意力层仍用 paged KV。 [源码事实] [官方材料]
3. **HF 参考实现不跑 MTP**(忽略 `mtp.*` 权重),也不读 `attn_output_gate`/`output_gate_type`(硬编码 sigmoid/silu);MTP 与门控字段的真正消费者是 vLLM/SGLang。 [源码事实]
4. **vLLM 把线性注意力当 Mamba state 管理** → 混合缓存 + "align" 前缀缓存(实验性)+ CUDA graph 坑;`qwen3_next.py` 是 Qwen3.5/3.6 的实现基类。 [源码事实]
5. **SGLang 的 NEXTN MTP** 与对 KDA 的 state-aware 优化可迁移;前缀缓存必须感知 delta 递推状态。 [官方材料] [推导]
6. **llama.cpp 把 MTP 头导成独立 MTP-GGUF** 作 draft(`--mtp`/`--no-mtp`),trunk 与 MTP 可异构量化;但 CPU/Metal 上 Gated DeltaNet 递推内核成熟度未证。 [源码事实] [未知]
7. **输出门控有两套**:全注意力 sigmoid(`attn_output_gate`)、线性注意力 swish(`output_gate_type=swish`);命名易误读,vLLM 进一步融合 QK-norm+RoPE+gate。 [源码事实]
8. **MRoPE `[11,11,10]` = (T,H,W)**,旋转维度仅 64(head_dim 256 的 25%);交错布局;文本退化为 1D。 [源码事实]
