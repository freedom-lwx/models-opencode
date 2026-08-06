# GLM-5.2 模型源码深度审计报告

> 审计对象: `zai-org/GLM-5.2` (744B params, 40B active, MIT)
> 源码来源: HuggingFace `transformers` 主分支 `src/transformers/models/glm_moe_dsa/`
> 关键事实: GLM-5.2 的 HF 实现**无 custom_code**,架构 `GlmMoeDsaForCausalLM` 继承自 `DeepseekV32*`(DeepSeek-V3.2)。GLM-5.2 = **DeepSeek-V3.2 NSA 架构 + IndexShare(跨层索引复用) + interleaved RoPE + GLM 配置**。
> 图例: [源码事实] / [官方材料](论文·模型卡·博客) / [推导] / [未知]

## 0. 继承关系与代码定位 [源码事实]

`modular_glm_moe_dsa.py:28-42` 明确导入:
```python
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention, DeepseekV3RMSNorm,
    apply_rotary_pos_emb_interleave, eager_attention_forward)
from ..deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32DecoderLayer, DeepseekV32ForCausalLM,
    DeepseekV32Indexer, DeepseekV32Model, ...)
```
- `GlmMoeDsaConfig(DeepseekV32Config)` — `modular:50`
- `GlmMoeDsaIndexer(DeepseekV32Indexer)` — `modular:133`,唯一改动: interleaved RoPE
- `GlmMoeDsaAttention(DeepseekV3Attention)` — `modular:197`,新增 `prev_topk_indices` 传播
- `GlmMoeDsaDecoderLayer/Model/ForCausalLM(DeepseekV32*)` — `modular:296/334/393`

DSA = **DeepSeek Sparse Attention**(DeepSeek-V3.2 的 NSA 原生稀疏注意力),非 GLM 独立设计。文档 `glm_moe_dsa.md` 原文: "using DeepSeek Sparse Attention (DSA)";`configuration.py:160` `layer_types = ["deepseek_sparse_attention"]*N`。

文件行号引用基于 `modeling_glm_moe_dsa.py`(生成文件,827 行)。

---

## 1. MLA (Multi-head Latent Attention)

**(a) 解决什么问题**: 标准 MHA 的 KV cache 随 `层数×头数×头维×序列长` 线性增长,长上下文(1M)下显存爆炸、解码带宽受限。MLA 将 Q/KV 低秩压缩到小潜变量,缓存压缩潜变量而非展开的 K/V。

**(b) 标准 Transformer**: Q=W_q h, K=W_k h, V=W_v h,每 token 缓存 `[H×d_k + H×d_v]`;GLM-5.2 若用标准 MHA 每 token 缓存 = 64×(192+256)=28672 元素/层。

**(c) 数学公式**:
- Q 压缩-展开: `c_q = LN_q(W_a^q h) ∈ ℝ^{2048}`, `Q = W_b^q c_q ∈ ℝ^{64×256}`,拆 `Q=[Q_nope(192) ‖ Q_rot(64)]`
- KV 压缩-展开(MQA 下投影+上投影):
  `c_kv = W_a^{kv} h ∈ ℝ^{576}` → 拆 `[c(512) ‖ k_pe(64)]`; `c = LN_kv(c)`
  `[K_nope ‖ V] = split(W_b^{kv} c) ∈ ℝ^{64×192}, ℝ^{64×256}`
- 解耦 RoPE: `K = [K_nope ‖ broadcast(k_pe)]`,rope 维独立广播,不参与低秩压缩
- 注意力: `O = softmax(Q K^T / √d_k + mask) V`,`Y = W_o O`
- KV cache 仅存 `c(512)+k_pe(64)=576` 元素/token/层 → 相对标准 28672 压缩 **~49.7×**

**(d) 关键代码**: `modeling_glm_moe_dsa.py:331-372`(投影定义), `403-417`(前向压缩展开)
```python
# modeling_glm_moe_dsa.py:403-417
q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))          # [B,S,2048]
q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2) # [B,64,S,256]
q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
compressed_kv = self.kv_a_proj_with_mqa(hidden_states)              # [B,S,576]
kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
k_pass = self.kv_a_layernorm(kv_pass)                               # [B,S,512]
...
key_states, value_states = self.expand_kv(k_pass, k_rot)            # [B,64,S,256],[B,64,S,256]
```

**(e) 输入输出 shape**:
- 输入 `hidden_states`: `[B, S, 6144]`
- `q_resid`: `[B,S,2048]` → `q_states`: `[B,64,S,256]`
- `compressed_kv`: `[B,S,576]` → `key_states/value_states`: `[B,64,S,256]`(cache 后 T=S+past)
- 输出 `attn_output`: `[B,S,6144]`(经 `o_proj`)

**(f) 为什么这样实现 / 与 DeepSeek-V3 对比**:
- 完全复用 DeepSeek-V3 MLA(`DeepseekV3Attention`),`q_lora_rank=2048`/`kv_lora_rank=512` 与 DS-V3 同量级。
- **差异: `rope_interleave=true`**。DS-V3 用 `rotate_half`(把 rope 维拆前后两半旋转);GLM 用 `apply_rotary_pos_emb_interleave`(`modeling:127-163`),按交错对 `(x0,x1),(x2,x3)` 旋转。代码注释 (`modeling:131-134`): "the output is bit-identical to the de-interleaved rotate_half formulation while avoiding the extra contiguous copy" —— 即数学等价但省一次 contiguous 拷贝。`qk_rope_head_dim=64` 被设为 `config.head_dim`(`configuration:153`),使继承的 Llama RoPE 直接对 64 维生效。
- 解耦 RoPE 是 MLA 的核心 trick: rope 部分位置编码需在 Q/K 间精确对齐,不能放进会被 `kv_b_proj` 混合的低秩潜空间,故单独拆出 `k_pe` 广播。

**(g) 计算/显存影响** [推导]:
- 参数: Q 侧 `6144×2048 + 2048×16384`≈37M;KV 侧 `6144×576 + 512×28672`≈18M/层。
- **注意**: eager/SDPA 路径下 `past_key_values.update(key_states, value_states,...)`(`modeling:420`)缓存的是**展开后**的 `[B,64,T,256]`,并非压缩潜变量 —— `_supports_flash_attn = False`(`modeling:648`,注释"flash-mla kernels need a bit more work")。因此**纯 transformers 不享受 MLA 的 KV 压缩收益**,真正压缩 cache 需 vLLM/SGLang 的 flash-mla 路径(消费 `sparse_indices`,`modeling:464`)。[源码事实]压缩潜变量 cache 仅在框架专用 MLA kernel 中实现。

---

## 2. DSA (DeepSeek Sparse Attention) — Indexer

**(a) 解决什么问题**: 长序列下核心注意力 `O(L²)`。DSA 用轻量"闪电索引器"为每个 query 选 top-k 个最相关 key,把核心注意力降到 `O(Lk)`,k=2048。

**(b) 标准 Transformer**: 全注意力 `softmax(QK^T/√d)V`,每个 query 看所有 key。

**(c) 数学公式** (NSA 式,`modeling:226-257`):
- 索引器独立投影: `q^I = W_q^b c_q ∈ ℝ^{32×128}`, `k^I = LN_k(W_k h) ∈ ℝ^{128}`(单头 key)
- 拆 rope: `[q_rot(64) ‖ q_pass(64)]`,对 q_rot/k_rot 施 interleave RoPE
- 逐头打分(ReLU 而非 softmax,使分数可加可比较):
  `S = ReLU(q^I (k^I)^T · d^{-0.5}) ∈ ℝ^{B,S,32,T}`
- 学习式逐头混合权重: `w = W_w h · H_I^{-0.5} ∈ ℝ^{B,S,32}`
- 跨头加权求和: `g = wᵀ S = Σ_h w_h S_h ∈ ℝ^{B,S,T}`(每 token 重要性)
- 选 top-k: `I = TopK(g + causal_mask, 2048).indices`
- 选中的 token 置 0、其余置 `-inf` 形成加性稀疏 mask(`modeling:440-449`),核心注意力只在 2048 个 key 上算

**(d) 关键代码**: `GlmMoeDsaIndexer` `modeling:166-257`;前向 `197-257`
```python
# modeling_glm_moe_dsa.py:241-257
scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
scores = F.relu(scores)                                  # ReLU 打分,非 softmax
weights = self.weights_proj(hidden_states...).float() * (self.n_heads**-0.5)  # [B,S,32]
index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)        # [B,S,T] 跨头聚合
if attention_mask is not None:
    index_scores = index_scores + attention_mask
topk = min(self.index_topk, index_scores.shape[-1])
return index_scores.topk(topk, dim=-1).indices.to(torch.int32)  # [B,S,2048]
```

**(e) 输入输出 shape**:
- 输入: `hidden_states [B,S,6144]`, `q_resid [B,S,2048]`, `position_embeddings`
- 中间: `q [B,S,32,128]`, `k [B,S,128]→[B,S,T,128]`(cache), `scores [B,S,32,T]`, `index_scores [B,S,T]`
- 输出: `topk_indices [B,S,2048]` int32
- 索引器有独立 key cache: `past_key_values.update_indexer(k, layer_idx)`(`modeling:239`),每层 `[B,T,128]`

**(f) 为什么这样实现**:
- 是 **NSA(原生稀疏注意力)**,文档/docstring 明确 "Same as `DeepseekV32Indexer.forward`"(`modular:147`)。非独立设计。
- ReLU 打分 + 学习头权重 + 跨头求和是 NSA 的标志性设计: 相比 softmax,ReLU 让多头的分数可线性叠加,便于用一个小 MLP 学到"哪些头对当前 token 重要"再做加权聚合。
- 索引器 `@torch.no_grad()`(`modeling:197`/`modular:134`): topk 本身不可导,no_grad 省显存;索引器参数经 IndexShare 的**多层蒸馏**训练(见 §3),而非端到端反传穿过 topk。
- 索引器 query 复用主注意力的 `q_resid`(压缩潜变量 c_q),共享 `q_a_proj`,省一份下投影参数。

**(g) 计算/显存影响** [推导]:
- 索引器本身仍是 `O(L²)`(32 头 × L×T 打分),这正是 IndexShare 要解决的成本(见 §3)。
- 核心注意力从 `64×L×T` 降到 `64×L×2048`,1M 上下文下核心 attn 减少 ~500×。
- 索引器 key cache 每层仅 `[B,T,128]`,远小于 MLA KV cache。

---

## 3. IndexShare (跨层索引复用 = IndexCache)

**(a) 解决什么问题**: DSA 把核心注意力降到 `O(Lk)`,但**索引器自身仍是 `O(L²)` 且每层独立运行**——而相邻层选出的 top-k 高度相似。IndexShare 让多数层直接复用最近的"full"层的 top-k,删掉冗余索引计算。

**(b) 标准 Transformer / 标准 DSA**: 每个 attention 层各自做完整的索引选择。

**(c) 数学公式**:
- 对层 ℓ,令 `type[ℓ] ∈ {full, shared}`:
  - `full`: `I_ℓ = Indexer(h_ℓ, c_qℓ)`(运行索引器)
  - `shared`: `I_ℓ = I_{ℓ*}`,`ℓ* = max{ℓ'≤ℓ : type[ℓ']=full}`(直接复用)
- 模式生成(`configuration:145-149`): `full iff (max(ℓ-offset+1,0) % freq)==0`,offset=3,freq=4
  → 层 0,1,2 为 full(引导),之后 6,10,14,... 每 4 层一个 full,其余 shared
  → **1 full + 3 shared = 删除 75% 索引计算**
- 训练(`IndexCache` 论文 [官方材料]): full 层的索引器经**多层蒸馏损失**训练,目标是其服务的所有层的平均注意力分布,使单索引器能服务 4 层。

**(d) 关键代码**: 配置 `configuration_glm_moe_dsa.py:136-149`;Attention `modeling:376-377,423-436`;DecoderLayer `modeling:616,629,638`;Model 传播 `modeling:732-743`
```python
# modeling_glm_moe_dsa.py:376-377  —— shared 层不建索引器
self.skip_topk = config.indexer_types[layer_idx] == "shared"
self.indexer = None if self.skip_topk else GlmMoeDsaIndexer(config, layer_idx)

# modeling_glm_moe_dsa.py:423-436  —— shared 层复用上一层传入的 topk
if self.indexer is not None:
    topk_indices = self.indexer(hidden_states, q_resid, ...)   # full: 计算
else:
    if prev_topk_indices is None:
        raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
    topk_indices = prev_topk_indices                            # shared: 直接复用

# modeling_glm_moe_dsa.py:732-743  —— 跨层 topk 传播
topk_indices = None
for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    hidden_states, topk_indices = decoder_layer(
        hidden_states, ..., prev_topk_indices=topk_indices, ...)  # 上轮 topk 喂给下层
```
代码中 4 处标注 `# MAIN DIFF with DSV3.2`(`modular:305,318,373,382`),即相对 DeepSeek-V3.2 的唯一实质改动。

**(e) 输入输出 shape**: `topk_indices [B,S,2048]` int32,在层间顺序传递(full 层产生新值覆写,shared 层透传)。无额外显存(仅一个张量在循环中流动)。

**(f) 为什么这样实现**:
- [官方材料](arXiv 2603.12201 摘要): "the resulting top-k selections are highly similar across consecutive layers" —— 相邻层 top-k 高度冗余是复用的前提。
- 论文提出两种配置法: Training-free(贪心搜索最小化校准 loss,不改权重)与 Training-aware(多层蒸馏)。GLM-5.2 采用固定交错模式 `F S S S`(freq=4)+ 蒸馏训练。
- 前 3 层强制 full(`index_skip_topk_offset=3`)作为引导: 早期层表征差异大、且无"前一个 full 层"可复用。
- `index_share_for_mtp_iteration=true` [推导]: MTP 草稿 token 的迭代也复用索引,避免每步重算索引器。

**(g) 计算/显存影响** [官方材料]:
- 删除 75% 索引器计算(1/4 层运行)。
- 论文(30B 模型): prefill 加速 **1.82×**,decode 加速 **1.48×**。
- GLM-5.2 README/文档(1M 上下文): per-token FLOPs 降低 **2.9×**。
- 代价 [推导]: shared 层用"过时"的 top-k,引入轻微精度损失(论文称 negligible);靠多层蒸馏把单索引器训成 4 层的折中表征来补偿。

---

## 4. MoE 路由 (sigmoid + noaux_tc + 无 bias)

**(a) 解决什么问题**: 256 个专家中为每 token 选 8 个,兼顾容量与负载均衡,且不用辅助损失(避免训练不稳/开销)。

**(b) 标准 Transformer**: 单个 dense MLP;或 Switch/GShard 用 softmax 路由 + 辅助负载均衡损失。

**(c) 数学公式** (`modeling:502-527`):
- 路由 logits: `z = W_g h ∈ ℝ^{256}`(W_g 无 bias),`W_g ∈ ℝ^{256×6144}`
- sigmoid 打分: `s = σ(z)`(非 softmax,各专家独立)
- noaux_tc: `s' = s + b`(b 为 `e_score_correction_bias`);分组选组: `g_score = TopK(s' 按组, 2).sum`;选 `topk_group` 个组,组外置 `-inf`
- top-k: `I = TopK(s', 8)`;`w = s_I`(用原 sigmoid 分数,不含 bias)
- 归一化+缩放: `w = w / (Σw + 1e-20)`,再 `w = w · γ`(γ=2.5)
- 输出: `y = Σ_{i∈I} w_i E_i(h) + E_shared(h)`(shared 专家恒开)

**(d) 关键代码**: `GlmMoeDsaTopkRouter` `modeling:489-527`;`GlmMoeDsaMoE.forward` `modeling:584-591`
```python
# modeling_glm_moe_dsa.py:500-527
self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))   # 无 bias
self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts, dtype=torch.float32))  # 置零
...
router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))  # float32 路由
scores = router_logits.sigmoid()
scores_for_choice = scores + self.e_score_correction_bias                    # b=0 → bias-free
group_scores = scores_for_choice.view(-1, self.num_group, ...).topk(2, dim=-1)[0].sum(dim=-1)
group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
...
topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
topk_weights = scores.gather(1, topk_indices)                                # 用原 sigmoid 分数
if self.norm_topk_prob:
    topk_weights = topk_weights / (topk_weights.sum(...) + 1e-20)
topk_weights = topk_weights * self.routed_scaling_factor                     # ×2.5
```

**(e) 输入输出 shape**:
- 输入 `hidden_states [B,S,6144]` → 展平 `[B*S, 6144]`
- `router_logits/scores [B*S, 256]`,`topk_indices [B*S, 8]`,`topk_weights [B*S, 8]`
- 专家输出 `final_hidden_states [B*S, 6144]`(经 `index_add_` 累加)

**(f) 为什么这样实现**:
- **noaux_tc** = "no auxiliary loss + top-k (score) correction": 用可学 bias `b` 替代辅助损失做负载均衡(DS-V3 思路)。但 GLM-5.2 把 `e_score_correction_bias` 初始化为 0(`modeling:500`)且 `init.zeros_`(`modeling:667`)→ **实际 bias-free**;又因 `n_group=1, topk_group=1`,分组机制退化为 no-op(单组含全部 256 专家,组选择恒选唯一组)。故 GLM-5.2 路由 = **纯 sigmoid + top-8 + L1 归一化 + ×2.5 缩放**,无 bias、无分组、无辅助损失。
- **sigmoid vs softmax**: sigmoid 让各专家分数独立(可多专家同时高),更适合 top-k 多选;softmax 偏互斥。
- **routed_scaling_factor=2.5**: 归一化后 top-8 权重和为 1,与恒开的 shared 专家(1 个)相加,总 MLP"幅度"需校准;2.5 补偿稀疏路由对激活尺度的压缩,使 routed 部分与 shared 部分量级匹配。
- 路由强制 float32(`moe_router_dtype="float32"`,`modeling:504`)+ `e_score_correction_bias` 保持 fp32(`modeling:658`):保证 topk 数值稳定。
- shared 专家(`n_shared_experts=1`)吸收通用知识,routed 专家专精 —— 标准 DS-V3 设计。

**(g) 计算/显存影响** [推导]:
- 每 token 仅激活 8/256 专家 + 1 shared,active 参数 ~40B(744B 总)。
- `GlmMoeDsaExperts` 用 3D 权重张量 + 按 expert_hit 循环(`modeling:543-567`),仅算被命中专家;TP plan `packed_colwise`/`grouped_gemm`(`configuration:87-88`)支持 grouped GEMM。

---

## 5. MTP (多 Token 预测)

**(a) 解决什么问题**: 自回归逐 token 生成受限于访存带宽。MTP 训练一个"下一 token 的下一 token"预测头,推理时作投机解码的草稿头,提升吞吐与接受长度。

**(b) 标准 Transformer**: 仅预测下一 token(`lm_head`)。MTP 额外加 1 个 NextN 预测层。

**(c) 数学公式** [推导,基于 DeepSeek-V3 NextN 继承]:
- 主模型输出 `h_t`;MTP 层取 `concat(h_{t}, h_{t+1})`(或嵌入相加)→ 1 个 transformer-like 层 → `lm_head` 预测 `t+2`
- 推理(投机解码): 草稿头快速生成 k 个候选,主模型并行验证,接受匹配前缀
- `index_share_for_mtp_iteration=true`: 草稿迭代复用 full 层索引,不重跑索引器

**(d) 关键代码** [源码事实 + 未知]:
- [源码事实] `config.json:202` `"num_nextn_predict_layers": 1` — 1 个 MTP 层
- [源码事实] `config.json:20` `"index_share_for_mtp_iteration": true`
- [源码事实] `modeling:659` / `modular:331`: `_keys_to_ignore_on_load_unexpected = [r"model\.layers\.78.*"]` — **MTP 层权重位于 `model.layers.78`(78 主层为 0-77),transformers 显式忽略不加载**
- [源码事实] 文档 `glm_moe_dsa.md` 原文: **"The implementation in transformers does not include an MTP layer."**
- [未知] MTP 头具体实现**不在 transformers**,在 **SGLang / vLLM**(README 指明 SGLang v0.5.13.post1+、vLLM v0.23.0+ 支持)。SGLang 中 GLM-5.2 复用 DeepSeek-V3 NextN 实现(GLM 在 transformers 层面继承 `DeepseekV32ForCausalLM`)。本次未拉取 SGLang 源码,代码细节标 [未知]。

**(e) 输入输出 shape** [推导]: MTP 层输入 `[B, S, 6144]`(主模型隐状态 + 草稿嵌入)→ 输出 logits `[B, S, vocab=154880]`。

**(f) 为什么这样实现** [官方材料]:
- README/文档: "We also improve GLM-5.2's MTP layer for speculative decoding, increasing the acceptance length by up to 20%"。
- `num_nextn_predict_layers=1`(单层 MTP)是吞吐与精度的折中: 层少则草稿快但接受率受限于单步预测能力。
- 索引复用 (`index_share_for_mtp_iteration`) 让草稿生成几乎不增加 DSA 索引开销。

**(g) 计算/显存影响** [推导]: 训练时多预测 1 个 token 提升信号密度;推理时草稿头为 1 层 transformer + lm_head,开销远小于主模型,接受长度 +20% 等效吞吐提升。MTP 层参数在 BF16 checkpoint 中存在(被 transformers 忽略,被 SGLang/vLLM 使用)。

---

## 6. 3 层 Dense + 75 层 MoE 混合 (`first_k_dense_replace=3`)

**(a) 解决什么问题**: 全 MoE 在浅层路由不稳(浅层表征尚未分化,路由难学到有意义的专家分配);浅层用 dense 保证基础表征学习稳定,深层用 MoE 提容量。

**(b) 标准 Transformer**: 全 dense MLP,或全 MoE。

**(c) 数学公式**: 对层 ℓ:
- `mlp_layer_types[ℓ] == "dense"`(ℓ=0,1,2): `y = MLP_dense(h) = W_down( SiLU(W_gate h) ⊙ W_up h )`,`intermediate=12288`
- `mlp_layer_types[ℓ] == "sparse"`(ℓ=3..77): `y = MoE(h)`(见 §4),`moe_intermediate=2048`

**(d) 关键代码**: `config.json:110-113`(前 3 个 dense);`configuration:155-157`(默认派生);`modeling:600-603`
```python
# modeling_glm_moe_dsa.py:600-603
if config.mlp_layer_types[layer_idx] == "sparse":
    self.mlp = GlmMoeDsaMoE(config)
else:
    self.mlp = GlmMoeDsaMLP(config)   # 前 3 层 dense
```

**(e) 输入输出 shape**: 两者均 `[B,S,6144] → [B,S,6144]`。dense MLP 中间维 12288;MoE 每 expert 中间维 2048(8 expert×2048 + 1 shared×2048)。

**(f) 为什么这样实现** [推导 + 官方材料惯例]:
- 与 DeepSeek-V3(`first_k_dense_replace=3`)完全一致 —— GLM 直接沿用此惯例。
- 浅层 dense 稳定早期特征学习;深层 MoE 增容量。3 层是经验值(DS-V3、GLM-4.5 均如此)。
- 注意: `configuration:164-166` 有一段看似将默认改为"1 dense + rest sparse"的兜底代码,但因前面 `:155-157` 已在 `mlp_layer_types is None` 时设过值,这段不会触发 —— config.json 显式给出了完整 78 项 `mlp_layer_types`(3 dense + 75 sparse),直接使用。[源码事实]
- DSA 注意力在**所有 78 层**都启用(`configuration:159-160` `layer_types = ["deepseek_sparse_attention"]*N`),仅 MLP 在前 3 层不同。

**(g) 计算/显存影响** [推导]:
- 3 层 dense MLP 参数: 3 × (6144×12288×3) ≈ 680M;其余 75 层 MoE。
- 浅层 dense 避免了早期路由的"冷启动"不稳定,小幅增加前 3 层 FLOPs(全激活)换取训练稳定性。

---

## 附: 关键配置一览 [源码事实 config.json]

| 字段 | 值 | 机制 |
|---|---|---|
| hidden_size / layers / heads | 6144 / 78 / 64 | 整体 |
| q_lora_rank / kv_lora_rank | 2048 / 512 | MLA §1 |
| qk_nope/rope_head_dim, v_head_dim | 192/64, 256 | MLA §1 |
| rope_interleave / indexer_rope_interleave | true / true | §1,§2 |
| index_topk / n_heads / head_dim | 2048 / 32 / 128 | DSA §2 |
| index_topk_freq / index_skip_topk_offset | 4 / 3 | IndexShare §3 |
| indexer_types | 3×full + (full,3×shared)×18 + full,3×shared | §3 |
| scoring_func / topk_method | sigmoid / noaux_tc | MoE §4 |
| routed_scaling_factor / n_shared | 2.5 / 1 | MoE §4 |
| n_routed_experts / per_tok | 256 / 8 | MoE §4 |
| first_k_dense_replace | 3 | §6 |
| num_nextn_predict_layers | 1 | MTP §5 |
| e_score_correction_bias | 置零(buffer) | §4 bias-free |

## 附: 源码文件清单(已下载至 `research/glm-5.2/`)
- `config.json`(224 行,完整模型配置)
- `modeling_glm_moe_dsa.py`(827 行,生成实现)
- `modular_glm_moe_dsa.py`(402 行,**源真相**,标注 `MAIN DIFF with DSV3.2`)
- `configuration_glm_moe_dsa.py`(171 行)

## 结论

GLM-5.2 在架构上**高度复用 DeepSeek-V3.2**: MLA + NSA 索引器 + MoE 路由 + NextN MTP 均源自 DS-V3.2。GLM 团队的**核心原创贡献是 IndexShare(=IndexCache 论文)**:通过 `prev_topk_indices` 跨层传递 + `indexer_types` 交错模式,让 75% 的层复用索引,1M 上下文下 per-token FLOPs 降 2.9×。其余 GLM 特有改动: interleaved RoPE(省拷贝)、`e_score_correction_bias` 置零(bias-free 路由)、1M 上下文训练、MTP 接受长度优化。transformers 实现不含 MTP 层与 MLA 压缩 cache(需 SGLang/vLLM flash-mla)。
