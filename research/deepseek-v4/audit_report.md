# DeepSeek-V4-Flash-0731 模型实现深度审计

> 审计对象: `deepseek-ai/DeepSeek-V4-Flash-0731` (HF, 最后修改 2026-08-01)
> 参考实现: 仓库 `inference/{model.py, kernel.py, generate.py, convert.py}` (无根目录 `modeling_deepseek_v4.py`,Transformers v4.57.1 内置 `deepseek_v4` 类型)
> 论文: arXiv 2606.19348 "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
> 标注: [源码事实]=来自 inference/ 代码; [官方材料]=来自论文/README/config; [推导]=逻辑推断; [未知]=无法确认

## 0. 关键事实核对

| 项 | 任务背景给出 | 实际审计结果 | 来源 |
|---|---|---|---|
| 参数量 | 304B | 论文称 Flash=**284B**(13B 激活);0731 为刷新版,304B 可能含 embedding 重计 | [官方材料] |
| `compress_ratios` 长度 | (未明) | **46 个** = 43 主层 + 3 DSpark 层 | [源码事实] |
| `num_nextn_predict_layers` | 1 (HF config) | HF config=1,但 `inference/config.json` `n_mtp_layers=**3**`;参考实现按 3 个 DSparkBlock 构建 | [源码事实] |
| `modeling_deepseek_v4.py` | (假设存在) | **404 不存在**;实现位于 `inference/model.py` (45KB) | [源码事实] |

---

## 1. NSA → V4 的 CSA + HCA 混合注意力

### compress_ratios 数组解剖 [源码事实]
```
长度 46:  [0, 0,  (4,128)×20,  4,  0,0,0]
          ↑↑     ↑↑↑↑↑↑↑↑↑↑   ↑   ↑↑↑
          主层    主层2~41      主层42  DSpark层43~45
计数: ratio=0 → 5层; ratio=4 → 21层; ratio=128 → 20层
```
- `ratio=0`(层 0,1,42 附近 + 3 DSpark 层):**纯滑窗**(window=128),无压缩,关闭 YaRN,用基础 `rope_theta=10000`。
- `ratio=4` = **CSA (Compressed Sparse Attention)**:4× 压缩 + Indexer 稀疏选 top-512。
- `ratio=128` = **HCA (Heavily Compressed Attention)**:128× 极端压缩,attend 全部压缩 token(无 Indexer)。
- 论文摘要原文:"hybrid attention that combines **Compressed Sparse Attention (CSA)** and **Heavily Compressed Attention (HCA)**"。**CSA/HCA 即 NSA 的演化命名**。 [官方材料]

### (a) 解决什么问题
百万 token 下,标准注意力的 KV cache 与 attention FLOPs 随序列 O(n²) 爆炸。论文称 V4-Pro 在 1M 上下文仅需 V3.2 的 **27% 单 token FLOPs、10% KV cache**。 [官方材料]

### (b) 标准 Transformer
全密集注意力:每层保留全部 token 的 K/V,`softmax(QK^T/√d)V`,KV cache = `n_layers × n_heads × head_dim × seq_len`。

### (c) 数学公式 [源码事实 model.py:285-383, kernel.py:277-352]
**压缩 (Compressor, 门控池化)**:
```
kv = W_kv · x                    # [b,s,head_dim]
g  = W_gate · x + APE            # 绝对位置嵌入 [ratio, head_dim]
# 每 ratio 个连续 token 一组, softmax 加权求和:
KV_comp[t] = Σ_{i∈block_t} softmax(g_i) · kv_i
```
ratio=4 时 `overlap=True`,`coff=2`,维护重叠窗口(当前窗 + 前一窗)使压缩边界平滑。

**稀疏选择 (Indexer, 仅 ratio=4)** [model.py:386-439]:
```
q_idx = rotate(Hadamard, W_qb(qr))   # 复用 Q 的低秩潜变量 qr
kv_idx = Compressor_with_Hadamard(x) # 独立压缩器, 带 Hadamard 旋转 + FP4 仿真
score = relu_(<q_idx, kv_idx>) · W_proj(x)   # einsum 后 relu
topk_idxs = score.topk(512, dim=-1)          # 每 query 选 512 个压缩位置
```
ratio=128 时跳过 Indexer,用 `get_compress_topk_idxs` 取**全部**压缩位置(因 128× 压缩后位置已极少)。

**稀疏注意力 (sparse_attn kernel, FlashAttention 风格)** [kernel.py:293-352]:
```
对每个 (b, pos): 按 topk_idxs gather KV 块 → 在线 softmax (running max/sum) → 含可学习 attn_sink
o = (Σ_blocks softmax(Q·KV_block·scale) · KV_block + e^{sink-max}) / (Σ_exp + e^{sink-max})
```

### (d) 关键代码位置
- 压缩门控池化: `model.py:322 Compressor.forward`
- Indexer top-k: `model.py:426 index_score = einsum(...).relu_()...topk`
- 稀疏注意力核: `kernel.py:277 sparse_attn_kernel` (含 `attn_sink` 偏置)
- 压缩位置用独立 RoPE: `model.py:482 compress_rope_theta=160000`

### (e) 输入输出 shape [源码事实]
```
x:        [b, s, 4096]
q:        [b, s, 64, 512]   (n_heads=64, head_dim=512)
kv (MQA): [b, s, 512]       (单 KV 头)
压缩后 KV: [b, s//ratio, 512]
topk_idxs:[b, s, win(128) + 512]   # 滑窗128 + 稀疏512
o:        [b, s, 64, 512] → [b, s, 4096]
```

### (f) 为什么这样实现
- **CSA(4×)保细节**:轻度压缩 + 学习式稀疏选 512,保留精细局部/检索能力。
- **HCA(128×)给全局**:极端压缩后 token 极少,直接 attend 全部,提供廉价全局上下文。
- **交替堆叠**(4,128,4,128...):CSA 与 HCA 互补,单层成本均衡下降。21 个 CSA + 20 个 HCA 近 1:1。
- 与纯滑窗(层 0,1,42)搭配:浅层用纯局部建模低阶特征,深层(42)纯滑窗供 DSpark 抽取干净隐状态。

### (g) 计算/显存影响 [推导]
- KV cache: 压缩层每 token 仅存 `512/ratio` 个有效位(ratio=128 层几乎不增长)。混合后全模型 KV ≈ 纯滑窗级别,论文称 10% V3.2。
- attention FLOPs: 每 query 仅算 `128+512` 个 KV(ratio=4 层)而非全序列 → 大幅下降,论文称 27%。
- 代价:压缩有损;Indexer 增加选 token 的额外 forward(但用低秩潜变量 + FP4 仿真,成本可控)。

### 与 GLM DSA 区别 [未知]
任务提及 "GLM DSA" 无法在本次源码中验证具体对照。从 DeepSeek 侧看:NSA/V4-CSA 的核心差异点是**压缩与选择均为可学习**(门控池化 + learned Indexer),而非启发式分块或固定稀疏模式。GLM 侧的具体机制需对照其源码确认。

---

## 2. MQA 变体 (num_key_value_heads=1) + q_lora + o_lora + o_groups

### (a) 解决什么问题
进一步压缩 KV cache 与 Q/O 投影参数,同时保留多头表达力。

### (b) 标准 Transformer / MLA
- MHA: 每头独立 KV,KV cache = `n_heads × head_dim`。
- DeepSeek-V2/V3 MLA: 用低秩潜变量 `c_kv` 压缩 KV(`kv_lora_rank`),解耦 cache 与头数。
- V4: 改用 **MQA**(单 KV 头共享于全部 64 query 头)+ 保留 **Q 低秩潜变量**。

### (c) 结构与公式 [源码事实 model.py:442-548]
```python
# Q: 低秩潜变量 (latent Q), 复用给 Indexer
qr = q_norm(wq_a(x))          # dim 4096 -> q_lora_rank 1024
q  = wq_b(qr)                 # 1024 -> 64*512, 再 reshape
q *= rsqrt(q.square().mean(-1) + eps)   # 额外逐头 RMS 归一化
apply_rotary_emb(q[..., -64:], freqs)   # 仅末 64 维 RoPE
# KV: 极致 MQA, 单头
kv = kv_norm(wkv(x))          # 4096 -> 512 (单头!)
apply_rotary_emb(kv[..., -64:], freqs)
act_quant(kv[..., :-64], ...) # 非 rope 维 FP8 仿真, rope 维留 bf16
# O: 分组低秩
o  = einsum('bsgd,grd->bsgr', o, wo_a)  # 8 组, 每组共享 o_lora_rank=1024
x  = wo_b(o.flatten(2))                 # row-parallel all-reduce
```

### (d) 关键代码位置
- 单 KV 头: `model.py:466 self.wkv = Linear(self.dim, self.head_dim)` (输出仅 `head_dim=512`,无头维)
- Q 低秩 + 额外归一化: `model.py:502-504`
- 分组 O 低秩: `model.py:468-469 wo_a/wo_b`,`n_groups=o_groups=8`
- 潜变量复用: `model.py:517 self.indexer(x, qr, ...)` —— **Indexer 直接吃 Q 的低秩潜变量 qr**,省一份投影。

### (e) 输入输出 shape [源码事实]
```
x:   [b, s, 4096]
qr:  [b, s, 1024]      (Q 潜变量, 同时喂 Indexer)
q:   [b, s, 64, 512]
kv:  [b, s, 512]       (MQA: 1 头 × 512)
o:   [b, s, 64, 512] -> wo_a(8组×1024) -> [b, s, 4096]
```

### (f) 为什么这样实现 / 与 MLA 关系 [推导]
- **MQA 比 MLA 更激进**:KV cache 每 token 仅 `512` 元素(且非 rope 维 FP8)。V4 之所以敢放弃 MLA 的潜变量 KV 压缩,是因为 **CSA/HCA 已在 token 维度大幅压缩 KV**——每层每 token 实际 KV 增量已极小,再叠加 MLA 收益递减,反而 MQA 更简单、Indexer 更易复用。
- **保留 Q 低秩**:Q 仍用 `q_lora_rank=1024` 潜变量,既省 Q 投影参数(64×512×4096 → 2 阶低秩),又把潜变量 `qr` **复用给 Indexer 的查询**,架构耦合度高。
- **o_groups=8 分组低秩**:O 投影用 8 组每组 `o_lora_rank=1024`,在参数量与表达力间折中(纯低秩会丢组间差异,分组恢复部分)。
- 额外逐头 RMS 归一化(`q *= rsqrt(...)`)稳定低比特定点推理下的 Q 幅度。

### (g) 计算/显存影响 [推导]
- KV cache: 单头 512/ token · 层;非 rope 维 FP8(e4m3)→ 实际字节约 `448×1 + 64×2` ≈ 576 字节/token/层(压缩层再除以 ratio)。
- Q 投影参数: `4096×1024 + 1024×64×512` ≈ 43M vs 全秩 `4096×64×512`=134M,省 ~3×。
- 注意:单 KV 头被 64 头共享 → attention 计算仍是 64 头并行(gather 同一 kv)。

---

## 3. Hash Capacity (hc_*) = Manifold-Constrained Hyper-Connections (mHC)

> **澄清**: `hc_*` **不是** MoE 哈希路由(MoE 哈希路由是 `Gate.hash`/`n_hash_layers`)。`hc_*` = **Hyper-Connections(超连接)**,即残差流的"多副本扩展"机制。论文摘要:"**Manifold-Constrained Hyper-Connections (mHC)** that enhance conventional residual connections"。 [官方材料]

### (a) 解决什么问题
标准残差 `x = x + f(x)` 在深层/并行分支下梯度流单一、表达受限。Hyper-Connections 维护隐状态的**多副本**,允许每层动态决定如何混合/分裂各副本,改善梯度流并支撑多 token 预测。mHC 加"流形约束"(双随机矩阵)防止副本坍缩。

### (b) 标准 Transformer
`x_{l+1} = x_l + Attn(Norm(x_l))` 单条残差流。

### (c) 数学公式 [源码事实 model.py:652-716, kernel.py:372-438]
每层维护 `hc_mult=4` 份隐状态副本,`hc_dim = 4×4096`。
```
# hc_pre: 4 副本 -> 1 (压缩成单输入给 Attn/FFN)
mixes = Linear(hc_fn, flatten(x)) · rsqrt(mean(x²)+eps)   # [b,s,(2+4)*4]
pre  = sigmoid(mixes[:4]·scale[0] + base[:4]) + eps        # [b,s,4]
post = 2·sigmoid(mixes[4:8]·scale[1] + base[4:8])          # [b,s,4]
comb_raw = mixes[8:]                                        # [b,s,4,4]
# comb 经 Sinkhorn 归一化为双随机矩阵:
comb = softmax(comb_raw, dim=-1) + eps
for it in 1..20:                      # hc_sinkhorn_iters=20
    comb /= (comb.sum(-1) + eps)      # 行归一化
    comb /= (comb.sum(-2) + eps)      # 列归一化
y_pre = Σ_j pre_j · x_j               # 加权求和成单份

# hc_post: 1 -> 4 (Attn 输出 + 残差副本混合)
y_post[j] = post_j · f(x) + Σ_k comb[j,k] · x_k
```

### (d) 关键代码位置
- `Block.hc_pre` / `hc_post`: `model.py:680-693`
- Sinkhorn 双随机核: `kernel.py:415-423` (交替行列归一化 20 次)
- 终层 `hc_head`(无 Sinkhorn,纯 sigmoid 4→1): `model.py:709-716`

### (e) 输入输出 shape [源码事实]
```
x:        [b, s, 4, 4096]   (4 副本)
hc_pre -> [b, s, 4096]      (Attn/FFN 单输入)
hc_post -> [b, s, 4, 4096]  (恢复 4 副本)
mixes:    [b, s, 24]        ((2+hc)*hc = 6*4)
comb:     [b, s, 4, 4]      (双随机)
```

### (f) 为什么用 Sinkhorn / "流形约束" [推导]
- 双随机矩阵(行和=列和=1)构成 **Birkhoff 多面体**(双随机矩阵的凸包 = 置换矩阵的凸组合),即一个凸"流形"。
- 约束 comb 为双随机 ⇒ 每个副本的总贡献量守恒(无副本被饿死或爆炸),等价于强制"平衡置换式混合"。这是 mHC 相对原始 Hyper-Connections(V3.2)的稳定性升级,论文称其增强残差连接。 [官方材料]+[推导]
- `hc_eps=1e-6` 防止除零;`pre` 加 eps 保非零下限。
- `n_hash_layers=3`(MoE 哈希路由,前 3 层)与此**无关**——那是 `Gate.tid2eid` 按 token_id 查表选专家,见第 5 节。两者共用 "hash" 字样易混淆。

### (g) 计算/显存影响 [推导]
- 隐状态显存 ×4(4 副本);但 Attn/FFN 输入仍是单份(`hc_pre` 后),故算力不 ×4。
- Sinkhorn 20 次迭代在 4×4 小矩阵上,开销可忽略(kernel.py 每线程处理一行)。
- 收益:更稳的深层训练 + 更好的多 token 预测支撑(隐状态多副本天然适配 MTP/DSpark)。

---

## 4. DSpark 推测解码

### (a) 解决什么问题
自回归解码受限于单步串行;推测解码用小草稿模型并行预测多 token,主模型批量验证以提升吞吐。DSpark 是 DeepSeek-V4 的自研版。

### (b) 标准 / 已有方案
- **MTP (DeepSeek-V3)**:额外 1 层预测下 1 token,串行。
- **EAGLE**:单层草稿 + 树形验证。
- **Medusa**:多头并行预测多 token(独立头)。

### (c) DSpark 结构与公式 [源码事实 model.py:750-874, 928-936]
```
# 深层特征注入 (从主模型 layer 40,41,42 抽取)
main_hidden = concat(mean_hc(layer40), mean_hc(layer41), mean_hc(layer42))  # [b,s,3*4096]
main_x = main_norm(main_proj(main_hidden))   # 12288 -> 4096

# 草稿块初始化: [真token, 噪声×4]  (block_size=5)
draft_ids = [input_id, 128799, 128799, 128799, 128799]
h = embed(draft_ids).unsqueeze(2).repeat(..., hc_mult, ...)   # 4 副本

# 3 个 DSparkBlock 串行 (n_mtp_layers=3)
for layer in mtp: h = layer(h, start_pos, ids, main_x)

# Markov 头: 大表式 bigram 偏置
for i in 0..4:
    logits[i] += markov_w2(markov_w1(output_ids[i]))   # embed(id)->256->vocab
    output_ids[i+1] = sample(logits[i])                # 自回归采样
confidence = proj(concat(hidden, markov_embed))        # 每 token 置信度
```
`DSparkAttention` 注意力对象:`[滑窗 KV(共享自主模型 main_kv) + 草稿块自身 KV]`(model.py:782-785)。

### (d) 关键代码位置
- 草稿 embed + 噪声初始化: `model.py:851-858 forward_embed`
- Markov 头: `model.py:795-804 DSparkMarkovHead`
- 置信度头: `model.py:807-815 DSparkConfidenceHead`
- 块内自回归采样: `model.py:860-874 forward_head`
- 主模型抽取目标层: `model.py:920-921 if i in target_layer_ids: main_hiddens.append(h.mean(2))`
- 推测入口: `model.py:928 forward_spec`

### (e) 输入输出 shape [源码事实]
```
main_hidden: [b, s, 12288] -> main_x [b, s, 4096]
draft h:     [b, 5, 4, 4096]   (block_size=5, hc_mult=4)
output_ids:  [b, 6]            (1 输入 + 5 草稿)
logits:      [b, 5, 129280]
confidence:  [b, 5, 1]
```

### (f) 为什么这样实现 / 与 EAGLE/MTP 区别 [推导]
| 维度 | MTP(V3) | EAGLE | **DSpark(V4)** |
|---|---|---|---|
| 草稿深度 | 1 层 | 1 层 | **3 层 DSparkBlock** |
| 特征来源 | 末层 | 末隐状态 | **深层多 layer(40,41,42) 拼接投影** |
| 多 token | 否(1 token) | 树 | **块(5)+ 噪声去噪式自回归** |
| 先验 | 无 | 无 | **Markov bigram 头(可学习查表)** |
| 验证信号 | 概率 | 概率 | **额外 confidence 头自评** |
| KV 复用 | 否 | 否 | **复用主模型 KV (main_kv)** |

- **噪声 token (128799)**:草稿块以"输入 + 4 噪声"起步,模型在块内自回归去噪生成 5 token——一次 forward 出多 token,共享草稿块内 KV 与主模型 KV,摊薄成本。
- **Markov 头 (`dspark_markov_rank=256`)**:轻量 bigram 查表偏置,提供廉价 token 先验,提升接受率。
- **深层注入 (target 40,41,42)**:不止用末层,而用倒数 3 层(均经 hc 副本均值),给草稿更丰富语义,接近"知识蒸馏"式特征。
- **confidence 头**:输出每 token 置信度,供验证阶段自适应接受/截断。

### (g) 计算/显存影响 [推导]
- 草稿 3 层 + Markov/置信度头参数相对主模型(43 层 304B)极小,显存增量可忽略。
- 每步出 5 草稿 token,若接受率高(目标 ≥3)则净吞吐提升 ~3×。
- 草稿块复用主模型 KV,避免重算长上下文,是高接受率关键。

> 注:`dspark_target_layer_ids=[40,41,42]` 对应 `compress_ratios` 中层 40=CSA(4)、41=HCA(128)、42=CSA(4),抽取的是混合注意力层的隐状态。 [源码事实]

---

## 5. sqrtsoftplus 路由打分 + swiglu_limit

### (a) 解决什么问题
MoE 路由打分函数选择影响专家利用与训练稳定;FFN 激活异常值会击穿 FP4/FP8 低比特定点。

### (b) 标准做法
- softmax:`exp(x_i)/Σexp(x_j)`,专家间耦合(归一化跨全部)。
- sigmoid:逐元素 `1/(1+e^{-x})`,非负但大输入饱和于 1。

### (c) 数学公式 [源码事实 model.py:569-589, 601-611]
```python
scores = softplus(x).sqrt()           # sqrt(ln(1+e^x)) ≥ 0
if bias is not None:
    sel = scores + bias               # bias 仅影响 top-k 选择
indices = sel.topk(6)                 # noaux_tc: 选 6 专家
weights = scores.gather(indices)      # 用原(无 bias)分数作权重
weights /= weights.sum(-1, keepdim=True)
weights *= route_scale               # 1.5
# SwiGLU 限幅:
up   = clamp(w3(x), -10, 10)
gate = clamp(w1(x), max=10)           # 仅限上界
x = silu(gate) * up
```

### (d) 关键代码位置
- `sqrtsoftplus`: `model.py:575-576 F.softplus(scores).sqrt()`
- noaux_tc 偏置分离: `model.py:579-585`
- swiglu_limit: `model.py:605-607`

### (e) 输入输出 shape [源码事实]
```
scores: [b*s, 256]  -> weights [b*s, 6], indices [b*s, 6]
expert FFN: x[b*s,4096] -> inter 2048 -> 4096
```

### (f) 为什么这样实现 [推导]
- **sqrtsoftplus vs sigmoid**:sqrt(softplus) 对大输入不饱和(∼√x 增长,而 sigmoid→1),能表达更大权重差;在 0 点光滑非负;逐元素(解耦)后再归一化,选专家与定权分离 → 训练更稳。
- **noaux_tc bias 分离**(V3 沿用):`e_score_correction_bias` 只改 top-k **选择**不改 **权重**,负载均衡不污染路由概率。
- **swiglu_limit=10.0**:FP4 上限 6.0、FP8 上限 448。若不限幅,SwiGLU 的 `silu(gate)*up` 可产生远超 10 的异常值,使 FP4/FP8 量化饱和→质量崩。限幅把激活框在低比特动态范围内,是**低比特定点推理的必要护栏**。gate 仅限上界(负向 silu 已抑制),up 双向限。

### (g) 计算/显存影响 [推导]
- sqrtsoftplus 比 softmax 省一次全局归一化(逐元素);softplus+sqrt 可融合。
- swiglu_limit 增加两次 clamp,成本极低但显著提升量化稳定性。
- 256 专家中每 token 激活 6 + 1 共享 = 7,稀疏度高。

---

## 6. FP4 专家 + FP8 权重

### (a) 解决什么问题
304B 级 MoE 权重显存巨大;用 4 比特存专家、8 比特存稠密权重,大幅降显存并提速 GEMM。

### (b) 标准做法
BF16 权重(2 字节/元素);或 INT8/INT4 整数量化(需零点/对称量化)。

### (c) 实现细节 [源码事实 model.py:129-158, kernel.py:128-200,441-536, convert.py:11-52]
**FP4 专家 (e2m1)**:
```python
# convert.py FP4_TABLE: e2m1 取值集
FP4_TABLE = [0,0.5,1,1.5,2,3,4,6, 0,-0.5,-1,-1.5,-2,-3,-4,-6]
# 存储: weight [out, in//2] float4_e2m1fn_x2 (2 个 fp4 打包 1 字节, 沿 K)
# scale: [out, in//32] float8_e8m0fnu  (每 32 元素 1 个 power-of-2 缩放)
```
**FP8 稠密 (e4m3)**:
```python
# weight [out, in] float8_e4m3fn
# scale [out//128, in//128] e8m0fnu  (128×128 块)
```
**fp4_gemm 核** (kernel.py:441-515):
```
策略: FP8(act) × FP4(weight)
  load FP4 weight sub-block [128,32]
  cast FP4 -> FP32 -> FP8       # 无原生 FP4 张量核, 转 FP8 算
  FP8×FP8 GEMM on tensor cores
  accum += partial * scale_a(per128) * scale_b(per32)
```
**scale_fmt=ue8m0**:`fast_round_scale` 把动态 scale 舍入到最近 2 的幂(kernel.py:22-37,用 IEEE754 位运算),匹配 MX 微缩放标准。

### (d) 关键代码位置
- FP4/FP8 Linear: `model.py:129-158`
- act_quant (FP8 动态,ue8m0): `kernel.py:40-125`
- fp4_act_quant: `kernel.py:128-200`
- fp4_gemm: `kernel.py:441-536`
- FP4→FP8 无损转换(含 6-bit offset): `convert.py:17-52 cast_e2m1fn_to_e4m3fn`

### (e) 输入输出 shape [源码事实]
```
FP4 GEMM: A[M,K] fp8 + scale[M,K/128]; B[N,K] fp4(+scale[N,K/32]) -> C[M,N] bf16
专家 w1: [2048, 4096] fp4 = [2048, 2048] 字节 + scale[2048,128]
稠密 wq_b: [64*512, 1024] fp8 + scale[...]
```

### (f) 为什么这样实现 / MXFP4 vs FP4 [推导]+[源码事实]
- **expert_dtype="fp4" 即 MXFP4**:e2m1 元素 + e8m0(power-of-2)共享 scale 每 32 元素 = OCP **MX (Microscaling) FP4** 标准。通用 "FP4" 可指任意 4 比特浮点(如 NF4、e1m2);此处精确为 e2m1+e8m0=MXFP4。 [源码事实]
- **专家用更激进的 FP4,稠密用 FP8**:专家占参数主体(256×3 矩阵),FP4 省 2×;稠密投影用 FP8 保精度。
- **FP4→FP8 转换路径**(convert.py):无 FP4 张量核的硬件可无损转 FP8(6.0×2^6=384<448),把 per-32 FP4 scale 折成 per-128 FP8 scale + 6-bit offset。 [源码事实]
- **dynamic 激活量化**:运行时按 absmax 算 scale(非离线校准),scale_fmt=ue8m0 舍入 2 的幂以匹配 MX。

### (g) 计算/显存影响 [推导]
- 专家权重视存:FP4 ≈ BF16 的 1/4(+少量 scale)。整模 ~48 个 safetensors,首片仅 ~1GB。
- GEMM 实际跑 FP8×FP8(转码后),用 FP8 张量核,带宽节省带来 2× 速度。
- QAT(代码注释 "We performed QAT here")表明权重经量化感知训练,FP4 可用性靠 swiglu_limit + 训练补偿。

---

## 7. YaRN 长上下文

### (a) 解决什么问题
将 RoPE 训练长度(65536)外推至 1M,且外推后质量不崩。

### (b) 标准 RoPE
`freqs = base^(-2i/d)`,对每维用旋转角 `θ_t = t·freq`;位置外推直接用大 t 会失真。

### (c) 数学公式 [源码事实 model.py:205-235]
```
factor=16, original_max=65536, beta_fast=32, beta_slow=1
max_position = 65536 × 16 = 1048576  (1M)
freqs = 1 / base^(2i/d)
# 找 "快/慢" 纠正维度区间:
low  = floor(correction_dim(beta_fast=32))   # 高旋转数 -> 低维
high = ceil (correction_dim(beta_slow=1))    # 低旋转数 -> 高维
ramp = clamp((arange - low)/(high-low), 0, 1)
smooth = 1 - ramp
# 混合: 高频维外推(原 freq), 低频维插值(freq/factor)
freqs = freqs/factor·(1-smooth) + freqs·smooth
```
**压缩层独立 RoPE**:`compress_rope_theta=160000`(vs 基础 10000),压缩位置索引 = `原位置 // ratio`。 [源码事实]

### (d) 关键代码位置
- YaRN 预计算: `model.py:205-235 precompute_freqs_cis`
- 压缩层启用 YaRN: `model.py:481-485` (`compress_ratio!=0` 时 `original_seq_len=65536`;否则关 YaRN 用基础 theta)
- 压缩 RoPE: `model.py:370-372` (压缩位置取 `freqs_cis[::ratio]`)

### (e) 输入输出 shape [源码事实]
```
freqs_cis: [max_seq_len, rope_head_dim/2] complex64
q rope 维: [b, s, 64] (head_dim=512 中末 64 维用 RoPE)
```

### (f) 为什么这样实现 [推导]
- **beta_fast=32 / beta_slow=1**:beta_fast 大 ⇒ 更多维度判为"高频→外推"(保留原旋转,精细位置分辨率);beta_slow 小 ⇒ 极少维度全插值。整体偏向外推,保局部精度,靠训练适配。
- **压缩层用大 theta(160000)**:压缩 token 代表每 ratio(4 或 128)个原 token 的聚合,其"时间轴"被压缩 ratio 倍,用更大 base 拉开相位间隔,避免相邻压缩位置相位塌缩。
- **纯滑窗层关 YaRN**(model.py:484):滑窗只看 128 局部,无需长程外推,关 YaRN 反而更稳。
- **factor=16**:65536→1M 正好 16×。

### (g) 计算/显存影响 [推导]
- YaRN 仅改 freqs 预计算(一次性,lru_cache),零运行时开销。
- 配合 CSA/HCA,1M 上下文下显存仍可控(论文 10% KV)。
- RoPE 仅作用于每头末 64 维(rope_head_dim),512 维中 448 维无位置编码——降低长程位置噪声。

---

## 附:其他审计发现

- **Muon 优化器** [官方材料]:训练用 Muon(论文提及),与推理无关。
- **MoE 哈希路由** `n_hash_layers=3` [源码事实 model.py:561-564]:前 3 层 MoE 用 `tid2eid` 表按 token_id 直接查专家(预算路由),非 score 路由。与 hc_* 无关。
- **sample 用 Gumbel-max** [model.py:939-946]:`probs/exp(Exp(1)).argmax` 等价多项式采样但避 GPU→CPU 同步,提速。
- **encoding** [encoding/README.md]:DSML 工具调用格式、`<think>` 推理块、reasoning_effort(low/high/max)纯文本前缀控制;`<｜action｜>` 等快速指令 token。
- **`attn_sink`** [model.py:462, kernel.py:346]:每头可学习偏置,加入 softmax 分母,稳定首批 token 注意力(attention sink 现象)。
- **并行**:TP 全程 ColumnParallel/RowParallel + all_reduce;专家按 `n_routed_experts/world_size` 切分。
