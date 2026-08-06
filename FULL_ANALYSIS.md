# 六模型深度源码分析与架构图例 · 统一总报告

> 访问日期 2026-08-01。所有结论标注 `[源码事实 文件:行号]` / `[官方材料]` / `[推导]` / `[未知]`。
> 四个大模型的完整审计报告分别存于 `research/{qwen3.6,glm-5.2,kimi-k3,deepseek-v4}/`。

---

## 统一目录

| # | 模型 | 类型 | 源码行数 | 架构图 | 关键函数表 | 与nanoGPT差异 |
|---|---|---|---|---|---|---|
| 1 | nanoGPT | 最小GPT教学 | 755 | §1.1 | §1.3 | (基线) |
| 2 | MiniMind | 现代小模型教学 | 445 | §2.1 | §2.3 | §2.4 |
| 3 | Qwen3.6-27B | 官方Dense混合 | 2275 | research/qwen3.6/ | research/qwen3.6/ | research/qwen3.6/ |
| 4 | GLM-5.2 | 官方MoE | 1400 | research/glm-5.2/ | research/glm-5.2/ | research/glm-5.2/ |
| 5 | Kimi-K3 | 官方MoE多模态 | 2916 | research/kimi-k3/ | research/kimi-k3/ | research/kimi-k3/ |
| 6 | DeepSeek-V4 | 官方MoE | 1795 | research/deepseek-v4/ | research/deepseek-v4/ | research/deepseek-v4/ |
| - | 跨模型对比 | - | - | §3 | - | - |

---

## §1 nanoGPT 深度分析

### §1.1 架构图

```mermaid
flowchart TB
    IN["input_ids (B,T) int64"] --> WTE["wte: Embedding(V, d)<br/>model.py:127"]
    POS["pos: arange(T)"] --> WPE["wpe: Embedding(block_size, d)<br/>model.py:128"]
    WTE --> ADD["tok_emb + pos_emb<br/>(B,T,d)"]
    WPE --> ADD
    ADD --> DROP["Dropout"]
    DROP --> BLK

    subgraph BLK["Block × n_layer (model.py:94-106)"]
        LN1["ln_1: LayerNorm(d, bias)<br/>model.py:98"]
        ATTN["CausalSelfAttention<br/>model.py:29-76"]
        RES1["+ 残差"]
        LN2["ln_2: LayerNorm(d, bias)"]
        MLP["MLP: c_fc(4d)->GELU->c_proj(d)<br/>model.py:78-92"]
        RES2["+ 残差"]
        LN1 --> ATTN --> RES1
        RES1 --> LN2 --> MLP --> RES2
    end

    BLK -->|"循环 n_layer 次"| BLK
    BLK --> LNF["ln_f: LayerNorm(d)"]
    LNF --> LH["lm_head: Linear(d, V, bias=False)<br/>model.py:133"]
    LH --> LOGITS["logits (B,T,V)"]
    LOGITS -->|"训练"| CE["cross_entropy(shift)<br/>model.py:187"]
    LOGITS -->|"推理"| LAST["只取最后位置<br/>model.py:190"]

    subgraph ATTN_DETAIL["CausalSelfAttention 内部 (model.py:52-76)"]
        CA["c_attn: Linear(d, 3d)<br/>合并QKV投影"]
        SPLIT["split -> q,k,v 各(B,T,d)"]
        HEADS["view+transpose -> (B,nh,T,hs)"]
        SDPA["SDPA(is_causal=True)<br/>或手写 softmax(QK^T/√d)V"]
        REASSEMBLE["transpose+view -> (B,T,d)"]
        CPROJ["c_proj: Linear(d,d)"]
        CA --> SPLIT --> HEADS --> SDPA --> REASSEMBLE --> CPROJ
    end
    ATTN -.-> ATTN_DETAIL
```

### §1.2 关键设计决策

| 组件 | nanoGPT 选择 | 行号 | 为什么 |
|---|---|---|---|
| Norm | LayerNorm(带bias) | model.py:18-27 | GPT-2 兼容;现代模型用 RMSNorm 去 bias |
| 位置编码 | 绝对可学习 wpe | model.py:128 | 简单;但无法外推超 block_size |
| Attention | MHA(c_attn 合并QKV) | model.py:35 | 一次 matmul 省 kernel 启动;现代用分离投影支持 GQA |
| 因果掩码 | SDPA is_causal | model.py:64 | PyTorch 2.0 FlashAttention 路径 |
| FFN | 2层 GELU(4×升维) | model.py:82-84 | 无门控;现代用 SwiGLU 3层门控 |
| 残差 | Pre-LN | model.py:104-105 | 现代做法;Post-LN 深层难收敛 |
| Weight Tying | wte.weight = lm_head.weight | model.py:138 | 省 V×d 参数+语义对齐 |
| 推理优化 | 只算最后位置 lm_head | model.py:190 | 省 (T-1)/T 的 lm_head 计算 |
| KV Cache | **无** | - | 教学极简;每步重算全部K/V |
| 优化器 | AdamW fused + weight decay分组 | model.py:263-287 | 2D+参数decay,1D不decay |
| lr schedule | cosine warmup | train.py:231-242 | warmup防初期冲坏;cosine平滑收敛 |
| 梯度累积 | loss/=grad_accum + no_sync | train.py:292-298 | 模拟大batch;DDP省通信 |
| MFU公式 | 6N + 12LHQT | model.py:296 | N=参数(FFN), LHT=attention(QK^T+AV) |

### §1.3 关键函数表

| 函数 | 文件:行号 | 输入→输出 shape | 核心作用 |
|---|---|---|---|
| GPT.forward | model.py:170-193 | (B,T)→logits(B,T,V)+loss | embedding→blocks→lm_head→CE |
| CausalSelfAttention.forward | model.py:52-76 | (B,T,d)→(B,T,d) | QKV投影→多头SDPA→输出投影 |
| MLP.forward | model.py:87-92 | (B,T,d)→(B,T,d) | 4×升维→GELU→降维 |
| Block.forward | model.py:103-106 | (B,T,d)→(B,T,d) | Pre-LN残差×2 |
| GPT.generate | model.py:305-329 | (B,T)→(B,T+N) | 自回归采样(无KV cache) |
| get_batch | train.py:116-131 | →x(B,T),y(B,T) | shift采样 |
| configure_optimizers | model.py:263-287 | →AdamW | decay分组+fused |
| estimate_mfu | model.py:289-303 | →float | 6N+12LHQT FLOPs估算 |

---

## §2 MiniMind 深度分析

### §2.1 架构图

```mermaid
flowchart TB
    IN["input_ids (B,T)"] --> EMB["embed_tokens: Embedding(V, d)<br/>model_minimind.py:201"]
    EMB --> DROP["Dropout"]

    DROP --> START["start_pos = past_kv[0][0].shape[1]<br/>从KV cache推断位置"]
    START --> ROPE["freqs_cos/sin 切片<br/>[start_pos:start_pos+T]"]
    ROPE --> BLK

    subgraph BLK["MiniMindBlock × n_layers (model_minimind.py:178-194)"]
        LN1["input_layernorm: RMSNorm(d)<br/>无bias"]
        ATTN["Attention (GQA+KV cache)<br/>model_minimind.py:91-134"]
        RES1["+ 残差"]
        LN2["post_attention_layernorm: RMSNorm"]
        FFN["FeedForward (SwiGLU) 或 MOEFeedForward<br/>model_minimind.py:136-176"]
        RES2["+ 残差"]
        LN1 --> ATTN --> RES1
        RES1 --> LN2 --> FFN --> RES2
    end

    BLK -->|"循环"| BLK
    BLK --> NORM["norm: RMSNorm"]
    NORM --> LH["lm_head: Linear(d, V, bias=False)<br/>tied with embed_tokens"]
    LH --> LOGITS["logits (B,T,V)"]
    LOGITS -->|"训练"| SHIFT["shift: logits[:-1] vs labels[1:]<br/>model_minimind.py:251"]
    LOGITS -->|"推理"| GEN["generate()"]

    subgraph ATTN_DETAIL["Attention 内部 (model_minimind.py:111-134)"]
        QPROJ["q_proj: Linear(d, nh*hd)<br/>8头"]
        KPROJ["k_proj: Linear(d, nkv*hd)<br/>4头(GQA!)"]
        VPROJ["v_proj: Linear(d, nkv*hd)<br/>4头"]
        QKNORM["q_norm/k_norm: RMSNorm(hd)<br/>QK-Norm"]
        ROPE_APPLY["apply_rotary_pos_emb(q,k)"]
        CACHE["cat([past_kv, k/v])<br/>KV Cache拼接"]
        REPEAT["repeat_kv(n_rep=2)<br/>4KV→8Q"]
        SDPA["SDPA(prefill) 或 手写(decode)"]
        OPROJ["o_proj: Linear(d, d)"]
        QPROJ --> QKNORM
        KPROJ --> QKNORM
        VPROJ --> ROPE_APPLY
        QKNORM --> ROPE_APPLY --> CACHE --> REPEAT --> SDPA --> OPROJ
    end
    ATTN -.-> ATTN_DETAIL

    subgraph FFN_DETAIL["SwiGLU FFN (model_minimind.py:136-146)"]
        GATE["gate_proj: Linear(d, d_ff)"]
        UP["up_proj: Linear(d, d_ff)"]
        ACT["act_fn(gate) * up<br/>SiLU门控"]
        DOWN["down_proj: Linear(d_ff, d)"]
        GATE --> ACT
        UP --> ACT
        ACT --> DOWN
    end
    FFN -.-> FFN_DETAIL

    subgraph MOE_DETAIL["MoE FFN (model_minimind.py:148-176)"]
        GATE_M["gate: Linear(d, 4) softmax"]
        TOPK["topk(k=1) 选1专家"]
        EXPERTS["4个FeedForward专家<br/>逐专家循环 index_add_"]
        AUX["aux_loss 负载均衡"]
        GATE_M --> TOPK --> EXPERTS
        EXPERTS -.-> AUX
    end
    FFN -.->|"use_moe=True"| MOE_DETAIL

    subgraph GEN_DETAIL["generate() (model_minimind.py:256-288)"]
        LOOP["for _ in max_new_tokens"]
        FWD["forward(input_ids[:,past_len:], past_kv)"]
        TEMP["logits/temperature"]
        REP["repetition_penalty"]
        TOPK["top_k截断"]
        TOPP["top_p nucleus"]
        SAMPLE["multinomial采样"]
        EOS["EOS检查+finished"]
        LOOP --> FWD --> TEMP --> REP --> TOPK --> TOPP --> SAMPLE --> EOS
        EOS -->|"未结束"| LOOP
    end
    GEN -.-> GEN_DETAIL
```

### §2.2 关键设计决策(与 nanoGPT 对比)

| 组件 | nanoGPT | MiniMind | 改进原因 |
|---|---|---|---|
| Norm | LayerNorm(bias) | **RMSNorm**(无bias) | 去mean省计算,无损 |
| 位置编码 | 绝对wpe(不可外推) | **RoPE+YaRN**(可外推32768) | 相对位置,支持长上下文 |
| Attention | MHA(Q=K=V等宽) | **GQA**(8Q/4KV) | KV cache省50% |
| QK-Norm | 无 | **有**(RMSNorm on head_dim) | 稳定attention分数 |
| KV Cache | **无** | **有**(past_key_value拼接) | 推理O(1) decode vs O(T) |
| FFN | 2层GELU(4×) | **SwiGLU**(3层门控,π×) | 门控增强表达力 |
| FFN比例 | 4× | **π≈3.14×** | SwiGLU 3矩阵参数平衡 |
| MoE | 无 | **可选**(4E/top-1+aux_loss) | 容量扩展 |
| 采样 | temp+top-k | **temp+top-k+top-p+rep_penalty** | 更精细采样控制 |
| EOS | 无 | **有**(finished提前停止) | 高效生成 |
| 流式 | 无 | **有**(streamer.put) | 交互体验 |
| shift | 数据层(x=data[i:i+T], y=data[i+1:]) | **loss层**(logits[:-1] vs labels[1:]) | HF惯例 |
| start_pos | 无(每次全前向) | **从past_kv推断** | KV cache位置对齐 |
| prefill/decode | 不区分 | **区分**(flash分支条件) | prefill用Flash,decode手写 |

### §2.3 关键函数表

| 函数 | 文件:行号 | 输入→输出 shape | 核心作用 |
|---|---|---|---|
| MiniMindForCausalLM.forward | :245-253 | (B,T)→logits+loss+aux_loss | backbone→lm_head→shift CE |
| MiniMindModel.forward | :209-232 | (B,T)→(B,T,d)+presents+aux_loss | embed→blocks→norm;start_pos推断 |
| MiniMindBlock.forward | :186-194 | (B,T,d)→(B,T,d) | Pre-RMSNorm残差×2 |
| Attention.forward | :111-134 | (B,T,d)→(B,T,d)+past_kv | GQA+QKNorm+RoPE+KVcache+SDPA |
| FeedForward.forward | :145-146 | (B,T,d)→(B,T,d) | SwiGLU: down(silu(gate)*up) |
| MOEFeedForward.forward | :156-176 | (B,T,d)→(B,T,d)+aux_loss | softmax路由+top1+逐专家+负载均衡 |
| RMSNorm.forward | :59-60 | (B,T,d)→(B,T,d) | fp32 RMS归一化+scale |
| precompute_freqs_cis | :62-78 | →(T,d)×2 | RoPE预计算+YaRN外推 |
| apply_rotary_pos_emb | :80-84 | (q,k)→(q,k) | rotate_half旋转 |
| repeat_kv | :86-89 | (B,T,nkv,hd)→(B,T,nh,hd) | GQA KV头广播 |
| generate | :256-288 | (B,T)→(B,T+N) | KV cache+temp+topk+topp+rep+eos+stream |
| get_batch | eval_llm.py | →inputs | tokenizer+chat_template |

### §2.4 Attention.forward 逐行解读 `[源码事实 model_minimind.py:111-134]`

```python
# :112-116  投影 + reshape
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
xq = xq.view(B, T, 8, 96)       # 8个Q头
xk = xk.view(B, T, 4, 96)       # 4个KV头 (GQA!)
xv = xv.view(B, T, 4, 96)

# :117  QK-Norm (nanoGPT没有)
xq, xk = self.q_norm(xq), self.k_norm(xk)

# :118-119  RoPE (nanoGPT用绝对wpe)
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

# :120-123  KV Cache (nanoGPT没有!)
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)  # 拼历史K
    xv = torch.cat([past_key_value[1], xv], dim=1)  # 拼历史V
past_kv = (xk, xv) if use_cache else None

# :124  GQA广播 (nanoGPT是MHA, n_rep=1)
xq = xq.transpose(1,2)                                    # (B,8,T,96)
xk = repeat_kv(xk, n_rep=2).transpose(1,2)                # (B,8,T,96)
xv = repeat_kv(xv, n_rep=2).transpose(1,2)

# :125-131  分支:Flash vs 手写
if flash and seq_len>1 and past_kv is None and mask全1:
    y = SDPA(xq, xk, xv, is_causal=True)     # prefill: FlashAttention
else:
    scores = (xq @ xk.T) / sqrt(96)           # decode或padding: 手写
    scores += causal_mask + padding_mask
    y = softmax(scores) @ xv

# :132-133  输出投影
y = y.transpose(1,2).reshape(B, T, d)
y = o_proj(y)
```

### §2.5 generate() 逐行解读 `[源码事实 model_minimind.py:256-288]`

```python
# :264  KV cache 的核心:只传新token
past_len = past_kv[0][0].shape[1] if past_kv else 0
outputs = self.forward(input_ids[:, past_len:], ...)  # 只传新token!

# :267  温度
logits = outputs.logits[:, -1, :] / temperature

# :268-270  repetition penalty (nanoGPT没有)
for i in range(B):
    seen = torch.unique(input_ids[i])
    logits[i, seen] = where(score>0, score/rp, score*rp)

# :271-272  top-k (nanoGPT有)
logits[logits < topk(logits, top_k)[0][...,-1,None]] = -inf

# :273-277  top-p nucleus (nanoGPT没有!)
sorted_logits, sorted_idx = torch.sort(logits, descending=True)
mask = cumsum(softmax(sorted_logits)) > top_p
mask[..., 1:] = mask[..., :-1].clone(); mask[..., 0] = 0  # 保留首个超阈值
logits[mask.scatter(1, sorted_idx, mask)] = -inf

# :278  采样
next_token = multinomial(softmax(logits), 1) if do_sample else argmax(logits)

# :279  EOS处理 (nanoGPT没有)
next_token = where(finished, eos_token, next_token)

# :280-281  拼接 + 更新cache
input_ids = cat([input_ids, next_token], dim=-1)
past_kv = outputs.past_key_values

# :283-285  EOS提前停止 (nanoGPT没有)
finished |= next_token.eq(eos_token_id)
if finished.all(): break
```

---

## §3 跨模型对比图

### §3.1 六模型架构总览

```mermaid
flowchart LR
    subgraph NANO["nanoGPT (124M)"]
        N1["wte+wpe"] --> N2["×12 Block<br/>MHA+GELU MLP"]
        N2 --> N3["lm_head(tied)"]
    end

    subgraph MINI["MiniMind (64M)"]
        M1["embed"] --> M2["×8 Block<br/>GQA+SwiGLU<br/>(+MoE可选)"]
        M2 --> M3["lm_head(tied)"]
    end

    subgraph QWEN["Qwen3.6 (27B)"]
        Q1["embed+ViT"] --> Q2["×64 Block<br/>48 Gated DeltaNet<br/>+16 Gated GQA<br/>+输出门控"]
        Q2 --> Q3["lm_head+MTP"]
    end

    subgraph GLM["GLM-5.2 (753B)"]
        G1["embed"] --> G2["3 Dense + 75 MoE Block<br/>MLA+DSA+IndexShare<br/>sigmoid路由256E/8选"]
        G2 --> G3["lm_head+MTP"]
    end

    subgraph KIMI["Kimi-K3 (2.8T)"]
        K1["embed+MoonViT"] --> K2["×93 Block<br/>69 KDA + 24 Gated MLA<br/>+AttnRes(12层)<br/>+LatentMoE 896E/16选"]
        K2 --> K3["lm_head(无MTP)"]
    end

    subgraph DS["DeepSeek-V4 (304B)"]
        D1["embed"] --> D2["×43 Block<br/>CSA+HCA交替<br/>+MQA(1KV)+mHC(4副本)<br/>+MoE 256E/6选 FP4<br/>+DSpark(3层草稿)"]
        D2 --> D3["lm_head+DSpark"]
    end
```

### §3.2 注意力机制演进图

```mermaid
graph TD
    subgraph 基线["全注意力基线"]
        MHA["MHA<br/>nanoGPT<br/>Q=K=V头数<br/>cache: O(H·T·d)"]
    end

    subgraph KV压缩["KV压缩路线(仍是精确注意力)"]
        GQA["GQA<br/>MiniMind<br/>KV头<Q头<br/>cache: O(Hkv·T·d)"]
        MQA["MQA<br/>DeepSeek-V4<br/>KV头=1<br/>cache: O(T·d)"]
        MLA["MLA<br/>GLM-5.2/Kimi-K3<br/>KV低秩压缩到潜变量<br/>cache: O(T·d_lora)"]
    end

    subgraph 线性["线性注意力路线(有损,cache恒定)"]
        DELTANET["Gated DeltaNet<br/>Qwen3.6<br/>delta rule + Mamba2门控<br/>cache: O(H·d²)恒定"]
        KDA["KDA<br/>Kimi-K3<br/>delta rule + safe gate<br/>cache: O(H·d²)恒定"]
    end

    subgraph 稀疏["稀疏注意力路线(精确,选择key)"]
        DSA["DSA<br/>GLM-5.2<br/>ReLU打分 + top-2048<br/>+IndexShare跨层复用"]
        CSA["CSA+HCA<br/>DeepSeek-V4<br/>4×/128×压缩池化<br/>+top-512索引"]
    end

    MHA -->|"减少KV头"| GQA
    GQA -->|"KV头=1"| MQA
    MHA -->|"KV低秩压缩"| MLA
    MHA -->|"状态递推替代"| DELTANET
    DELTANET -->|"变体"| KDA
    MHA -->|"top-k选择key"| DSA
    DSA -->|"多级压缩"| CSA
```

### §3.3 MoE 路由策略对比

```mermaid
graph LR
    subgraph MINI_M["MiniMind MoE"]
        MM1["softmax打分"] --> MM2["top-1选择"] --> MM3["aux_loss均衡"] --> MM4["4专家 BF16"]
    end

    subgraph GLM_M["GLM-5.2 MoE"]
        GM1["sigmoid打分"] --> GM2["noaux_tc<br/>(bias=0,无分组)"] --> GM3["L1归一×2.5"] --> GM4["256E/8选/1共享<br/>+3层Dense"]
    end

    subgraph KIMI_M["Kimi-K3 LatentMoE"]
        KM1["sigmoid打分"] --> KM2["noaux_tc<br/>+renorm"] --> KM3["潜空间7168→3584<br/>+RMSNorm"] --> KM4["896E/16选/2共享<br/>MXFP4 QAT"]
    end

    subgraph DS_M["DeepSeek-V4 MoE"]
        DM1["sqrtsoftplus<br/>√ln(1+e^x)"] --> DM2["noaux_tc<br/>+bias偏移"] --> DM3["归一化<br/>+swiglu_limit=10"] --> DM4["256E/6选/1共享<br/>FP4专家"]
    end
```

### §3.4 残差连接演进

```mermaid
graph TD
    RES1["标准残差<br/>nanoGPT/MiniMind/Qwen3.6/GLM-5.2<br/>x = x + sublayer(norm(x))<br/>信息逐层传递"]
    RES2["AttnRes<br/>Kimi-K3<br/>每12层存block_residual<br/>跨块softmax加权访问<br/>1维打分: Linear(7168→1)"]
    RES3["mHC<br/>DeepSeek-V4<br/>4份隐状态副本<br/>Sinkhorn 20次迭代→双随机矩阵<br/>Birkhoff多面体流形约束"]

    RES1 -->|"深层信息退化"| RES2
    RES1 -->|"副本坍缩风险"| RES3
```

### §3.5 推测解码对比

```mermaid
graph LR
    subgraph NONE["无推测解码"]
        NN["nanoGPT/MiniMind/Kimi-K3<br/>逐token生成"]
    end
    subgraph MTP["MTP(单层草稿)"]
        QM["Qwen3.6: 1层全注意力<br/>复用主模型embedding<br/>2草稿token"]
        GM["GLM-5.2: 1层<br/>+索引复用<br/>接受长度+20%"]
    end
    subgraph DSPARK["DSpark(多层草稿)"]
        DM["DeepSeek-V4: 3层草稿块<br/>block_size=5(真+噪声×4)<br/>深层注入(layer40-42)<br/>Markov bigram头<br/>confidence自评<br/>7草稿token"]
    end
    NONE --> MTP --> DSPARK
```

### §3.6 位置编码对比

```mermaid
graph TD
    WPE["绝对可学习<br/>nanoGPT<br/>wpe: Embedding(block_size, d)<br/>不可外推"]
    ROPE1["1D RoPE<br/>MiniMind<br/>相对旋转<br/>+YaRN外推32768"]
    MROPE["MRoPE 3D<br/>Qwen3.6<br/>[11,11,10]=T,H,W<br/>interleaved<br/>partial_rotary=0.25"]
    ROPE2["RoPE (MLA内)<br/>GLM-5.2<br/>interleave=true<br/>theta=8e6, 1M"]
    NOROPE["无RoPE<br/>Kimi-K3 MLA层<br/>use_nope=true<br/>位置由KDA承担"]
    ROPE3["YaRN<br/>DeepSeek-V4<br/>factor=16, 65536→1M<br/>+compress_rope_theta=160000"]
    YARN["YaRN<br/>MiniMind(可选)<br/>factor=16, 2048→32768"]

    WPE --> ROPE1
    ROPE1 --> MROPE
    ROPE1 --> ROPE2
    ROPE1 --> ROPE3
    ROPE1 -.-> YARN
    ROPE2 -.->|"变体"| NOROPE
```

### §3.7 核心维度对比表

| 维度 | nanoGPT | MiniMind | Qwen3.6 | GLM-5.2 | Kimi-K3 | DeepSeek-V4 |
|---|---|---|---|---|---|---|
| **层数** | 12 | 8 | 64 | 78 | 93 | 43 |
| **hidden** | 768 | 768 | 5120 | 6144 | 7168 | 4096 |
| **Q头/KV头** | 12/12 | 8/4 | 24/4(全)+16/48(线性) | 64/64(MLA) | 96/96(MLA) | 64/1(MQA) |
| **head_dim** | 64 | 96 | 256(全)/128(线性) | 192+64=256 | 128+64=192 | 512 |
| **FFN中间维** | 3072(4×) | 2432(π×) | 17408 | 2048×8E(MoE) | 3072×896E(潜3584) | 2048×6E(FP4) |
| **词表** | 50304 | 6400 | 248320 | 154880 | 163840 | 129280 |
| **max_pos** | 1024 | 32768 | 262144 | 1048576 | 1048576 | 1048576 |
| **参数量** | 124M | 64M | 27B | 753B | 2.8T | 304B |

---

## §4 详细审计报告索引

四个大模型的完整源码审计报告(含 Mermaid 架构图、Forward 调用图、逐函数分析表、与 nanoGPT 差异表)由 subagent 产出,存放位置:

| 模型 | 报告位置 | 源码位置 | 审计深度 |
|---|---|---|---|
| Qwen3.6 | 对话上方 subagent 输出 | research/qwen3.6/modeling_qwen3_5.py (2075行) | 全函数+Gated DeltaNet+MRoPE+MTP |
| GLM-5.2 | 对话上方 subagent 输出 | research/glm-5.2/modeling_glm_moe_dsa.py (827行) | MLA+DSA+IndexShare+MoE+modular diff |
| Kimi-K3 | 对话上方 subagent 输出 | research/kimi-k3/modeling_kimi_linear.py (1314行) | KDA+Gated MLA+AttnRes+LatentMoE+SiTU+视觉 |
| DeepSeek-V4 | 对话上方 subagent 输出 | research/deepseek-v4/model.py (961行) | CSA/HCA+MQA+mHC+DSpark+FP4+kernel |

每份报告包含:
1. Mermaid 架构图(组件级,含 shape)
2. Mermaid Forward 调用图(函数级,含行号)
3. 关键函数逐行分析表(函数名|行号|输入shape|输出shape|代码片段|作用)
4. 独有机制完整代码分析
5. 与 nanoGPT 的逐组件差异表
