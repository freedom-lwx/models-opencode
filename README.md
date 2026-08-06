# 大模型算法与推理工程 · 六模型源码深度审计

> 访问日期:2026-08-01。所有结论标注 `[源码事实 文件:行号]` / `[官方材料]` / `[推导]` / `[未知]`。

## 研究对象

| # | 模型 | 类型 | 参数量 |
|---|---|---|---|
| 1 | [nanoGPT](https://github.com/karpathy/nanoGPT) | 最小化 GPT 教学 | 124M |
| 2 | [MiniMind](https://github.com/jingyaogong/minimind) | 现代小模型教学(Dense+MoE) | 64M/198M |
| 3 | [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | 官方 Dense 混合注意力 | 27B |
| 4 | [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | 官方 MoE(MLA+DSA+IndexShare) | 753B |
| 5 | [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | 官方 MoE 多模态(KDA+Gated MLA) | 2.8T |
| 6 | [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 官方 MoE(CSA/HCA+MQA+mHC+DSpark) | 304B |

## 文件结构

```
├── FULL_ANALYSIS.md          # 统一总报告:nanoGPT+MiniMind 架构图+代码分析 + 六模型对比图
├── arch_qwen3.6.md           # Qwen3.6 完整架构图+函数表+代码分析(414行)
├── arch_glm5.2.md            # GLM-5.2 完整架构图+函数表+代码分析(962行)
├── arch_kimi_k3.md           # Kimi-K3 完整架构图+函数表+代码分析(1475行)
├── arch_deepseek_v4.md       # DeepSeek-V4 完整架构图+函数表+代码分析(782行)
├── configs/                  # 六模型 config.json(含 DeepSeek-V4 四个变体)
│   ├── glm-5.2-config.json
│   ├── kimi-k3-config.json
│   ├── qwen3.6-27b-config.json
│   ├── deepseek-v4-flash-config.json         # 0731
│   ├── deepseek-v4-flash-preview-config.json # Preview
│   ├── deepseek-v4-flash-dspark-config.json  # DSpark
│   └── deepseek-v4-pro-config.json           # Pro
├── research/                 # 四个大模型的源码(subagent 下载)
│   ├── qwen3.6/              # modeling_qwen3_5.py (2075行) + config + vllm实现
│   ├── glm-5.2/              # modeling_glm_moe_dsa.py (827行) + modular (402行)
│   ├── kimi-k3/              # modeling_kimi_linear.py (1314行) + modeling_kimi_k3.py (1317行)
│   └── deepseek-v4/          # model.py (961行) + kernel.py (536行) + convert.py
└── tasks/
    └── plan.md               # 实施计划
```

## 阅读顺序

1. `FULL_ANALYSIS.md` - 统一入口:nanoGPT/MiniMind 架构图 + 六模型对比
2. `configs/` - 对照看六个模型的架构参数
3. `arch_qwen3.6.md` / `arch_glm5.2.md` / `arch_kimi_k3.md` / `arch_deepseek_v4.md` - 逐个模型深挖
4. `research/` - 对应源码,用 grep 定位关键函数

## 核心发现

- **GLM-5.2** 继承 DeepSeek-V3.2,唯一原创是 IndexShare(跨 4 层复用 DSA 索引,省 73% 计算)
- **Kimi-K3** 最激进组合:69 KDA + 24 Gated MLA + AttnRes + LatentMoE + SiTU + 原生 MXFP4 QAT
- **DeepSeek-V4** Flash->0731 架构唯一变化是加 DSpark 模块;agentic 提升(DeepSWE 7.3->54.4)来自后训练
- **Qwen3.6** 用 75% Gated DeltaNet(线性注意力)+ 25% Gated GQA 混合,两套输出门控

## License

分析内容为原创。源码文件归各自原作者所有(MIT/Apache-2.0)。
