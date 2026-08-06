# 实现计划:六模型深度代码分析与架构图例

## 概述
对 6 个研究对象逐一进行深度源码分析,每个模型输出:架构图(Mermaid)、模块结构、Forward 调用图、关键函数逐行分析(行号+shape)、与 nanoGPT 基线的差异。最终产出统一入口和跨模型对比图。

## 任务列表

### Phase 1: 小模型深度分析(我亲自做,已有源码)
- [ ] T1: nanoGPT 深度分析(model.py 全函数 + 架构图 + 调用图)
- [ ] T2: MiniMind 深度分析(model_minimind.py 全函数 + 架构图 + 调用图 + 训练链)

### Phase 2: 大模型深度分析(subagent 并行,读 research/ 源码)
- [ ] T3: Qwen3.6 深度分析(modeling_qwen3_5.py 2075行)
- [ ] T4: GLM-5.2 深度分析(modeling_glm_moe_dsa.py 827行 + modular 402行)
- [ ] T5: Kimi-K3 深度分析(modeling_kimi_linear.py 1314行 + modeling_kimi_k3.py 1317行)
- [ ] T6: DeepSeek-V4 深度分析(model.py 961行 + kernel.py 536行)

### Phase 3: 统一整合
- [ ] T7: 统一入口目录 + 跨模型架构对比图 + 注意力/MoE/位置编码演进图

## 每个 subagent 的输出要求
1. Mermaid 架构图(组件级,含数据流)
2. Mermaid Forward 调用图(函数级,含 shape)
3. 关键函数表:函数名 | 文件:行号 | 输入shape | 输出shape | 作用
4. 与 nanoGPT 的逐组件差异表
5. 独有机制详解(公式+代码+shape)
