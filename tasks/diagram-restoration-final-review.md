## Review

### 结论

- **暂不建议封板/推送。**
- 数量和内容层级已经真正恢复：6 张模型图、26 张原理卡、6 张工程图均直接可见，不再是摘要卡替代。
- 但 **Qwen、GLM、Kimi、DeepSeek 四张主结构图的箭头没有完整表达真实 DecoderLayer 执行顺序**。页面却称其为“完整结构图”，这是本轮用户核心诉求上的发布阻断。

### Correct

- **六模型图谱已真实恢复且未隐藏**
  - 六个模型均直接包含 `<figure>`、文字数据流和至少 3 张原理卡：`docs/index.html:172-230`
  - 核心图不在默认关闭的 `<details>` 内；`details` 只承载边界补充。
  - 六模型锚点索引完整：`docs/index.html:170`

- **六张工程图已恢复**
  - 总体边界、GLM、Kimi、DeepSeek、Qwen、PD 均直接展示：`docs/index.html:233-277`
  - 工程图正确区分 reference 与 deployment，例如 DeepSeek reference all-reduce 与可选 AllToAll：`docs/index.html:258-262`

- **高风险 shape 总体与 config/model-data 一致**
  - Qwen：hidden 5120、64 层、24Q/4KV、DeltaNet K=16×128/V=48×128、combined QKV=10240：`diagrams/model-qwen.mmd:2-25`、`configs/qwen3.6-27b-config.json:16-20,88-99`
  - GLM：78 层、Q LoRA 2048、KV 512+64、64×256 展开、21 Indexer：`diagrams/model-glm.mmd:4-31`、`configs/glm-5.2-config.json:17,108,195-222`
  - Kimi：69 KDA + 24 MLA、Q/K=192、V=128、optimized 512+64、AttnRes 8 次：`diagrams/model-kimi.mmd:6-40`、`configs/kimi-k3-config.json:26,50,59,181-201,246,266`
  - DeepSeek：43 层、64×512 Q、单 512 KV、2/21/20 压缩分布、DSpark 三入口分离：`diagrams/model-deepseek.mmd:5-59`、`configs/deepseek-v4-flash-config.json:13-35,66-67`

- **缓存口径清楚**
  - GLM reference 展开与 optimized 576 分离：`docs/index.html:206-209`
  - Kimi 两种 MLA 路径均声明 KDA 未完整计量：`docs/index.html:216-219`
  - DeepSeek raw quantized subtotal 未伪装成完整总量：`docs/index.html:226-229`
  - PD 图数值与当前计算层一致，并标明不完整项：`diagrams/system-pd.mmd:5-20`

- **渐进增强与无障碍基础正确**
  - 图容器可聚焦、可局部滚动，并有 alt、figcaption 和 HTML 文字等价内容：`docs/index.html:176-228`
  - 移动端溢出限制在 `.diagram-scroll`：`docs/assets/course.css:129-132`
  - `#model-*` 与 `#system-*` hash 正确映射顶部导航：`docs/assets/course.js:33-60`
  - 现有 Playwright、Axe、无 JS 和 Lighthouse 结果支持这些实现。

- **SVG 安全和预算门禁充分**
  - 禁止 script、事件属性、外链、foreignObject，并检查 viewBox 与暗色 cluster：`tests/site.test.mjs:42-58`
  - 单图、图总量和全站预算分别受限：`tests/site.test.mjs:154-169`
  - Pages 前置门禁也执行同类扫描：`scripts/check-site.mjs:15-25,66-81`

### Blocker

- **四张高级模型图没有画出真实的逐层主执行链，却被称为“完整结构图”。**

  1. **Qwen**
     - `STACK` 只包含 DeltaNet/GQA 子图，SwiGLU 被画在整个 64 层 stack 之后：`diagrams/model-qwen.mmd:7-31`
     - 视觉语义容易被读成“64 个 attention 层执行完后只做一次 MLP”，而 HTML 正文明确说每层都有 SwiGLU：`docs/index.html:197`
     - 修复：在 DecoderLayer 模板内画出  
       `Norm → selected attention → residual → Norm → SwiGLU → residual`，再表达 `[linear,linear,linear,full] ×16`。

  2. **GLM**
     - MLA、DSA、MoE 是三个并列且缺少主链连接的 subgraph；只有 `ROPE → STACK → Final`：`diagrams/model-glm.mmd:5-42`
     - 图没有表达 `hidden → MLA/DSA → residual → FFN/MoE → residual`。
     - 修复：增加明确 overview lane；DSA 只作为 mask/indices 输入 MLA，MoE 必须接在 attention residual 后。

  3. **Kimi**
     - 图把 93 层 attention stack、AttnRes、FFN/LatentMoE 串成先后阶段：`diagrams/model-kimi.mmd:6-39`
     - 实际上 Dense/MoE 属于每个 DecoderLayer，AttnRes 也穿插在层边界，不是 93 层 attention 完成后的单独尾部。
     - 修复：画出单层  
       `KDA/MLA → AttnRes application → Dense/MoE → residual`，并在外层标 93 层及 12 层 block residual 边界。

  4. **DeepSeek**
     - mHC、Attention、MoE 三个子图互不相连；只有 `HCEXP → MAIN → hc_head`：`diagrams/model-deepseek.mmd:5-42`
     - 缺失真实的 attention 与 FFN 两段 hc_pre/hc_post 顺序。
     - 修复：显式画出  
       `hc_pre(attn) → Norm → sparse attention → hc_post → hc_pre(ffn) → Norm → MoE → hc_post`。

- 生成的 SVG 会忠实保留上述语义问题，因此不是只改文字即可；需同时修改 MMD、重渲染 SVG，并补浏览器截图复核。

### High

- **超宽 PD 图被统一缩到过小尺寸，正文嵌入时可能不可读。**
  - 所有图统一 `min-width:760px`：`docs/assets/course.css:131`
  - PD 图逻辑宽度约 2628，HTML 声明 `2628×404`：`docs/index.html:275`、`docs/assets/diagrams/system-pd.svg:1`
  - SVG 根节点使用百分比宽度时，统一规则可能把 PD 图压到 760/1178px，内部 15px 图中文字相当于约 4–7px。
  - 修复建议：
    - 将 SVG 根节点写入基于 viewBox 的数值 width/height；或
    - 为 wide/ultrawide 图增加专用 class，PD 最小宽度至少约 1800–2200px；
    - Playwright 增加 `system-pd` 在 320px 和桌面的滚动宽度/截图断言，而不只测试模型图容器。

- **MMD 源与生成 SVG 没有一致性门禁。**
  - 当前测试读取 SVG 的安全性、数量和 HTML 引用，但完全不读取 `diagrams/*.mmd`：`tests/site.test.mjs:11-18,42-58`
  - `scripts/check-site.mjs:10-25` 同样只检查 SVG。
  - 因此修改 MMD 后忘记重渲染，或 SVG 与源漂移，仍会全绿。
  - 修复建议：增加离线 manifest，记录每个 MMD 与 SVG 的 SHA-256、viewBox 和生成器版本；零依赖检查 manifest。或者在维护门禁中运行固定版本渲染器并比较产物。

### Note

- Qwen combined-QKV 节点只写了 K=16×128、V=48×128，省略了 Q=16×128：`diagrams/model-qwen.mmd:10`
  - 总维度 10240 是对的，但“完整 shape”建议明确写成 `Q 16×128 + K 16×128 + V 48×128`。

- DeepSeek DSpark 图使用 `HF config → inference snapshot` 的箭头：`diagrams/model-deepseek.mmd:45-53`
  - 三者是不同入口，不是顺序执行或派生链。建议并列连到“版本边界”节点，避免图形语义与文字声明冲突。

- nanoGPT/MiniMind 的 residual 目前以“residual add”节点表示，但没有画旁路 skip arrow：`diagrams/model-nanogpt.mmd:9-20`、`diagrams/model-minimind.mmd:5-21`
  - 不构成数值错误，但若继续使用“完整结构图”措辞，建议补出 bypass 连线。

- 完成上述 Blocker 后，请由 supervisor 重跑：
  - `npm run verify`
  - 图谱 Playwright 矩阵，额外覆盖 `system-pd`
  - Axe 暗/亮主题
  - Lighthouse 移动/桌面
  - 12 张图的人工截图审查。
## 复核关闭（父会话，2026-08-20）

上表 Blocker / High / Note 已全部修复并由父会话复核关闭：

- 四张主链图按真实 DecoderLayer 顺序重画：Qwen/GLM/Kimi 为 `Norm -> attention -> residual -> Norm -> MLP/MoE -> residual`，DeepSeek 为双段 `hc_pre(attn) -> attention -> hc_post -> hc_pre(ffn) -> MoE -> hc_post`；渲染 SVG 中 `residual×2 / skip×2 / AttnRes×3 / hc_pre×2 / hc_post×2` 均已确认（`diagrams/model-*.mmd`）。
- High：PD 图改用 `diagram-scroll-ultrawide`（2200px），Chrome 实测桌面/320px 均有独立横向滚动容器且可键盘滚动，页面级溢出为 0。
- High：新增 `diagrams/manifest.json` + `scripts/update-diagram-manifest.mjs`，MMD/SVG SHA-256 与 viewBox 漂移门禁已接入 `tests/site.test.mjs` 与 `scripts/check-site.mjs`；当前 12/12 无漂移。
- Note：Qwen combined QKV 明确 `Q 16×128 + K 16×128 + V 48×128`；DSpark 改为三个并列证据入口；nanoGPT/MiniMind 补 skip 旁路箭头。
- 最终门禁：`npm run verify` 退出 0；Chrome 151 + Playwright 浏览器矩阵 PASS（12 图解码、Axe 亮/暗 0 violation、无 JS 12 图 + 12 份文字流程、0 控制台错误、0 第三方请求）。
