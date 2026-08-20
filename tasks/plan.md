# 实施计划：Transformer 原理可视化课程 v2

## 概述

把当前“六模型高密度审计报告”升级为三层产品：面向初学者的概念递进课程、面向工程师的六模型完整结构/原理图谱，以及并行/通信/PD 工程图。课程只能叠加在原有图谱之上，不能替代或删减核心架构图。第一优先级是修正计算口径和源码/部署实现混淆；第二优先级是信息层级、移动端与无障碍。保持 GitHub Pages 静态部署，不引入前端框架或运行时第三方依赖。

## 已批准的产品决策

- 受众：初学者主线 + 工程师深度展开。
- 范围：本轮完整改造 `docs/`，同步更新根 README；长篇 `arch_*.md` 后续分批校正。
- 视觉：深色科研编辑风格，支持亮/暗主题。
- 技术：纯 HTML/CSS/原生 JavaScript；GitHub Pages 零构建部署。
- 证据：事实区分 `[官方材料]`、`[源码事实]`、`[配置值]`、`[推导]`、`[部署假设]`。

## 架构决策

1. **单一数据源**：模型规格、缓存公式、来源与版本写入 `docs/assets/model-data.js`，页面和测试共用，避免不同章节数字漂移。
2. **纯函数计算**：缓存与通信计算放入 `docs/assets/calculators.js`；UI 只负责读取输入和渲染结果。
3. **渐进增强**：HTML 默认完整可读；JavaScript 加载成功后仅增强导航、主题、计算器、课程进度和交互图。
4. **移除运行时 Mermaid，但不移除图**：保留 12 份可维护图源，离线编译为仓库内静态 SVG；页面不加载 Mermaid/CDN。每图同时提供 HTML 文字数据流。
5. **课程与图谱叠加**：概念课程先讲心智模型；六模型完整结构图与工程图作为一级内容始终可见，只有额外证据深挖使用原生 `<details>`。
6. **证据快照固定**：记录 2026-08-20 核验日期、官方 URL、模型仓库 SHA/上游 commit 与本地快照说明。
7. **零依赖测试**：使用 Node 22 内置 `node:test`，不添加 npm 生产依赖。

## 验收契约

### 准确性

- 同一模型、上下文长度、实现模式和 dtype 在全站只有一个计算结果，误差不超过 1%。
- 测试至少覆盖：MiniMind 32K BF16、Qwen 1M GQA、GLM latent/reference、Kimi latent/reference、DeepSeek 分层压缩缓存。
- 明确区分当前 HF/reference cache 与理论/优化 kernel cache。
- Qwen `tie_word_embeddings=false`、MiniMind YaRN 默认关闭、KV Cache 非 O(1) decode、nanoGPT shift 位于数据层。
- 通信结果必须展示 GB/GiB、Gb/s/GB/s、有效带宽利用率和 top-k 路由假设。

### 教学与视觉

- 首页提供先修、学习目标、预计路径和六章课程。
- 至少三个真正可操作的教学组件：Transformer 数据流、注意力拓扑切换、KV/通信计算器。
- 每课包含“直觉 → shape → 公式/例子 → 理解检查 → 下一步”。
- 六个模型各有完整结构图、文字数据流、至少三张原理卡和可打开的具体来源；另有六张工程图。
- 图不只依赖颜色；复杂图拥有图注、文字等价说明、独立键盘滚动区和“打开原图”入口。

### 可访问性与渐进增强

- 禁用 JavaScript 后课程、六模型完整结构图、六张工程图及其文字流程仍可阅读。
- 无内联 `onclick`；导航使用真实锚点并支持 hash 深链。
- 所有交互支持键盘；焦点可见；折叠优先使用 `<details>/<summary>`。
- 表格包含 `<caption>`、`scope`；滚动容器有可访问名称。
- 支持 `prefers-reduced-motion`；活动文字对比达到 WCAG AA。

### 性能与部署

- 页面无第三方运行时请求；首屏不依赖 CDN。
- JavaScript总量 < 50KB、CSS < 75KB；12 张 lazy 静态 SVG 单张 < 85KB、合计 < 850KB；不加载字体或第三方图片。
- 同一交互只绑定一次，不做全页重复渲染。
- GitHub Pages 工作流继续发布 `docs/`。

### 验证命令

- `npm test`
- `npm run check`
- `python3 -m http.server 4173 --directory docs` 后用 `curl` 检查页面与静态资源
- `git diff --check`

## 任务列表

### Phase 1：准确性地基

#### Task 1：建立失败测试与事实数据契约

**验收标准**
- 新增 Node 内置测试并先证明旧站缺少目标模块/语义。
- 测试覆盖数据结构、来源字段、关键缓存基准与站点语义。
- 不添加第三方依赖。

**文件**：`package.json`、`tests/*.test.mjs`
**规模**：中
**依赖**：无

#### Task 2：实现模型数据与计算纯函数

**验收标准**
- 六模型规格和来源集中到一个模块。
- 缓存、状态、通信计算支持实现模式、dtype、GB/GiB。
- 所有 Task 1 测试转绿。

**文件**：`docs/assets/model-data.js`、`docs/assets/calculators.js`
**规模**：中
**依赖**：Task 1

### Checkpoint A

- `npm test` 通过。
- 已知错误数字拥有回归测试。

### Phase 2：课程壳与视觉系统

#### Task 3：重建语义化课程页面

**验收标准**
- 六章学习路径、六模型完整图谱和六张工程图均在无 JS 情况可读。
- 页面包含跳转链接、语义地标、来源区、理解检查、图注和文字数据流。
- 不再加载 Mermaid CDN；禁止用“移除运行时”作为删除图内容的理由。

**文件**：`docs/index.html`
**规模**：大，内容迁移型重写
**依赖**：Task 2

#### Task 4：实现科研编辑风格与响应式布局

**验收标准**
- 亮暗主题、层次化排版、课程进度、卡片/图表/表格视觉一致。
- 320px 至桌面布局无页面级横向溢出。
- 焦点、对比度、reduced-motion 和 44px 点击目标有明确样式。

**文件**：`docs/assets/course.css`
**规模**：中
**依赖**：Task 3

#### Task 5：实现渐进增强交互

**验收标准**
- 主题、hash 导航、当前章节、课程进度和练习反馈可用。
- 使用事件监听而非内联处理器。
- localStorage 不可用时静默退化且不锁内容。

**文件**：`docs/assets/course.js`
**规模**：中
**依赖**：Task 3

### Checkpoint B

- 静态结构测试与脚本语法检查通过。
- 禁用 JS 的 HTML 仍包含完整课程和参考内容。

### Phase 3：教学可视化

#### Task 6：Transformer 数据流与注意力拓扑实验

**验收标准**
- 可逐步查看 token → embedding → attention → FFN → logits 的 shape。
- 可切换 MHA/GQA/MQA/线性/稀疏注意力，解释保留与共享的信息。
- 键盘可切换，状态有文本说明。

**文件**：`docs/index.html`、`docs/assets/course.js`、`docs/assets/course.css`
**规模**：中
**依赖**：Tasks 3–5

#### Task 7：KV Cache 与通信计算实验

**验收标准**
- 模型、上下文、dtype、reference/optimized 模式可交互选择。
- 结果显示组成、公式、假设和来源类型。
- 通信计算区分 Gb/s 与 GB/s，并显示理论下限而非伪精确实测值。

**文件**：`docs/index.html`、`docs/assets/course.js`
**规模**：中
**依赖**：Task 2

#### Task 8：六模型证据化案例手册

**验收标准**
- 每个模型呈现“解决什么问题、关键替换、参数、缓存口径、来源”。
- 修复审计中的 P0/P1 数字与概念错误。
- 推测、部署假设和当前源码事实不混写。

**文件**：`docs/index.html`、`docs/assets/model-data.js`
**规模**：大，内容校准型
**依赖**：Tasks 2–7

### Checkpoint C

- 所有计算测试和站点检查通过。
- 页面不存在旧冲突数字或无解释口径。

### Phase 4：文档、发布与复核

#### Task 9：更新入口文档与维护说明

**验收标准**
- README 说明课程入口、证据等级、核验日期、命令与文件结构。
- 明确长篇报告尚未全部同步校正的边界。

**文件**：`README.md`
**规模**：小
**依赖**：Task 8

#### Task 10：独立审查、修复与最终验证

**验收标准**
- 分别完成准确性、UX/无障碍、代码质量/性能审查。
- 所有阻断项与必修项关闭，或明确记录无法在本环境验证的风险。
- 最终 `npm test`、`npm run check`、静态服务器探测和 `git diff --check` 通过。

**依赖**：Tasks 1–9

## 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 本机浏览器实验不等于线上 field data | 无法证明真实用户网络与设备上的 Core Web Vitals | 已用现有 Chrome 临时搭建 Playwright/Axe/Lighthouse 验收并人工检查截图；上线后再补 Pages field data |
| 优化 kernel 未全部纳入仓库 | 缓存结论可能被误写为源码事实 | 分开 reference、theoretical、runtime 三种模式；无源码即标部署假设 |
| 现有 `docs/index.html` 近 100KB | 重写 diff 较大 | 拆分数据、计算、样式和交互模块；评审按轴进行 |
| 长篇 `arch_*.md` 仍含旧口径 | 读者可能混淆 | README 明示校准范围；下一批按相同数据源迁移 |

## 非目标

- 本轮不修改模型实现源码或权重配置。
- 本轮不完整重写四份 `arch_*.md` 与 `FULL_ANALYSIS.md`。
- 不声称完成真实 GPU 性能基准、线上 GitHub Pages field data、Safari/Firefox、NVDA 或 VoiceOver 实测；本轮仅完成本机 Chrome 的 Playwright/Axe/Lighthouse 实验。
- 不提交、推送或部署，除非用户另行授权。
