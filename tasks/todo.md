# Transformer 原理可视化课程 v2 · 执行清单

## Phase 1：准确性地基

- [x] T1 建立 Node 内置测试与失败基线
- [x] T2 建立六模型单一事实数据源
- [x] T3 实现缓存/状态/通信纯函数
- [x] Checkpoint A：关键数字测试通过

## Phase 2：课程壳与视觉系统

- [x] T4 重建无 JS 可读的语义化课程页
- [x] T5 实现科研编辑风格、亮暗主题与响应式
- [x] T6 实现 hash 导航、进度和键盘交互
- [x] Checkpoint B：结构、语法、无障碍静态检查通过

## Phase 3：教学可视化

- [x] T7 Transformer shape 数据流实验
- [x] T8 MHA/GQA/MQA/线性/稀疏注意力切换
- [x] T9 KV Cache 与通信计算器
- [x] T10 六模型证据化案例手册
- [x] Checkpoint C：数字单一来源且旧冲突清零

## Phase 4：发布准备

- [x] T11 更新 README 与维护边界
- [x] T12 准确性独立复核（父会话完成并综合问题）
- [x] T13 UX/无障碍独立复核（父会话完成并综合问题）
- [x] T14 代码质量/性能独立复核（父会话完成并综合问题）
- [x] T15 修复评审问题并完成最终验证
- [x] T16 纠正信息架构偏差：恢复 6 张模型结构图、26 张原理卡与 6 张工程图
- [x] T17 将 Mermaid 仅用于离线图源，部署静态 SVG；补齐文字流程、移动滚动与安全门禁
- [x] 写入者实现后自检（不替代独立审查）

## 最终验证

- [x] `npm test`
- [x] `npm run check`
- [x] 静态服务器 + `curl` 探测成功
- [x] `git diff --check`
- [x] 无第三方运行时请求（静态扫描）
- [x] 无 JavaScript 时正文完整可读（结构测试 + Chrome 禁用 JS 实测）
- [x] Chrome 151 + Playwright：交互、键盘、hash/history、持久化、320/480/1440px 与截图检查通过
- [x] 12 张 SVG 全部解码；320px 图容器可键盘横向滚动且页面级溢出为 0；无 JS 仍有 12 图与 12 份文字流程
- [x] Axe 暗/亮主题 WCAG 扫描：0 violation
- [x] Lighthouse 图谱恢复后移动/桌面：Performance / Accessibility / Best Practices / SEO 均为 100
- [x] 残余风险已记录：未做线上 Pages field data、Safari/Firefox、NVDA/VoiceOver 或 GPU 实测
