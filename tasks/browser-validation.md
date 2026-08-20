# Transformer 课程浏览器验收记录

- 日期：2026-08-20
- 方式：使用系统已有 Google Chrome，在 `/tmp/models-opencode-browser-validation` 临时安装验收工具；未写入项目依赖。
- 环境：[实验结果] Chrome 151.0.7922.140、Playwright Core 1.62.1、Axe Core 4.13.0、Lighthouse 13.4.1。

## Playwright 真实浏览器矩阵

[实验结果] 以下检查全部通过：

- 页面加载：全部运行资源同源，HTTP 200；第三方运行请求 0；console error 0；pageerror 0。
- 交互：主题切换及持久化、数据流按钮的鼠标/键盘操作、注意力拓扑、缓存计算器、通信计算器、答题反馈与 localStorage 恢复。
- 导航：hash 深链、`hashchange`、浏览器返回、模型手册映射、阅读进度。
- 渐进增强：禁用 JavaScript 后六课和六模型仍可读，增强控件保持禁用且不进入 Tab 顺序。
- 响应式：320×568、480×800、1440×900 下页面级横向溢出均为 0。
- 移动导航：320px 下 Lesson 05 活动项位于 nav 可视区内（nav 8–312px；active 118–186px）。
- 动效：`prefers-reduced-motion: reduce` 下根节点 scroll behavior 为 `auto`。
- 截图：暗色桌面、亮色桌面、320/480px、无 JS 及移动锚点视口均已生成并人工检查；未发现遮挡或页面级溢出。

## Axe

[实验结果] 暗色主题与亮色主题分别运行 WCAG 2 A/AA、2.1 AA、2.2 AA 规则：

- dark violations：0
- light violations：0

## Lighthouse 本地实验

本地服务为 `python3 -m http.server`，以下是 lab data，不是线上 GitHub Pages field data。

| 模式 | Performance | Accessibility | Best Practices | SEO | FCP | LCP | TBT | CLS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mobile | 100 | 100 | 100 | 100 | 1.130 s | 1.505 s | 0 ms | 0.00122 |
| Desktop | 100 | 100 | 100 | 100 | 0.303 s | 0.323 s | 0 ms | 0.00013 |

## 浏览器验收发现并修复

1. [源码事实] 通信带宽输入 `min=0.01` 但缺少匹配 step，默认值 400 被浏览器判为 step mismatch；已改为 `step=0.01`。
2. [源码事实] 有效利用率 `min=0.01, step=0.05, value=0.5` 同样无效；已改为 `step=0.01`。
3. [实验结果] 缺少 favicon 导致 Chrome 控制台 404；已添加本地 SVG favicon 并纳入资源预算。
4. [实验结果] 移动端深链时 `aria-current` 正确但活动导航项可能位于横向可视区外；已在 current 更新时自动横向居中。
5. [实验结果] 品牌链接的 `aria-label` 覆盖可见文字，触发 WCAG 2.5.3 label-content-name-mismatch；已移除覆盖性名称。
6. [实验结果] Lighthouse 无法在 `connect-src 'none'` 下读取同源 robots；已添加有效 `robots.txt`，并将 CSP 收敛为 `connect-src 'self'`，仍禁止第三方连接，静态检查继续禁止 fetch/XHR/WebSocket。

以上行为均增加了静态回归测试。

## 残余边界

- 未在 Safari/Firefox 上执行相同自动化矩阵。
- 未做 NVDA/VoiceOver 人工读屏；Axe 与语义检查不能替代真实辅助技术用户测试。
- 未部署，因此没有 GitHub Pages 线上网络、缓存或真实用户 Core Web Vitals 数据。
- 未做真实 GPU 性能基准。
