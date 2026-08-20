# Transformer 原理可视化课程 · 六模型结构图谱

这是一个可直接由 GitHub Pages 发布的静态课程：面向初学者讲清 Transformer 数据流、注意力、KV Cache、MoE 与推理系统，同时为工程师保留 6 张完整模型结构图、26 张组件原理卡和 6 张并行/通信/PD 工程图。

## 课程入口

- 在线入口：仓库启用 Pages 后访问站点根路径。
- 本地入口：启动静态服务器后打开 <http://localhost:4173/>。
- 无 JavaScript 时静态正文、六章课程、12 张静态 SVG、12 份文字数据流与来源仍完整可读；增强控件保持禁用。JavaScript 只负责主题、进度、练习和计算器。

## 学习路径

1. Transformer 数据流与 shape
2. MHA / GQA / MQA / 线性 / 稀疏注意力
3. KV Cache、状态与 GB/GiB 换算
4. MoE top-k 路由与 AllToAll 通信
5. TP / EP / PD 分离与推测解码口径
6. 官方材料、源码、配置、推导和部署假设的证据边界

模型图谱覆盖 nanoGPT、MiniMind、Qwen3.6-27B、GLM-5.2、Kimi-K3 与 DeepSeek-V4-Flash-0731。每个模型同时提供可见结构图、HTML 文字流程、核心原理卡及 reference / optimized / deployment 边界；工程区恢复总体集群、GLM、Kimi、DeepSeek、Qwen 和 PD 分离六图。

## 证据等级

| 标记 | 含义 |
|---|---|
| `[官方材料]` | 官方模型卡、仓库说明或部署示例 |
| `[源码事实]` | 当前核验 revision 的实际执行路径与 tensor shape |
| `[配置值]` | 当前官方 config 快照中的声明值 |
| `[推导]` | 由公开 shape、dtype 与公式计算，不是性能实测 |
| `[部署假设]` | 可选 kernel、量化或拓扑；不冒充当前 reference 实现 |

证据快照核验日期为 **2026-08-20**。官方 revision、上游 commit 和本地对应文件列在课程的“来源与版本快照”区，也集中维护在 `docs/assets/model-data.js`。

## 本地验证

要求 Node.js 22；没有生产依赖，也不需要构建步骤。

```bash
npm test
npm run check
npm run verify
python3 -m http.server 4173 --directory docs
curl -I http://127.0.0.1:4173/
curl -I http://127.0.0.1:4173/assets/course.js
git diff --check
```

`npm test` 使用 Node 内置 `node:test`，覆盖关键缓存基准、模式/输入契约、通信单位、证据数据和静态页面语义。`npm run check` 做脚本语法、真实 hash、重复 id、CSP、危险运行时 API 与全静态资源预算检查。`npm run verify` 是本地与 Pages 工作流共用的完整门禁。

## 文件结构

```text
docs/
├── index.html                 # 课程、六模型图谱与六张工程图
└── assets/
    ├── diagrams/              # 12 张离线生成、静态安全扫描后的 SVG
    ├── model-data.js          # 六模型规格、cache 结构、来源和 revision
    ├── calculators.js         # cache / 状态 / 通信纯函数
    ├── course.js              # 渐进增强交互
    └── course.css             # 主题、图容器、响应式与无障碍样式
diagrams/                      # 12 张 SVG 的可维护 Mermaid 源，仅用于离线渲染
tests/
├── calculators.test.mjs       # 数字和口径回归测试
└── site.test.mjs              # HTML/CSS/JS 行为契约
scripts/check-site.mjs         # 零依赖静态检查
```

`.github/workflows/deploy.yml` 继续把 `docs/` 作为 GitHub Pages artifact 发布，站点不加载 CDN、第三方脚本、字体或样式。

## 维护规则与范围边界

- `model-data.js` 是交互计算的结构化规格与来源入口，`calculators.js` 是 byte 公式入口。为保证无 JavaScript 可读，静态正文、图注和文字数据流会保留必要的 shape 与配置；测试负责防止它们与计算契约冲突。
- `diagrams/*.mmd` 是可维护图源，`docs/assets/diagrams/*.svg` 是离线产物。部署不运行 Mermaid，也不加载 Mermaid CDN；`diagrams/manifest.json` 用 SHA-256 绑定源与产物。门禁要求恰好 12 张本地图，并扫描 script、事件属性、外链、`foreignObject`、viewBox、固有宽高、暗色 subgraph 和独立资源预算。
- 静态正文不复制精确 fallback byte 结果；新增实现模式必须声明 `supportedModes`，并标明 `reference`、`theoretical` 或 `deployment-assumption`。FP8/FP4 未计 scale、metadata 与对齐时只能显示 raw payload 小计，不能称完整总量。
- 通信必须区分 Gb/s 与 GB/s，并说明有效带宽、top-k 激活复制和未计入的开销。
- 本轮只校准静态课程与模型数据模块。`arch_*.md`、`FULL_ANALYSIS.md` 和模型实现源码尚未按同一数据源完整同步，可能保留旧口径；阅读长篇报告时应优先以课程证据快照和对应官方源码为准。
- 本轮已在本机 Chrome 151 上完成 Playwright 交互矩阵、Axe 暗/亮主题扫描与 Lighthouse 移动/桌面实验；结果见 `tasks/browser-validation.md`。这不等同于线上 GitHub Pages field data，也不代表 Safari/Firefox、NVDA/VoiceOver 或真实 GPU 实测。
