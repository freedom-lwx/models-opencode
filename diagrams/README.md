# 静态图源

这里的 12 个 `.mmd` 文件是模型结构图和工程图的可维护源。GitHub Pages 不运行 Mermaid；部署的是 `docs/assets/diagrams/*.svg`。

离线更新方式：

```bash
# Mermaid CLI 11 与 SVGO 只需存在于维护环境的 PATH，不写入生产依赖
bash scripts/render-diagrams.sh
npm run verify
```

`mermaid-config.json` 禁用 HTML labels，`diagram.css` 修正 Mermaid 11 path 型 subgraph 和 edge label 的暗色主题。`scripts/check-site.mjs` 会拒绝 script、事件属性、外链、`foreignObject`、缺失 viewBox、浅色 cluster 回归和超预算 SVG。

图的证据约定：实线表示当前 config/source 路径；虚线必须明确标为 optimized、theoretical 或 deployment。修改 shape 前应先核对对应 config 与源码，不能从旧图抄数。
