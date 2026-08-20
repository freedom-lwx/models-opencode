import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';

const url = (path) => new URL(`../${path}`, import.meta.url);
const rawHtml = await readFile(url('docs/index.html'), 'utf8');
const html = rawHtml.replace(/<!--[\s\S]*?-->/g, '');
const course = await readFile(url('docs/assets/course.js'), 'utf8');
const css = await readFile(url('docs/assets/course.css'), 'utf8');
const modelData = await readFile(url('docs/assets/model-data.js'), 'utf8');
const calculators = await readFile(url('docs/assets/calculators.js'), 'utf8');
const modelDiagramIds = ['nanogpt', 'minimind', 'qwen', 'glm', 'kimi', 'deepseek'];
const systemDiagramIds = ['overview', 'glm-tp-ep', 'kimi-tp-ep', 'deepseek-tp-ep', 'qwen-tp', 'pd'];
const diagramFiles = [
  ...modelDiagramIds.map((id) => `docs/assets/diagrams/model-${id}.svg`),
  ...systemDiagramIds.map((id) => `docs/assets/diagrams/system-${id}.svg`),
];
const diagramManifest = JSON.parse(await readFile(url('diagrams/manifest.json'), 'utf8'));
const sha256 = (content) => createHash('sha256').update(content).digest('hex');

const segment = (start, end) => html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start)));

test('page is progressive HTML with six lessons and six model references', () => {
  for (let index = 1; index <= 6; index += 1) assert.match(html, new RegExp(`<section\\b[^>]+id="lesson-${index}"`));
  for (const id of ['nanogpt', 'minimind', 'qwen', 'glm', 'kimi', 'deepseek']) assert.match(html, new RegExp(`<article\\b[^>]+id="model-${id}"`));
  assert.doesNotMatch(html, /class="[^"]*\bsec\b[^"]*"/);
});

test('six model dossiers restore architecture figures, transcripts, and principle cards', () => {
  for (const id of modelDiagramIds) {
    const article = segment(`id="model-${id}"`, '</article>');
    assert.match(article, new RegExp(`<figure\\b[^>]+class="architecture-figure"[^]*src="\\./assets/diagrams/model-${id}\\.svg"`));
    assert.match(article, /<figcaption>/);
    assert.match(article, /class="diagram-transcript"/);
    assert.ok((article.match(/class="principle-card"/g) || []).length >= 3, `${id} needs at least three principle cards`);
    if (id === 'qwen' || id === 'kimi') assert.match(article, /class="diagram-scroll diagram-scroll-ultrawide"/);
    if (id === 'glm' || id === 'deepseek') assert.match(article, /class="diagram-scroll diagram-scroll-wide"/);
  }
});

test('engineering reference restores six local system figures', () => {
  const systems = segment('id="systems"', '</section>');
  for (const id of systemDiagramIds) assert.match(systems, new RegExp(`src="\\./assets/diagrams/system-${id}\\.svg"`));
  assert.equal((systems.match(/<figure\b[^>]+class="architecture-figure/g) || []).length, 6);
  assert.match(systems, /class="diagram-scroll diagram-scroll-ultrawide"[^>]+六模型 PD 分离/);
  assert.match(css, /\.diagram-scroll-wide img\s*\{[^}]*min-width:\s*1500px/);
  assert.match(css, /\.diagram-scroll-ultrawide img\s*\{[^}]*min-width:\s*2200px/);
  assert.match(html, /<a href="#systems">工程图<\/a>/);
});

test('diagram SVGs are static, local, accessible, and safely embeddable', async () => {
  for (const file of diagramFiles) {
    const svg = await readFile(url(file), 'utf8');
    assert.match(svg, /<svg\b[^>]+viewBox=/, `${file} needs viewBox`);
    assert.doesNotMatch(svg, /<script\b|<foreignObject\b|\son[a-z]+\s*=|(?:href|src)=["']https?:|url\(https?:/i, `${file} must be inert and local`);
    for (const cluster of svg.matchAll(/class="cluster"[^>]*><path\b([^>]*)>/g)) {
      const inlineDark = /fill:#111821!important/.test(cluster[1]);
      const stylesheetDark = /\.cluster path\{fill:#111821!important/.test(svg);
      assert.ok(inlineDark || stylesheetDark, `${file} cluster path must stay dark`);
    }
    const basename = file.slice(file.lastIndexOf('/') + 1);
    const escapedBasename = basename.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const imagePattern = new RegExp(`<img\\b(?=[^>]*src="\\./assets/diagrams/${escapedBasename}")(?=[^>]*alt="[^"]+")(?=[^>]*width="\\d+")(?=[^>]*height="\\d+")(?=[^>]*loading="lazy")[^>]*>`);
    assert.match(html, imagePattern);
  }
});

test('diagram manifest binds every MMD source to its SVG and intrinsic aspect ratio', async () => {
  assert.notEqual(diagramManifest.renderer.mermaidCli, 'unknown');
  assert.notEqual(diagramManifest.renderer.svgo, 'unknown');
  assert.deepEqual(diagramManifest.files.map(({ name }) => name), diagramFiles.map((file) => file.match(/([^/]+)\.svg$/)[1]).sort());
  for (const entry of diagramManifest.files) {
    const [source, svg] = await Promise.all([
      readFile(url(`diagrams/${entry.name}.mmd`)),
      readFile(url(`docs/assets/diagrams/${entry.name}.svg`)),
    ]);
    assert.equal(sha256(source), entry.sourceSha256, `${entry.name} MMD drift`);
    assert.equal(sha256(svg), entry.svgSha256, `${entry.name} SVG drift`);
    const svgText = svg.toString('utf8');
    assert.equal(svgText.match(/viewBox="([^"]+)"/)?.[1], entry.viewBox);
    const [, , viewWidth, viewHeight] = entry.viewBox.split(/\s+/).map(Number);
    const imageTag = html.match(new RegExp(`<img\\b[^>]*src="\\./assets/diagrams/${entry.name}\\.svg"[^>]*>`))?.[0] ?? '';
    const width = Number(imageTag.match(/\bwidth="(\d+)"/)?.[1]);
    const height = Number(imageTag.match(/\bheight="(\d+)"/)?.[1]);
    assert.ok(width > 0 && height > 0, `${entry.name} intrinsic dimensions missing`);
    const aspectError = Math.abs((width / height) / (viewWidth / viewHeight) - 1);
    assert.ok(aspectError < 0.002, `${entry.name} intrinsic aspect ratio drift`);
  }
});

test('strict CSP and local runtime resources are declared', () => {
  assert.match(html, /<meta\b[^>]+http-equiv="Content-Security-Policy"/i);
  for (const directive of ["default-src 'self'", "script-src 'self'", "style-src 'self'", "connect-src 'self'", "object-src 'none'", "base-uri 'none'"]) assert.ok(html.includes(directive));
  assert.doesNotMatch(html, /connect-src[^;]*https?:/);
  assert.doesNotMatch(html, /<(?:script|link)\b[^>]+(?:src|href)=["']https?:\/\//i);
  assert.match(html, /<link\b[^>]+href="\.\/assets\/course\.css"/);
  assert.match(html, /<link\b[^>]+rel="icon"[^>]+href="\.\/assets\/favicon\.svg"/);
  assert.match(html, /<script\b[^>]+src="\.\/assets\/course\.js"><\/script>/);
});

test('visible brand label remains in its accessible name and robots policy is valid', async () => {
  assert.match(html, /<a class="brand" href="#top">TRANSFORMER \/ LAB NOTES<\/a>/);
  const robots = await readFile(url('docs/robots.txt'), 'utf8');
  assert.equal(robots, 'User-agent: *\nAllow: /\n');
});

test('no-JS fallback keeps prose readable while every enhancement control starts disabled', () => {
  assert.match(html, /<noscript>[^]*JavaScript[^]*正文[^]*<\/noscript>/);
  assert.match(html, /id="theme-toggle"[^>]+disabled/);
  assert.match(html, /id="flow-next"[^>]+disabled/);
  assert.equal((segment('id="flow-steps"', '</ol>').match(/<button\b[^>]+disabled/g) || []).length, 5);
  assert.equal((segment('id="topology-lab"', '</fieldset>').match(/<input\b[^>]+disabled/g) || []).length, 5);
  for (const id of ['cache-calculator', 'communication-calculator']) {
    const form = segment(`id="${id}"`, '</form>');
    const controls = [...form.matchAll(/<(?:input|select|button)\b[^>]*>/g)].map((match) => match[0]);
    assert.ok(controls.length > 0 && controls.every((tag) => /\bdisabled\b/.test(tag)), `${id} controls must default disabled`);
    assert.match(form, /role="status"/);
  }
  for (const form of html.matchAll(/<form\b[^>]+class="knowledge-check"[^]*?<\/form>/g)) {
    const controls = [...form[0].matchAll(/<(?:input|button)\b[^>]*>/g)].map((match) => match[0]);
    assert.ok(controls.every((tag) => /\bdisabled\b/.test(tag)));
  }
});

test('dataflow uses native buttons and progress exposes synchronized semantics', () => {
  const flow = segment('id="flow-steps"', '</ol>');
  assert.equal((flow.match(/<button\b/g) || []).length, 5);
  assert.doesNotMatch(flow, /role="button"|tabindex=/);
  assert.match(html, /<progress\b[^>]+id="course-progress"[^>]+max="100"[^>]+value="0"/);
  assert.match(course, /requestAnimationFrame/);
  assert.match(course, /progress\.value\s*=/);
  assert.match(course, /#models/);
  assert.match(course, /location\.hash/);
  assert.match(course, /hashchange/);
  assert.match(course, /nav\.scrollTo/);
  assert.match(course, /scrollIntoView\(\{ block: 'start'/);
  assert.match(course, /addEventListener\('load'/);
  assert.match(course, /comparison/);
  assert.match(course, /sources/);
});

test('each setup enables only its own complete component and has null guards', () => {
  for (const name of ['setupTheme', 'setupNavigation', 'setupDataflow', 'setupTopology', 'setupCacheCalculator', 'setupCommunicationCalculator', 'setupChecks']) assert.match(course, new RegExp(`function ${name}\\(`));
  assert.ok((course.match(/enableControls\(/g) || []).length >= 6);
  assert.match(course, /if \(!form \|\| !output/);
  assert.match(course, /if \([^\n]*!memory[^\n]*!description/);
  assert.match(course, /supportedModes/);
  assert.match(course, /option\.disabled/);
});

test('restored exercise completion is visible, textual, and live', () => {
  assert.match(course, /completion-state/);
  assert.match(course, /已完成/);
  assert.match(course, /aria-live/);
  assert.match(css, /\.completion-state/);
  assert.match(css, /\.complete/);
});

test('accessibility styling includes 44px theme target, no rotated mobile prose, and AA light derived badge', () => {
  assert.match(css, /:focus-visible/);
  assert.match(css, /prefers-reduced-motion/);
  assert.match(css, /prefers-color-scheme:\s*light/);
  assert.match(css, /\.quiet-button[^}]*min-height:\s*44px/s);
  assert.doesNotMatch(css, /\.system-map i\s*\{[^}]*transform:\s*rotate/s);
  assert.match(css, /--derived:\s*#[0-9a-f]{6}/i);
  assert.equal((css.match(/--accent:\s*#006b61/g) || []).length, 2);
  assert.match(css, /\.badge\.derived\s*\{\s*color:\s*var\(--derived\)/);
  assert.match(html, /有效利用率（0–1）/);
  assert.match(html, /id="comm-bandwidth"[^>]+step="0\.01"[^>]+value="400"/);
  assert.match(html, /id="comm-efficiency"[^>]+min="0\.01"[^>]+max="1"[^>]+step="0\.01"[^>]+value="0\.5"/);
});

test('semantic navigation, tables, status regions, and textual accuracy caveats remain', () => {
  assert.match(html, /<main\b[^>]+id="main-content"/);
  assert.match(html, /<nav\b[^>]+aria-label=/);
  assert.ok((html.match(/<caption>/g) || []).length >= 2);
  assert.ok((html.match(/scope="col"/g) || []).length >= 2);
  for (const id of ['flow-description', 'topology-description', 'cache-result', 'communication-result']) assert.match(html, new RegExp(`id="${id}"[^>]+role="status"`));
  for (const phrase of ['YaRN 默认关闭', '数据加载阶段', 'tie_word_embeddings=false', '当前 HF/reference', 'Indexer cache 单列', '7 speculative tokens', 'n_mtp_layers=3', 'Gb/s', 'GB/s', '不会自动减少激活通信']) assert.ok(html.includes(phrase), `missing phrase: ${phrase}`);
  assert.doesNotMatch(html, /512 维无 RoPE latent|本地 config 是 1 个 MTP block|65\.536 GB|150\.995 MB|1\.475 TB|2\.771 GB|5\.542 GB/);
  assert.doesNotMatch(html, /两份 modeling 源码相同/);
  assert.match(html, /DeepSeek-V4-Flash-0731\/blob\/7872f01b1d1fe23eabc4c98b48bffcef5a386062\/README\.md/);
});

test('all executable modules reject network and unsafe HTML APIs without whole-file exemptions', () => {
  const forbidden = /\b(?:fetch|XMLHttpRequest|WebSocket|EventSource|sendBeacon|document\.write)\b|\.(?:innerHTML|outerHTML)\s*=|insertAdjacentHTML\s*\(/;
  for (const [name, source] of Object.entries({ course, calculators, modelData })) assert.doesNotMatch(source, forbidden, `${name} contains forbidden runtime API`);
  assert.doesNotMatch(html, /\son[a-z]+\s*=/i);
  assert.doesNotMatch(html, /mermaid/i);
});

test('resource budgets cover prose, runtime code, and restored static diagrams independently', async () => {
  const coreFiles = ['docs/index.html', 'docs/assets/course.js', 'docs/assets/calculators.js', 'docs/assets/model-data.js', 'docs/assets/course.css', 'docs/assets/favicon.svg', 'docs/robots.txt'];
  const [coreSizes, diagramSizes] = await Promise.all([
    Promise.all(coreFiles.map((file) => stat(url(file)))),
    Promise.all(diagramFiles.map((file) => stat(url(file)))),
  ]);
  const [htmlSize, courseSize, calculatorsSize, dataSize, cssSize] = coreSizes.map((entry) => entry.size);
  assert.ok(htmlSize < 130_000);
  assert.ok(courseSize + calculatorsSize + dataSize < 50_000);
  assert.ok(cssSize < 75_000);
  assert.ok(diagramSizes.every((entry) => entry.size < 85_000));
  assert.ok(diagramSizes.reduce((sum, entry) => sum + entry.size, 0) < 850_000);
  assert.ok([...coreSizes, ...diagramSizes].reduce((sum, entry) => sum + entry.size, 0) < 1_050_000);
});

test('package verify and Pages workflow gate deployment on Node 22 verification', async () => {
  const pkg = JSON.parse(await readFile(url('package.json'), 'utf8'));
  const workflow = await readFile(url('.github/workflows/deploy.yml'), 'utf8');
  assert.equal(pkg.scripts.verify, 'npm test && npm run check');
  assert.match(workflow, /actions\/setup-node@v4/);
  assert.match(workflow, /node-version:\s*22/);
  assert.match(workflow, /run:\s*npm run verify/);
  assert.ok(workflow.indexOf('npm run verify') < workflow.indexOf('actions/upload-pages-artifact'));
});

test('README describes intentional static duplication instead of claiming all specs live once', async () => {
  const readme = await readFile(url('README.md'), 'utf8');
  assert.doesNotMatch(readme, /模型规格、缓存 shape、来源与版本只在/);
  assert.match(readme, /无 JavaScript/);
  assert.match(readme, /静态正文/);
});
