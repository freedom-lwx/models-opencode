import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, stat } from 'node:fs/promises';

const url = (path) => new URL(`../${path}`, import.meta.url);
const rawHtml = await readFile(url('docs/index.html'), 'utf8');
const html = rawHtml.replace(/<!--[\s\S]*?-->/g, '');
const course = await readFile(url('docs/assets/course.js'), 'utf8');
const css = await readFile(url('docs/assets/course.css'), 'utf8');
const modelData = await readFile(url('docs/assets/model-data.js'), 'utf8');
const calculators = await readFile(url('docs/assets/calculators.js'), 'utf8');

const segment = (start, end) => html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start)));

test('page is progressive HTML with six lessons and six model references', () => {
  for (let index = 1; index <= 6; index += 1) assert.match(html, new RegExp(`<section\\b[^>]+id="lesson-${index}"`));
  for (const id of ['nanogpt', 'minimind', 'qwen', 'glm', 'kimi', 'deepseek']) assert.match(html, new RegExp(`<article\\b[^>]+id="model-${id}"`));
  assert.doesNotMatch(html, /class="[^"]*\bsec\b[^"]*"/);
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

test('resource budgets cover HTML, JavaScript, CSS, and total static payload', async () => {
  const files = ['docs/index.html', 'docs/assets/course.js', 'docs/assets/calculators.js', 'docs/assets/model-data.js', 'docs/assets/course.css', 'docs/assets/favicon.svg', 'docs/robots.txt'];
  const sizes = await Promise.all(files.map((file) => stat(url(file))));
  const [htmlSize, courseSize, calculatorsSize, dataSize, cssSize] = sizes.map((entry) => entry.size);
  assert.ok(htmlSize < 60_000);
  assert.ok(courseSize + calculatorsSize + dataSize < 50_000);
  assert.ok(cssSize < 60_000);
  assert.ok(sizes.reduce((sum, entry) => sum + entry.size, 0) < 150_000);
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
