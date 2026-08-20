import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const paths = {
  html: 'docs/index.html', css: 'docs/assets/course.css',
  data: 'docs/assets/model-data.js', calculators: 'docs/assets/calculators.js', course: 'docs/assets/course.js', icon: 'docs/assets/favicon.svg', robots: 'docs/robots.txt',
};
const diagramPaths = [
  'model-nanogpt.svg', 'model-minimind.svg', 'model-qwen.svg', 'model-glm.svg', 'model-kimi.svg', 'model-deepseek.svg',
  'system-overview.svg', 'system-glm-tp-ep.svg', 'system-kimi-tp-ep.svg', 'system-deepseek-tp-ep.svg', 'system-qwen-tp.svg', 'system-pd.svg',
].map((name) => `docs/assets/diagrams/${name}`);
const contents = Object.fromEntries(await Promise.all(Object.entries(paths).map(async ([key, path]) => [key, await readFile(new URL(path, root), 'utf8')])));
const diagrams = Object.fromEntries(await Promise.all(diagramPaths.map(async (path) => [path, await readFile(new URL(path, root), 'utf8')])));
const diagramManifest = JSON.parse(await readFile(new URL('diagrams/manifest.json', root), 'utf8'));
const sha256 = (content) => createHash('sha256').update(content).digest('hex');
const html = contents.html.replace(/<!--[\s\S]*?-->/g, '');

assert.doesNotMatch(html, /\son[a-z]+\s*=/i, 'inline event handlers are forbidden');
assert.doesNotMatch(html, /mermaid/i, 'Mermaid runtime/content must be removed from deployed HTML');
for (const [path, svg] of Object.entries(diagrams)) {
  assert.match(svg, /<svg\b[^>]+viewBox=/, `${path} needs viewBox`);
  assert.doesNotMatch(svg, /<script\b|<foreignObject\b|\son[a-z]+\s*=|(?:href|src)=["']https?:|url\(https?:/i, `${path} must be inert and local`);
  for (const cluster of svg.matchAll(/class="cluster"[^>]*><path\b([^>]*)>/g)) {
    const inlineDark = /fill:#111821!important/.test(cluster[1]);
    const stylesheetDark = /\.cluster path\{fill:#111821!important/.test(svg);
    assert.ok(inlineDark || stylesheetDark, `${path} cluster path must stay dark`);
  }
}
const expectedDiagramNames = diagramPaths.map((path) => path.match(/([^/]+)\.svg$/)[1]).sort();
assert.deepEqual(diagramManifest.files.map(({ name }) => name), expectedDiagramNames, 'diagram manifest file set');
assert.notEqual(diagramManifest.renderer.mermaidCli, 'unknown');
assert.notEqual(diagramManifest.renderer.svgo, 'unknown');
for (const entry of diagramManifest.files) {
  const source = await readFile(new URL(`diagrams/${entry.name}.mmd`, root));
  const svg = await readFile(new URL(`docs/assets/diagrams/${entry.name}.svg`, root));
  assert.equal(sha256(source), entry.sourceSha256, `${entry.name} MMD drift`);
  assert.equal(sha256(svg), entry.svgSha256, `${entry.name} SVG drift`);
  assert.equal(svg.toString('utf8').match(/viewBox="([^"]+)"/)?.[1], entry.viewBox, `${entry.name} viewBox drift`);
  const [, , viewWidth, viewHeight] = entry.viewBox.split(/\s+/).map(Number);
  const imageTag = html.match(new RegExp(`<img\\b[^>]*src="\\./assets/diagrams/${entry.name}\\.svg"[^>]*>`))?.[0] ?? '';
  const width = Number(imageTag.match(/\bwidth="(\d+)"/)?.[1]);
  const height = Number(imageTag.match(/\bheight="(\d+)"/)?.[1]);
  const aspectError = Math.abs((width / height) / (viewWidth / viewHeight) - 1);
  assert.ok(width > 0 && height > 0 && aspectError < 0.002, `${entry.name} intrinsic aspect ratio drift`);
}
assert.doesNotMatch(html, /<(?:script|link)\b[^>]+(?:src|href)=["']https?:\/\//i, 'runtime script/style must be local');
assert.doesNotMatch(html, /class=["'][^"']*\bsec\b/i, 'content must not depend on legacy hidden tabs');
assert.match(html, /<meta\b[^>]+http-equiv="Content-Security-Policy"/i, 'strict CSP meta is required');
for (const directive of ["default-src 'self'", "script-src 'self'", "style-src 'self'", "connect-src 'self'", "object-src 'none'", "base-uri 'none'"]) assert.ok(html.includes(directive), `CSP missing ${directive}`);
assert.doesNotMatch(html, /connect-src[^;]*https?:/, 'CSP connect-src must not allow third-party origins');

const ids = [...html.matchAll(/\sid=["']([^"']+)["']/g)].map((match) => match[1]);
assert.equal(new Set(ids).size, ids.length, 'HTML ids must be unique');
const idSet = new Set(ids);
for (const match of html.matchAll(/\shref=["']#([^"']+)["']/g)) assert.ok(idSet.has(match[1]), `hash target #${match[1]} must exist`);

for (const forbidden of ['25MB', '27.8GB', '89.9GB', '44.1GB', 'cdn.jsdelivr.net', '65.536 GB', '150.995 MB', '1.475 TB', '2.771 GB', '5.542 GB']) assert.ok(!html.includes(forbidden), `legacy/duplicated value remains: ${forbidden}`);

const enhancementIds = ['theme-toggle', 'flow-next', 'cache-model', 'cache-tokens', 'cache-batch', 'cache-dtype', 'cache-mode', 'comm-batch', 'comm-tokens', 'comm-hidden', 'comm-topk', 'comm-remote-fraction', 'comm-dtype', 'comm-bandwidth', 'comm-unit', 'comm-efficiency'];
for (const id of enhancementIds) assert.match(html, new RegExp(`id=["']${id}["'][^>]*\\bdisabled\\b`), `${id} must default disabled`);
for (const id of ['flow-description', 'topology-description', 'cache-result', 'communication-result']) assert.match(html, new RegExp(`id=["']${id}["'][^>]*role=["']status["']`), `${id} needs status semantics`);
assert.match(html, /<progress\b[^>]+id="course-progress"[^>]+max="100"[^>]+value="0"/);

const dangerous = /\b(?:fetch|XMLHttpRequest|WebSocket|EventSource|sendBeacon|document\.write)\b|\.(?:innerHTML|outerHTML)\s*=|insertAdjacentHTML\s*\(/;
for (const key of ['data', 'calculators', 'course']) assert.doesNotMatch(contents[key], dangerous, `${key} contains forbidden runtime API`);
assert.match(contents.css, /prefers-reduced-motion/, 'reduced motion support is required');
assert.match(contents.css, /prefers-color-scheme:\s*light/, 'no-JS system theme is required');

const sizes = Object.fromEntries(await Promise.all(Object.entries(paths).map(async ([key, path]) => [key, (await stat(new URL(path, root))).size])));
const diagramSizes = await Promise.all(diagramPaths.map(async (path) => (await stat(new URL(path, root))).size));
const jsBytes = sizes.data + sizes.calculators + sizes.course;
const diagramBytes = diagramSizes.reduce((sum, size) => sum + size, 0);
const totalBytes = Object.values(sizes).reduce((sum, size) => sum + size, 0) + diagramBytes;
assert.ok(sizes.html < 130_000, `HTML budget exceeded: ${sizes.html}`);
assert.ok(jsBytes < 50_000, `JavaScript budget exceeded: ${jsBytes}`);
assert.ok(sizes.css < 75_000, `CSS budget exceeded: ${sizes.css}`);
assert.ok(diagramSizes.every((size) => size < 85_000), `single SVG budget exceeded: ${Math.max(...diagramSizes)}`);
assert.ok(diagramBytes < 850_000, `diagram budget exceeded: ${diagramBytes}`);
assert.ok(totalBytes < 1_050_000, `total static budget exceeded: ${totalBytes}`);

console.log(`site check passed: ${ids.length} unique ids, HTML ${sizes.html} B, JS ${jsBytes} B, CSS ${sizes.css} B, SVG ${diagramBytes} B, total ${totalBytes} B`);
