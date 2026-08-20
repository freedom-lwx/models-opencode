import assert from 'node:assert/strict';
import { readFile, stat } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const paths = {
  html: 'docs/index.html', css: 'docs/assets/course.css',
  data: 'docs/assets/model-data.js', calculators: 'docs/assets/calculators.js', course: 'docs/assets/course.js', icon: 'docs/assets/favicon.svg', robots: 'docs/robots.txt',
};
const contents = Object.fromEntries(await Promise.all(Object.entries(paths).map(async ([key, path]) => [key, await readFile(new URL(path, root), 'utf8')])));
const html = contents.html.replace(/<!--[\s\S]*?-->/g, '');

assert.doesNotMatch(html, /\son[a-z]+\s*=/i, 'inline event handlers are forbidden');
assert.doesNotMatch(html, /mermaid/i, 'Mermaid runtime/content must be removed');
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
const jsBytes = sizes.data + sizes.calculators + sizes.course;
const totalBytes = Object.values(sizes).reduce((sum, size) => sum + size, 0);
assert.ok(sizes.html < 60_000, `HTML budget exceeded: ${sizes.html}`);
assert.ok(jsBytes < 50_000, `JavaScript budget exceeded: ${jsBytes}`);
assert.ok(sizes.css < 60_000, `CSS budget exceeded: ${sizes.css}`);
assert.ok(totalBytes < 150_000, `total static budget exceeded: ${totalBytes}`);

console.log(`site check passed: ${ids.length} unique ids, HTML ${sizes.html} B, JS ${jsBytes} B, CSS ${sizes.css} B, total ${totalBytes} B`);
