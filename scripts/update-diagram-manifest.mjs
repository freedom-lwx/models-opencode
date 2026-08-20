import { createHash } from 'node:crypto';
import { readdir, readFile, writeFile } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const sourceDir = new URL('diagrams/', root);
const outputDir = new URL('docs/assets/diagrams/', root);
const manifestURL = new URL('diagrams/manifest.json', root);
const sha256 = (content) => createHash('sha256').update(content).digest('hex');

const names = (await readdir(sourceDir))
  .filter((name) => name.endsWith('.mmd'))
  .map((name) => name.slice(0, -4))
  .sort();

const files = await Promise.all(names.map(async (name) => {
  const [source, svg] = await Promise.all([
    readFile(new URL(`${name}.mmd`, sourceDir)),
    readFile(new URL(`${name}.svg`, outputDir)),
  ]);
  const viewBox = svg.toString('utf8').match(/viewBox="([^"]+)"/)?.[1];
  if (!viewBox) throw new Error(`${name}.svg has no viewBox`);
  return { name, sourceSha256: sha256(source), svgSha256: sha256(svg), viewBox };
}));

const manifest = {
  renderer: {
    mermaidCli: process.argv[2] || 'unknown',
    svgo: process.argv[3] || 'unknown',
  },
  files,
};
await writeFile(manifestURL, `${JSON.stringify(manifest, null, 2)}\n`);
